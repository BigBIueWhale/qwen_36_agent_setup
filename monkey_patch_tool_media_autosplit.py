"""Strict, fail-loud server-side auto-split of tool-role media into a follow-up user message.

Why this patch must exist
-------------------------

vLLM's chat-completion ingest at ``vllm/entrypoints/chat_utils.py``
lines 1549-1564 has a ``role == "tool"`` branch that flattens any
list-shaped tool message content to ``"\\n".join(text_items)`` — every
non-text part (``image_url``, ``audio_url``, ``video_url``, ``file``,
``input_image``, ``image_pil``, ``image_embeds``, ``audio_embeds``,
``input_audio``) is silently discarded before the chat template ever
runs. Concrete shape::

    elif role == "tool":
        ...
        msg_content = result_msg.get("content")
        if isinstance(msg_content, list):
            texts = [
                item.get("text", "")
                for item in msg_content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            result_msg["content"] = "\\n".join(texts) if texts else ""

The drop is silent (no warning, no refusal) and load-bearing for the
agentic vision pipeline this deployment was built for: MCP-returned
screenshots, ``read_file`` -emitted images, and any tool-call that
returns mixed text+image disappears from the model's prompt while the
client believes the round-trip succeeded.

Worse, the silent drop also breaks vLLM's own internal accounting: the
``MultiModalItemTracker`` already registered each media part inside
``_parse_chat_message_content_part`` (the per-content-part dispatcher
upstream of the role-gate), so the encoder runs on data the rendered
prompt never references — silent encoder/template desync. This is a
class A internal-inconsistency bug akin to §6.1 (the ingest path's
"what counts as reasoning" disagreement with the chat template's
notion of the same).

Companion (client-side) workaround: qwen-code commit 414b330 (PR #3617)
added a ``splitToolMedia: true`` setting that mirrors this transform on
the client. Defaults to ``false``, so out-of-the-box agents on
Qwen-Code, OpenAI Python SDK, the OpenAI-compatible TypeScript SDKs,
and Qwen-Agent all hit the silent-drop path. Closing the bug
server-side closes it for ALL clients regardless of the per-client flag
state.

What the patch does
-------------------

Wraps the **outer funnel** ``parse_chat_messages`` and
``parse_chat_messages_async`` (both at
``vllm/entrypoints/chat_utils.py``) — NOT the per-message
``_parse_chat_message_content`` (which is patch 1's target). The
wrapper runs a pre-pass over the messages list that splits any
``role == "tool"`` message containing media parts into:

1. **The original tool message**, content reduced to its text-only
   parts. ``tool_call_id`` is preserved (the chat template uses it for
   linkage), and the message stays at its existing position so the
   chat template still renders it inside ``<tool_response>...
   </tool_response>``.
2. **A synthetic follow-up ``role: "user"`` message** containing ONLY
   the media parts, inserted immediately after the tool message in
   the messages list. No synthetic text shim is added — qwen-code's
   shim is a strict-OpenAI workaround, not a model-quality
   requirement; Qwen3.6's chat template (``chat_template.jinja``)
   renders media-only user content cleanly.

Chat-template invariants this preserves
---------------------------------------

* ``chat_template.jinja:33`` raises ``'Unexpected item type in
  content.'`` if a tool message contains non-text content. Our split
  ensures the tool message is text-only by the time the template runs.
* ``chat_template.jinja:67-77``'s ``multi_step_tool`` reverse-walk
  identifies the last user query by checking whether the rendered
  content starts with ``<tool_response>``. The follow-up user message
  carries media-only content, so its rendered content does NOT start
  with ``<tool_response>`` — the reverse-walk correctly identifies it
  as a real user query, exactly the path Qwen3.6 was trained on.
* The ``MultiModalItemTracker`` registration moves with the media
  parts (they are still parsed downstream by patch 1's wrapper as
  parts of the user message), so encoder/template alignment is
  restored.

Why the OUTER funnel and not the per-message function
-----------------------------------------------------

The transform produces 1→2 messages. Wrapping
``_parse_chat_message_content`` (the per-message function) would
require either (a) returning a list of two ``ConversationMessage``
results from a function whose contract is one input → one
``list[ConversationMessage]`` output for ONE input message (clean,
but every caller would have to be re-audited for whether it expects
1:1 message:result correspondence), or (b) a thread-local stash of
"deferred messages to inject after this one" (state-ful, fragile).
The outer funnels (``parse_chat_messages``,
``parse_chat_messages_async``) iterate over a Python list of message
dicts and concatenate per-message results into a single
``conversation`` list. Pre-splitting that list on the way in costs
one shallow-copy per affected tool message and composes cleanly with
patch 1's per-message wrapper (which sees the already-split user
message and processes it like any other media-bearing user input).

Removal trigger
---------------

vLLM upstream rewrites the role:"tool" content reducer at
``vllm/entrypoints/chat_utils.py:1549-1564`` to preserve media (or
hoist the media into a follow-up message itself). The buggy
``'"\\n".join(texts) if texts else ""'`` predicate is the landmark
this patch asserts on; any reshape changes its body and forces this
patch to refuse via :class:`MonkeyPatchRefusedError`.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.entrypoints import chat_utils as _chat_utils_mod


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-tool-media-autosplit-v1"

# Source landmarks — substrings required in the buggy reducer's body
# inside ``_parse_chat_message_content`` so an upstream refactor of the
# role:"tool" path surfaces a typed refusal at install time rather than
# silently coexisting with our outer wrapper.
_LANDMARK_BUGGY_REDUCER: str = '"\\n".join(texts) if texts else ""'
_LANDMARK_ROLE_GATE: str = 'elif role == "tool":'
_LANDMARK_LIST_GUARD: str = "if isinstance(msg_content, list):"

# Names of the outer funnels we wrap. Both call
# ``_parse_chat_message_content`` per-message; we splice the tool-media
# split BEFORE that loop. Order must match: the sync funnel first
# (matches the source order at chat_utils.py:1600,1636), the async
# funnel second.
_FUNNEL_NAMES: tuple[str, ...] = ("parse_chat_messages", "parse_chat_messages_async")

# OpenAI content-part type strings we consider media (split into a
# follow-up role:"user" message). Anything not in this set OR
# ``_TEXT_TYPES`` is treated as text-ish and stays with the tool
# message — let downstream code raise on truly unknown shapes.
_MEDIA_TYPES: frozenset[str] = frozenset({
    "image_url", "input_image", "image_pil", "image_embeds",
    "audio_url", "input_audio", "audio_embeds",
    "video_url", "file",
})
_TEXT_TYPES: frozenset[str] = frozenset({
    "text", "input_text", "output_text", "refusal", "thinking",
})


_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class MonkeyPatchRefusedError(RuntimeError):
    """Precondition for the tool-media auto-split patch was violated.

    Raised at import time only; the patch either applies cleanly or
    the process does not come up. Same idiom as the other Qwen3.6
    patches.
    """


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise MonkeyPatchRefusedError(f"[{_PATCH_TAG}] refusing to patch: {msg}")


# ---------------------------------------------------------------------------
# Phase 1 — landmark the buggy reducer in _parse_chat_message_content.
# ---------------------------------------------------------------------------

# Read the source of the per-message function (patch 1's target — we
# don't wrap it, but its current shape is the reason this patch must
# exist). If a sibling patch has already wrapped it, walk through the
# ``__wrapped_original__`` chain to landmark the original body.
_inner_fn = getattr(_chat_utils_mod, "_parse_chat_message_content", None)
_require(
    _inner_fn is not None and callable(_inner_fn),
    "_parse_chat_message_content missing from vllm.entrypoints.chat_utils.",
)

_inner_to_inspect: Any = _inner_fn
_walk_guard = 0
while getattr(_inner_to_inspect, "__qwen36_patch__", None) is not None:
    _walk_guard += 1
    _require(
        _walk_guard < 16,
        "patch chain on _parse_chat_message_content exceeds depth limit "
        "(16); refusing landmark check on a runaway wrapper stack.",
    )
    _next = getattr(_inner_to_inspect, "__wrapped_original__", None)
    if _next is None:
        break
    _inner_to_inspect = _next

try:
    _inner_src = inspect.getsource(_inner_to_inspect)
except (TypeError, ValueError, OSError) as _exc:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect _parse_chat_message_content "
        f"original: {_exc!r}"
    ) from _exc

_require(
    _LANDMARK_BUGGY_REDUCER in _inner_src,
    f"buggy reducer landmark {_LANDMARK_BUGGY_REDUCER!r} not found in "
    f"_parse_chat_message_content source; upstream may have already "
    f"fixed the silent-drop bug — re-audit before patching.",
)
_require(
    _LANDMARK_ROLE_GATE in _inner_src,
    f"role gate landmark {_LANDMARK_ROLE_GATE!r} missing in "
    f"_parse_chat_message_content source; the function shape has "
    f"shifted upstream.",
)
_require(
    _LANDMARK_LIST_GUARD in _inner_src,
    f"list-guard landmark {_LANDMARK_LIST_GUARD!r} missing in "
    f"_parse_chat_message_content source; the function shape has "
    f"shifted upstream.",
)


# ---------------------------------------------------------------------------
# Phase 2 — locate the funnels we will wrap and assert their signatures.
# ---------------------------------------------------------------------------

_originals: dict[str, Callable[..., Any]] = {}
for _name in _FUNNEL_NAMES:
    _fn = getattr(_chat_utils_mod, _name, None)
    _require(
        _fn is not None and callable(_fn),
        f"{_name} missing from vllm.entrypoints.chat_utils.",
    )
    # Refuse if a previous (different) patch already installed itself
    # here — this isolates load-order regressions before they corrupt
    # state. A rerun of THIS patch is permitted (idempotent re-install
    # short-circuits at install time below).
    _existing_tag = getattr(_fn, "__qwen36_patch__", None)
    if _existing_tag not in (None, _PATCH_TAG):
        raise MonkeyPatchRefusedError(
            f"[{_PATCH_TAG}] {_name} is already wrapped by "
            f"{_existing_tag!r}; cannot install over a sibling patch."
        )
    # Confirm both funnels have the expected first parameter name —
    # our wrapper passes ``messages`` positionally, so a rename
    # would silently shift arguments.
    try:
        _sig = inspect.signature(_fn)
    except (TypeError, ValueError) as _exc:
        raise MonkeyPatchRefusedError(
            f"[{_PATCH_TAG}] cannot introspect {_name}: {_exc!r}"
        ) from _exc
    _params = list(_sig.parameters)
    _require(
        len(_params) >= 1 and _params[0] == "messages",
        f"{_name} expected first param 'messages', got {_params!r}.",
    )
    _originals[_name] = _fn


# ---------------------------------------------------------------------------
# Phase 3 — pure transformation: split tool-role media into follow-up user.
# ---------------------------------------------------------------------------

def _split_tool_media_in_messages(messages: Any) -> Any:
    """Pre-ingest pass; idempotent. Returns a NEW list when any split
    happens, or passes the input through by identity when no split is
    needed. Unmodified messages are passed through by identity even
    inside a freshly-allocated list; only tool messages that get split
    are shallow-copied.

    Rules:

    * non-list ``messages`` → passed through unchanged (let downstream
      raise on the wrong type).
    * ``role != "tool"`` → message passes through unchanged.
    * ``role == "tool"`` with non-list content → unchanged (already a
      string; legacy clients and ``splitToolMedia=true`` clients hit
      this path and need no further work).
    * ``role == "tool"`` with list content but no media parts →
      unchanged.
    * ``role == "tool"`` with list content AND at least one media part
      → emit two messages: the tool message with media stripped from
      its content, followed by a synthetic ``role: "user"`` message
      carrying ONLY the media parts (no text shim — let the chat
      template render the media markers directly).
    """
    if not isinstance(messages, list):
        return messages
    out: list[Any] = []
    any_split = False
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "tool":
            out.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue  # idempotent no-op for already-string content
        text_parts: list[Any] = []
        media_parts: list[Any] = []
        for p in content:
            if not isinstance(p, dict):
                # Bare strings or unknown shapes — keep with the tool
                # message; downstream parsing handles them.
                text_parts.append(p)
                continue
            t = p.get("type")
            if t in _MEDIA_TYPES:
                media_parts.append(p)
            else:
                # _TEXT_TYPES and any unknown text-ish part stay with
                # the tool message.
                text_parts.append(p)
        if not media_parts:
            out.append(m)
            continue  # no-op; pass original by identity
        any_split = True
        new_tool = dict(m)
        new_tool["content"] = text_parts
        out.append(new_tool)
        out.append({"role": "user", "content": list(media_parts)})
    return out if any_split else messages


# ---------------------------------------------------------------------------
# Phase 4 — primitive verify on synthetic message lists.
# ---------------------------------------------------------------------------

# Case A: tool with text + image_url → splits into [tool(text-only),
# user(image)] preserving tool_call_id.
_probe_a_in = [
    {"role": "user", "content": "look at this"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "fetch_image", "arguments": "{}"},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [
            {"type": "text", "text": "Here is the image:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXXX"}},
        ],
    },
]
_probe_a_out = _split_tool_media_in_messages(_probe_a_in)
_require(
    isinstance(_probe_a_out, list) and len(_probe_a_out) == 4,
    f"Phase 4 case A: expected 4 messages after split, got "
    f"{len(_probe_a_out) if isinstance(_probe_a_out, list) else type(_probe_a_out).__name__}.",
)
_require(
    _probe_a_out[2].get("role") == "tool"
    and _probe_a_out[2].get("tool_call_id") == "c1"
    and _probe_a_out[2].get("content") == [
        {"type": "text", "text": "Here is the image:"}
    ],
    "Phase 4 case A: tool message did not retain text-only content "
    "with tool_call_id preserved.",
)
_require(
    _probe_a_out[3].get("role") == "user"
    and _probe_a_out[3].get("content")
    == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,XXXX"}}],
    "Phase 4 case A: follow-up user message did not carry the image "
    "part exactly once.",
)
_require(
    _probe_a_out[0] is _probe_a_in[0] and _probe_a_out[1] is _probe_a_in[1],
    "Phase 4 case A: non-tool messages preceding the split lost identity "
    "(unnecessary copy in the no-op path).",
)

# Case B: tool with image_url only → tool gets empty list; user gets
# the image. tool_call_id preserved.
_probe_b_in = [
    {
        "role": "tool",
        "tool_call_id": "c2",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,YYYY"}},
        ],
    },
]
_probe_b_out = _split_tool_media_in_messages(_probe_b_in)
_require(
    isinstance(_probe_b_out, list)
    and len(_probe_b_out) == 2
    and _probe_b_out[0].get("content") == [],
    "Phase 4 case B: media-only tool message did not get empty content list.",
)
_require(
    _probe_b_out[0].get("tool_call_id") == "c2",
    "Phase 4 case B: tool_call_id was not preserved on the split tool message.",
)
_require(
    _probe_b_out[1].get("role") == "user"
    and isinstance(_probe_b_out[1].get("content"), list)
    and len(_probe_b_out[1].get("content", [])) == 1,
    "Phase 4 case B: follow-up user did not carry exactly the one image part.",
)

# Case C: tool with text only → no split; passes through by identity.
_probe_c_in = [
    {"role": "tool", "tool_call_id": "c3", "content": [{"type": "text", "text": "ok"}]}
]
_probe_c_out = _split_tool_media_in_messages(_probe_c_in)
_require(
    _probe_c_out is _probe_c_in,
    "Phase 4 case C: text-only tool message did not pass the WHOLE "
    "messages list through by identity (idempotency broken on the "
    "no-op path).",
)

# Case D: tool with string content (post-splitToolMedia client) →
# identity. Confirms idempotency under double-application.
_probe_d_in = [{"role": "tool", "tool_call_id": "c4", "content": "already a string"}]
_probe_d_out = _split_tool_media_in_messages(_probe_d_in)
_require(
    _probe_d_out is _probe_d_in,
    "Phase 4 case D: string-content tool message did not pass the "
    "WHOLE messages list through by identity.",
)

# Case E: non-tool messages with media → unchanged (we only touch role:tool).
_probe_e_in = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZZZ"}},
            {"type": "text", "text": "hi"},
        ],
    },
]
_probe_e_out = _split_tool_media_in_messages(_probe_e_in)
_require(
    _probe_e_out is _probe_e_in,
    "Phase 4 case E: user-role message with media was incorrectly "
    "modified; only role:'tool' messages are in remit.",
)

# Idempotency check: applying the transform twice on the same input
# yields a result equal to applying it once. (Apply-once on the result
# of apply-once must be a no-op; the second pass sees text-only tool
# messages and a media-only user message, so nothing splits.)
_probe_idem_first = _split_tool_media_in_messages(_probe_a_in)
_probe_idem_second = _split_tool_media_in_messages(_probe_idem_first)
_require(
    _probe_idem_second is _probe_idem_first,
    "Phase 4 idempotency: second application of the transform produced "
    "a different list object — re-runs are not free.",
)


# ---------------------------------------------------------------------------
# Phase 5 — build wrappers around the funnels (sync + async).
# ---------------------------------------------------------------------------

def _make_wrapper(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    is_async = name.endswith("_async")
    if is_async:
        @functools.wraps(original)
        async def wrapper_async(messages, *args, **kwargs):
            return await original(
                _split_tool_media_in_messages(messages), *args, **kwargs
            )

        wrapper_async.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
        wrapper_async.__wrapped_original__ = original  # type: ignore[attr-defined]
        return wrapper_async

    @functools.wraps(original)
    def wrapper_sync(messages, *args, **kwargs):
        return original(_split_tool_media_in_messages(messages), *args, **kwargs)

    wrapper_sync.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
    wrapper_sync.__wrapped_original__ = original  # type: ignore[attr-defined]
    return wrapper_sync


_wrappers: dict[str, Callable[..., Any]] = {
    name: _make_wrapper(name, original) for name, original in _originals.items()
}


# ---------------------------------------------------------------------------
# Phase 6 — install + verify (both dynamic and static lookups agree).
# ---------------------------------------------------------------------------

for _install_name, _install_wrapper in _wrappers.items():
    setattr(_chat_utils_mod, _install_name, _install_wrapper)
    _dyn = getattr(_chat_utils_mod, _install_name, None)
    _static = inspect.getattr_static(_chat_utils_mod, _install_name, None)
    _require(
        _dyn is _install_wrapper,
        f"dynamic lookup of vllm.entrypoints.chat_utils."
        f"{_install_name} does not see the patched wrapper.",
    )
    _require(
        _static is _install_wrapper,
        f"static lookup of vllm.entrypoints.chat_utils."
        f"{_install_name} does not see the patched wrapper "
        f"(metaclass shim or re-import racing).",
    )
    _require(
        getattr(_dyn, "__qwen36_patch__", None) == _PATCH_TAG,
        f"vllm.entrypoints.chat_utils.{_install_name} carries "
        f"unexpected tag {getattr(_dyn, '__qwen36_patch__', None)!r}.",
    )
    _require(
        getattr(_dyn, "__wrapped_original__", None) is _originals[_install_name],
        f"vllm.entrypoints.chat_utils.{_install_name} bears tag but "
        f"__wrapped_original__ is not the original — forging defense "
        f"failed.",
    )


_logger.info(
    "[%s] applied: wrapped vllm.entrypoints.chat_utils.{%s} for vLLM "
    "commit %s (pre-ingest auto-split of role:'tool' messages with "
    "media into a follow-up role:'user' message; idempotent, "
    "preserves tool_call_id, mirrors qwen-code PR #3617 server-side).",
    _PATCH_TAG,
    ", ".join(_FUNNEL_NAMES),
    _PINNED_VLLM_COMMIT,
)
