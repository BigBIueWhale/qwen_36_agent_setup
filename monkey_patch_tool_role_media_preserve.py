"""Strict, fail-loud server-side preserve-tool-role-media ingest patch.

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
prompt never references — silent encoder/template desync. Class A
internal-inconsistency bug akin to §6.1 (the ingest path's "what
counts as reasoning" disagreement with the chat template's notion of
the same).

The discrepancy audit on 2026-04-28 verified that the previous
auto-split design produced a wire shape that diverged from the
released Qwen3-VL-8B-Instruct training distribution in three concrete
ways.

1. The split-into-follow-up-user message put image markers OUTSIDE
   ``<tool_response>...</tool_response>``, breaking the
   tool-result→image attribution that training data shaped (the
   released chat template's ``role == "tool"`` branch emits image
   markers INSIDE ``<tool_response>``).
2. Empty ``<tool_response></tool_response>`` shells appeared whenever
   a tool result was media-only — never seen in training.
3. The synthetic media-only user message became the chat template's
   "last user query" via the reverse-walk at
   ``chat_template.jinja:67-77``, biasing the model away from
   training-time semantics where the last query was textual.

Plus a parallel-tool-calls contiguity bug: the auto-split inserted a
user turn between every tool message in a multi-tool round, while the
canonical reference algorithm (qwen-code PR #3617) accumulates media
into ONE follow-up user after the LAST tool message.

What the patch does
-------------------

The released Qwen3-VL-8B-Instruct chat template (the canonical
artifact representing training-time conversation shape) has an
explicit ``role == "tool"`` branch that emits
``<|vision_start|><|image_pad|><|vision_end|>`` markers **INSIDE**
the ``<tool_response>`` block, by passing the tool message's content
list to the same ``render_content`` macro used for user/system
messages.

Our QuantTrio chat template at ``chat_template.jinja:131-142`` calls
``render_content(message.content, true)`` on tool messages — and
``render_content`` at lines 3-41 already renders image markers
correctly when ``content`` is a list with image parts. **The
infrastructure for the training shape is already in our chat
template.** The shipped vLLM is the only thing in the way, because it
flattens list content to a string before the template ever runs.

The patch wraps the **per-message** ``_parse_chat_message_content``
inner function (NOT the outer ``parse_chat_messages`` /
``parse_chat_messages_async`` funnels) and:

1. When ``role == "tool"`` AND content is a list AND any element has
   a known media type, snapshot a list-shaped, type-filtered copy of
   the content BEFORE the original runs.
2. Let the original ``_parse_chat_message_content`` run unchanged
   (so ``MultiModalItemTracker`` registrations and ``tool_call_id``
   linkage flow through the upstream path exactly as before — they
   are identical in the patched and unpatched cases).
3. Replace the resulting ``ConversationMessage``'s flattened-string
   ``content`` with the saved list. The chat template then renders
   ``<|vision_start|><|image_pad|><|vision_end|>`` natively inside
   ``<tool_response>`` — exactly the training-distribution shape.

This eliminates ALL of: the synthetic user message, the
parallel-tool-call contiguity bug, the empty-shell case, and the
text-shim question. Strictly more elegant: one wrap target instead
of two, no message-list splicing, no synthetic dicts.

Composition with patch 1
------------------------

Patch 1 (``monkey_patch_reasoning_field_ingest.py``) ALREADY wraps
``_parse_chat_message_content`` for the ``role == "assistant"``
branch. Our wrapper guards on ``role == "tool"``. The two patches
touch DISJOINT role branches; composition is clean regardless of
install order. The ``__wrapped_original__`` chain walk in Phase 1
handles arbitrarily-nested patch chains so the landmark check
operates against the original-original body.

Removal trigger
---------------

vLLM upstream rewrites the ``role:"tool"`` content reducer at
``vllm/entrypoints/chat_utils.py:1549-1564`` to preserve list
content. The buggy ``'"\\n".join(texts) if texts else ""'``
predicate is the landmark this patch asserts on; any reshape changes
its body and forces this patch to refuse via
:class:`MonkeyPatchRefusedError`.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.entrypoints import chat_utils as _chat_utils_mod


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-tool-role-media-preserve-v1"

# Source landmarks — substrings required in the buggy reducer's body
# inside ``_parse_chat_message_content`` so an upstream refactor of the
# role:"tool" path surfaces a typed refusal at install time rather than
# silently coexisting with our preserve wrapper. The landmarks are the
# SAME the deleted auto-split patch landmarked: we are not changing the
# bug we anchor to — we are changing what we do about it. We let the
# original run, and overwrite its lossy output with a structure-
# preserved one.
_LANDMARK_BUGGY_REDUCER: str = '"\\n".join(texts) if texts else ""'
_LANDMARK_ROLE_GATE: str = 'elif role == "tool":'
_LANDMARK_LIST_GUARD: str = "if isinstance(msg_content, list):"

# Exact parameter list we verified against the pinned commit. Patch 1
# (reasoning_field_ingest) wraps the same target with the same
# signature; if either patch's expected signature drifts, BOTH would
# refuse. Drift means the wrapper's positional/keyword contract with
# the original has broken and silent arg-shifts become possible.
_EXPECTED_PARAMS: list[str] = [
    "message",
    "mm_tracker",
    "content_format",
    "interleave_strings",
    "mm_processor_kwargs",
]

# OpenAI/vLLM content-part type strings the chat template's
# ``render_content`` macro at chat_template.jinja:3-41 dispatches on.
# Membership in this set is the GATE for inclusion in the preserved
# list — any unknown type would trigger the chat template's
# ``Unexpected item type in content.`` raise at line 33 once the list
# survives flattening.
#
# Cross-checked against vLLM's ``MM_PARSER_MAP`` at chat_utils.py:1260
# AND ``_parse_chat_message_content_part`` dispatch at chat_utils.py:
# 1424+. We include only types that exist in BOTH the chat template's
# render path AND vLLM's part-parser. Anything else is dropped from the
# preserved list (it would have been silently dropped or raised by the
# upstream code anyway; explicit is better than implicit).
_TEXT_PART_TYPES: frozenset[str] = frozenset({
    # chat_template.jinja:30 dispatches on `'text' in item` — items
    # with these types all carry an item.text field after upstream
    # parsing.
    "text",
    "input_text",
    "output_text",
    "refusal",
    "thinking",
})
_IMAGE_PART_TYPES: frozenset[str] = frozenset({
    # chat_template.jinja:8 dispatches on `'image' in item or
    # 'image_url' in item or item.type == 'image'`. vLLM's
    # _parse_chat_message_content_part wraps each into a `{"type":
    # "image"}` placeholder when wrap_dicts=True; we keep the original
    # part type so the chat template's `'image_url' in item` short-
    # circuit also fires for clients that send the raw image_url
    # shape.
    "image",
    "image_url",
    "input_image",
    "image_pil",
    "image_embeds",
})
_VIDEO_PART_TYPES: frozenset[str] = frozenset({
    # chat_template.jinja:19 dispatches on `'video' in item or
    # item.type == 'video'`.
    "video",
    "video_url",
    "video_embeds",
})
_MEDIA_PART_TYPES: frozenset[str] = (
    _IMAGE_PART_TYPES | _VIDEO_PART_TYPES
)
_KNOWN_PART_TYPES: frozenset[str] = (
    _TEXT_PART_TYPES | _MEDIA_PART_TYPES
)


_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class MonkeyPatchRefusedError(RuntimeError):
    """Precondition for the tool-role-media-preserve patch was violated.

    Raised at import time only; the patch either applies cleanly or
    the process does not come up. Same idiom as the other Qwen3.6
    patches.
    """


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise MonkeyPatchRefusedError(f"[{_PATCH_TAG}] refusing to patch: {msg}")


# ---------------------------------------------------------------------------
# Phase 1 — locate _parse_chat_message_content; walk patch chain to landmark
# the original body.
# ---------------------------------------------------------------------------

_inner_fn = getattr(_chat_utils_mod, "_parse_chat_message_content", None)
_require(
    _inner_fn is not None and callable(_inner_fn),
    "_parse_chat_message_content missing from vllm.entrypoints.chat_utils.",
)

# Walk the ``__wrapped_original__`` chain in case patch 1 (or any
# sibling) is already installed at this target. The landmark check
# must operate on the ORIGINAL function body, not on a wrapper that
# does not contain the source we are anchoring to. Bounded by depth
# 16 to surface a runaway wrapper stack as a typed refusal.
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
# Phase 2 — signature check on the function (against the live attr, not
# against the unwrapped landmark target — patch 1's wrapper exposes the
# same param list via functools.wraps + explicit __name__ stamp, so
# inspect.signature here returns the contract we wrap against).
# ---------------------------------------------------------------------------

try:
    _sig = inspect.signature(_inner_fn)
except (TypeError, ValueError) as _exc:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect signature of "
        f"_parse_chat_message_content: {_exc!r}"
    ) from _exc

_param_names = list(_sig.parameters)
_require(
    _param_names == _EXPECTED_PARAMS,
    f"_parse_chat_message_content signature drifted; expected "
    f"{_EXPECTED_PARAMS!r}, got {_param_names!r}.",
)


# ---------------------------------------------------------------------------
# Phase 3 — pure transformation: filter list content, preserve text +
# media in original order; signal "no intervention" when the message is
# not in remit.
# ---------------------------------------------------------------------------

def _filter_tool_content_preserving_media(content: Any) -> list[dict] | None:
    """Decide whether to override the original's flattened-string output
    with a structured list, and if so, what list to substitute.

    Contract:

    * non-list content (string, None) → return ``None``. Signals "no
      intervention"; let the original return its native shape.
    * list content with NO media parts → return ``None``. Either the
      list is text-only (the original's text-flatten output is fine)
      or the list contains only types neither we nor the chat template
      know about (which would have been dropped by the original's
      flatten anyway).
    * list content with at least one media part → return a NEW list
      filtered to known TEXT + MEDIA types only, preserving original
      order. Items of unknown type are dropped (the chat template
      would raise on them at line 33; we treat the strictest possible
      contract as the safest).

    The returned list is ALWAYS a fresh list (not the caller's
    reference); each item is the SAME dict reference the caller
    provided (no copy of inner dicts — the preserved structure is
    read-only downstream).
    """
    if not isinstance(content, list):
        return None
    has_media = False
    filtered: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            # Bare strings and unknown shapes have no role in the
            # chat-template's list-mode rendering. Drop them; the
            # original would have flattened them out too.
            continue
        part_type = part.get("type")
        # Keys-without-type fallback path matching the chat template's
        # `'image' in item or 'image_url' in item` / `'video' in item`
        # / `'text' in item` shorthands at lines 8, 19, 30. Required so
        # clients that send the simple-image shape (raw `image_url`
        # key, no `type` field) still get media-preserve behaviour.
        is_text = part_type in _TEXT_PART_TYPES or "text" in part
        is_image = (
            part_type in _IMAGE_PART_TYPES
            or "image" in part
            or "image_url" in part
        )
        is_video = (
            part_type in _VIDEO_PART_TYPES
            or "video" in part
            or "video_url" in part
        )
        if is_image or is_video:
            has_media = True
            filtered.append(part)
            continue
        if is_text:
            filtered.append(part)
            continue
        # Unknown type — drop. The chat template would raise on it at
        # line 33 (`Unexpected item type in content.`); we honor the
        # strictest contract.
    if not has_media:
        return None  # text-only or empty → no intervention
    return filtered


# ---------------------------------------------------------------------------
# Phase 4 — behavioural probes A-E directly testing
# _filter_tool_content_preserving_media. Load-bearing — the install
# verifier (Phase 6) is tag-only; the wrapper's correctness lives here.
# ---------------------------------------------------------------------------

# Case A: text + image_url → returns [text_dict, image_dict] in order.
_probe_a_in = [
    {"type": "text", "text": "Here is the image:"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXXX"}},
]
_probe_a_out = _filter_tool_content_preserving_media(_probe_a_in)
_require(
    isinstance(_probe_a_out, list)
    and len(_probe_a_out) == 2
    and _probe_a_out[0] is _probe_a_in[0]
    and _probe_a_out[1] is _probe_a_in[1],
    f"Phase 4 case A: text+image content did not pass through preserved "
    f"order with original dict identity. Got {_probe_a_out!r}.",
)

# Case B: image_url only → returns [image_dict].
_probe_b_in = [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,YYYY"}},
]
_probe_b_out = _filter_tool_content_preserving_media(_probe_b_in)
_require(
    isinstance(_probe_b_out, list)
    and len(_probe_b_out) == 1
    and _probe_b_out[0] is _probe_b_in[0],
    f"Phase 4 case B: image-only content did not yield single-element "
    f"preserved list. Got {_probe_b_out!r}.",
)

# Case C: text only → returns None (no intervention; let original's
# flatten run).
_probe_c_in = [{"type": "text", "text": "ok"}]
_probe_c_out = _filter_tool_content_preserving_media(_probe_c_in)
_require(
    _probe_c_out is None,
    f"Phase 4 case C: text-only content should yield None (no "
    f"intervention), got {_probe_c_out!r}.",
)

# Case D: string content (already flattened, e.g. from a
# splitToolMedia=true client) → returns None.
_probe_d_out = _filter_tool_content_preserving_media("already flat")
_require(
    _probe_d_out is None,
    f"Phase 4 case D: string content should yield None, got "
    f"{_probe_d_out!r}.",
)

# Case E: unknown type — TWO sub-cases.
# E1: unknown + text only (no media) → None (the unknown is just
# noise; without media the original's flatten output is fine).
_probe_e1_in = [
    {"type": "text", "text": "hello"},
    {"type": "mystery_widget", "data": "..."},
]
_probe_e1_out = _filter_tool_content_preserving_media(_probe_e1_in)
_require(
    _probe_e1_out is None,
    f"Phase 4 case E1 (unknown+text, no media): expected None, got "
    f"{_probe_e1_out!r}.",
)
# E2: unknown + text + image → [text, image] (unknown filtered out;
# media triggers preservation).
_probe_e2_in = [
    {"type": "text", "text": "hello"},
    {"type": "mystery_widget", "data": "..."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZZZ"}},
]
_probe_e2_out = _filter_tool_content_preserving_media(_probe_e2_in)
_require(
    isinstance(_probe_e2_out, list)
    and len(_probe_e2_out) == 2
    and _probe_e2_out[0] is _probe_e2_in[0]
    and _probe_e2_out[1] is _probe_e2_in[2],
    f"Phase 4 case E2 (unknown+text+image): expected [text, image] "
    f"with unknown filtered, got {_probe_e2_out!r}.",
)


# ---------------------------------------------------------------------------
# Phase 5 — build the wrapper.
# ---------------------------------------------------------------------------
#
# The wrapper signature must match the parameter list the original
# accepts. Patch 1's wrapper (already at this target if installed)
# uses positional+default for ``mm_processor_kwargs``; we mirror that
# shape to keep the call-site contract identical regardless of which
# wrapper executes first.

_original = _inner_fn  # Whatever the patch sees at install time —
# patch 1's wrapper if it is installed, else the original. Both honour
# the contract; we delegate to whichever it is.


@functools.wraps(_original)
def _parse_chat_message_content_with_tool_media_preserved(
    message: Any,
    mm_tracker: Any,
    content_format: Any,
    interleave_strings: bool,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Wrapper around ``_parse_chat_message_content``. For
    ``role == "tool"`` messages with list-shaped media-bearing content,
    snapshot a filtered preserve-list BEFORE calling the original,
    then restore that list onto the returned ``ConversationMessage``'s
    ``content`` field — replacing the original's lossy flattened
    string. All other messages pass through unchanged.

    Composes with patch 1 (``monkey_patch_reasoning_field_ingest``):
    that wrapper guards on ``role == "assistant"``; ours guards on
    ``role == "tool"``. Disjoint role branches, clean composition
    regardless of install order.
    """
    saved_list: list[dict] | None = None
    if isinstance(message, dict) and message.get("role") == "tool":
        saved_list = _filter_tool_content_preserving_media(
            message.get("content")
        )
    result = _original(
        message,
        mm_tracker,
        content_format,
        interleave_strings,
        mm_processor_kwargs,
    )
    if saved_list is not None and isinstance(result, list) and result:
        # The role:"tool" branch returns a single-element list with a
        # synthesized ConversationMessage (role+content+tool_call_id+...).
        # Replace the flattened-string content with our preserved list.
        target = result[0]
        if isinstance(target, dict):
            target["content"] = saved_list
    return result


_parse_chat_message_content_with_tool_media_preserved.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_parse_chat_message_content_with_tool_media_preserved.__wrapped_original__ = _original  # type: ignore[attr-defined]
_parse_chat_message_content_with_tool_media_preserved.__name__ = (
    "_parse_chat_message_content"
)
_parse_chat_message_content_with_tool_media_preserved.__qualname__ = (
    "_parse_chat_message_content"
)
_parse_chat_message_content_with_tool_media_preserved.__module__ = (
    _chat_utils_mod.__name__
)


# ---------------------------------------------------------------------------
# Phase 6 — install + verification (dynamic and static lookups must agree).
# ---------------------------------------------------------------------------

_chat_utils_mod._parse_chat_message_content = (
    _parse_chat_message_content_with_tool_media_preserved
)

_dyn = getattr(_chat_utils_mod, "_parse_chat_message_content", None)
_require(
    _dyn is _parse_chat_message_content_with_tool_media_preserved,
    "post-install: dynamic lookup of vllm.entrypoints.chat_utils."
    "_parse_chat_message_content does not see the patched wrapper.",
)
_static = inspect.getattr_static(
    _chat_utils_mod, "_parse_chat_message_content"
)
_require(
    _static is _parse_chat_message_content_with_tool_media_preserved,
    "post-install: static lookup of vllm.entrypoints.chat_utils."
    "_parse_chat_message_content does not see the patched wrapper "
    "(metaclass shim or re-import racing).",
)
_require(
    getattr(_dyn, "__qwen36_patch__", None) == _PATCH_TAG,
    f"post-install: vllm.entrypoints.chat_utils._parse_chat_message_content "
    f"carries unexpected tag "
    f"{getattr(_dyn, '__qwen36_patch__', None)!r}, expected {_PATCH_TAG!r}.",
)
_require(
    getattr(_dyn, "__wrapped_original__", None) is _original,
    "post-install: vllm.entrypoints.chat_utils._parse_chat_message_content "
    "bears tag but __wrapped_original__ is not the value the wrapper "
    "captured at install time — forging defense failed.",
)


_logger.info(
    "[%s] applied: wrapped vllm.entrypoints.chat_utils."
    "_parse_chat_message_content for vLLM commit %s (preserves list-shaped "
    "tool-role content with media so the chat template renders "
    "<|vision_start|><|image_pad|><|vision_end|> markers natively inside "
    "<tool_response>; composes with patch 1 — disjoint role branches).",
    _PATCH_TAG,
    _PINNED_VLLM_COMMIT,
)
