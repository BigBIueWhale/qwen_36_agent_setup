"""Strict, fail-loud ingest patch for the ``reasoning`` / ``reasoning_content`` split.

Target: vLLM commit ``32e45636e3d7e02615facc8c63645ce4ac1d7e11`` (README §3.2).
Companion to ``monkey_patch_reasoning_field_egress.py`` (§7.3); this file
is §7.4. Same fail-loud import-time discipline as every other patch in
this repo.

What this fixes
---------------

vLLM's chat-completion ingest at ``vllm/entrypoints/chat_utils.py``
line 1510-1548 (``_parse_chat_message_content``) accepts ``reasoning``
from assistant messages::

    reasoning = message.get("reasoning")
    ...
    if reasoning is not None:
        result_msg["reasoning"] = cast(str, reasoning)
        result_msg["reasoning_content"] = cast(str, reasoning)

and **silently drops** ``reasoning_content``. The downstream chat
template at ``chat_template.jinja`` line 91-92 reads
``message.reasoning_content``, which means the existing code only
satisfies the template when the client happens to have sent
``reasoning``. The two clients in our production stack both send
``reasoning_content``:

* Qwen Code CLI — ``qwen-code/packages/core/src/core/openaiContentGenerator/converter.ts:335,514``
* Qwen-Agent — ``Qwen-Agent/qwen_agent/llm/oai.py:149, 169-173``

Both are silently losing prior-turn reasoning today. Prior-turn
reasoning is load-bearing for interleaved thinking: without it the
model re-derives context from scratch, loses chain-of-thought
continuity across tool-call round-trips, and degrades tool-selection
quality. The README calls this out; this patch is the fix.

Why a wrapper, not a rewrite
----------------------------

The branch we care about is one ``message.get(...)`` call inside a
~40-line function that also handles tool-call passthrough, content
normalization, and multi-modal item tracking. Rewriting the whole
function to fix one line triples the surface area that drifts when
upstream moves. A **wrapper that normalizes the input dict before
delegating** is the minimum-surface change that preserves every other
code path verbatim.

Patch-discipline contract
-------------------------

This file is a patch, not a library. At import it:

1. Imports vLLM. Failure is a hard ImportError; we do not hide it.
2. Imports ``vllm.entrypoints.chat_utils``; failure is fatal.
3. Verifies ``_parse_chat_message_content`` exists, is callable, and
   has exactly the signature
   ``(message, mm_tracker, content_format, interleave_strings,
   mm_processor_kwargs=None)``.
4. Reads the function's source via ``inspect.getsource`` and verifies
   two landmarks are present:

   * ``reasoning = message.get("reasoning")`` — the exact bug line.
     If it is missing, upstream has already restructured this code and
     our assumption that ``reasoning_content`` is dropped may no
     longer hold; we refuse to wrap.
   * ``if role == "assistant":`` — the role-gated branch this patch
     relies on. If it is missing, the downstream semantics of the
     synthesized ``reasoning`` field have changed and our input
     normalization may be wrong; we refuse to wrap.

5. Installs the wrapper on the module, tags it with
   ``__qwen36_patch__``, verifies both ``getattr`` and
   ``inspect.getattr_static`` resolve to the tagged wrapper.
6. Logs a single INFO line via ``vllm.logger.init_logger`` naming the
   module, the wrapped function, and the pinned commit.

Any step failing raises :class:`IngestPatchRefusedError` and the
interpreter does not continue. There is no ``SystemExit(0)`` or
``try/except Exception: pass`` on any install path.

Critical correctness invariants
-------------------------------

* **Input normalization only.** The wrapper NEVER mutates the caller's
  ``message`` dict. When it needs to synthesize ``reasoning``, it
  takes a shallow copy and modifies the copy before calling the
  original. Callers may hold references to the inbound dict (for
  logging, audit, retry buffers); we must not surprise them.
* **No silent resolution of client ambiguity.** If the client sent
  both ``reasoning`` and ``reasoning_content`` with different non-None
  values, we do not guess which one it "really" meant. We raise
  :class:`ReasoningFieldAmbiguityError` — a ``ValueError`` subclass so
  vLLM's request-level error handling converts it to HTTP 400 rather
  than 500. The alternative — prefer one, drop the other — is exactly
  the silent-degradation class this patch exists to remove.
* **Role-gated.** The synthesis only runs on ``role == "assistant"``.
  The template reads ``reasoning_content`` only from assistant turns;
  synthesizing on ``user`` / ``system`` / ``tool`` turns would inject
  a field the downstream consumers do not expect.
* **Non-None, non-string inputs are not coerced.** If
  ``reasoning_content`` is present but not a string, we refuse
  (``ReasoningFieldAmbiguityError``) rather than stringifying it.
  Silent coercion of a dict or list to ``str(dict)`` would corrupt
  downstream template rendering in exactly the way this whole patch
  stack is designed to prevent.
* **Pass-through on happy paths.** If neither field is set, or if
  only ``reasoning`` is set (matching upstream's existing behavior),
  the original is called with the original ``message`` object — not a
  copy. No allocation, no identity change.
* **No try/except on the original call.** Exceptions from the
  upstream function propagate unchanged. The wrapper adds exactly one
  pre-processing step and delegates.
"""

from __future__ import annotations

import inspect
from typing import Any


_PINNED_VLLM_COMMIT: str = "32e45636e3d7e02615facc8c63645ce4ac1d7e11"
_PATCH_TAG: str = "qwen36-agent-setup-reasoning-ingest-v1"


# Source landmarks proving the ingest shape is the one we expect.
# ``_BUG_LANDMARK`` is the exact line that drops ``reasoning_content``;
# ``_ROLE_GATE_LANDMARK`` is the role-gated branch whose template
# consumer assumes ``reasoning_content`` is populated.
_BUG_LANDMARK: str = 'reasoning = message.get("reasoning")'
_ROLE_GATE_LANDMARK: str = 'if role == "assistant":'


# Exact parameter list we verified against commit 8936118. Any drift
# here means the wrapper's positional/keyword contract with the
# original has broken and silent arg-shifts become possible.
_EXPECTED_PARAMS: list[str] = [
    "message",
    "mm_tracker",
    "content_format",
    "interleave_strings",
    "mm_processor_kwargs",
]


class IngestPatchRefusedError(RuntimeError):
    """A precondition for the reasoning-ingest wrapper was violated.

    Raised at import time only. The patch either applies cleanly and
    correctly, or the process does not come up. A half-installed
    ingest patch — upstream unchanged, clients continuing to send
    ``reasoning_content``, reasoning silently dropped — is exactly
    the failure mode this file exists to eliminate. We refuse to
    enter that state.
    """


class ReasoningFieldAmbiguityError(ValueError):
    """Client sent both ``reasoning`` and ``reasoning_content`` with different values.

    Raised at request time inside the wrapper, not at import time.
    Subclasses :class:`ValueError` so vLLM's request-level error
    handling converts it to HTTP 400 (client error) rather than 500
    (server error). The client must resolve the ambiguity; we do not
    guess on their behalf.
    """


def _require(condition: object, failure_message: str) -> None:
    if not condition:
        raise IngestPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {failure_message}"
        )


# --------------------------------------------------------------------
# Phase 1: Import vLLM and locate the target module.
# --------------------------------------------------------------------

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.entrypoints import chat_utils as _chat_utils_mod

_logger = init_logger(__name__)


# --------------------------------------------------------------------
# Phase 2: Landmark the function we intend to wrap.
# --------------------------------------------------------------------

_original: Any = getattr(_chat_utils_mod, "_parse_chat_message_content", None)
_require(
    _original is not None and callable(_original),
    "vllm.entrypoints.chat_utils._parse_chat_message_content is "
    "missing or not callable. Upstream has moved or renamed the "
    "function; re-audit before bumping the pinned commit.",
)

try:
    _sig = inspect.signature(_original)
except (TypeError, ValueError) as _exc:
    raise IngestPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect signature of "
        f"_parse_chat_message_content: {_exc!r}"
    ) from _exc

_param_names = list(_sig.parameters)
_require(
    _param_names == _EXPECTED_PARAMS,
    f"_parse_chat_message_content signature changed; expected "
    f"{_EXPECTED_PARAMS!r}, got {_param_names!r}.",
)

# The trailing parameter must remain optional (the wrapper forwards
# ``**kwargs``-style via explicit named binding). A drift from
# keyword-with-default to required would not trip the name check
# above, so we assert the default explicitly.
_mm_processor_param = _sig.parameters["mm_processor_kwargs"]
_require(
    _mm_processor_param.default is None,
    f"_parse_chat_message_content.mm_processor_kwargs default "
    f"changed from None to {_mm_processor_param.default!r}; the "
    f"wrapper's default-forwarding assumption no longer holds.",
)


# --------------------------------------------------------------------
# Phase 3: Landmark the bug shape inside the function body.
# --------------------------------------------------------------------

try:
    _original_src = inspect.getsource(_original)
except (OSError, TypeError) as _exc:
    raise IngestPatchRefusedError(
        f"[{_PATCH_TAG}] cannot read source of "
        f"_parse_chat_message_content (is vLLM installed without "
        f"accompanying .py files?): {_exc!r}"
    ) from _exc

_require(
    _BUG_LANDMARK in _original_src,
    f"bug landmark {_BUG_LANDMARK!r} not found in "
    f"_parse_chat_message_content source. Upstream has restructured "
    f"the reasoning-ingest path; refusing to wrap a function whose "
    f"contract has drifted.",
)

_require(
    _ROLE_GATE_LANDMARK in _original_src,
    f"role-gate landmark {_ROLE_GATE_LANDMARK!r} not found in "
    f"_parse_chat_message_content source. The assistant-only branch "
    f"this patch relies on no longer exists in the expected shape; "
    f"refusing to wrap.",
)


# --------------------------------------------------------------------
# Phase 4: The wrapper.
# --------------------------------------------------------------------


def _normalize_assistant_reasoning(message: Any) -> Any:
    """Return a message dict where ``reasoning`` is populated if ``reasoning_content`` was.

    Contract:

    * Non-``dict`` / non-mapping inputs: returned unchanged. The
      upstream function is typed ``ChatCompletionMessageParam``
      (``TypedDict``); if a caller has passed something else, that is
      upstream's problem to diagnose, not ours to mask.
    * Non-assistant roles: returned unchanged.
    * Assistant role, both fields ``None`` or absent: returned
      unchanged.
    * Assistant role, ``reasoning`` is non-None: returned unchanged
      (matches existing upstream behavior; ``reasoning_content``, if
      also present and equal, is redundant but not ambiguous — see
      below for the unequal case).
    * Assistant role, ``reasoning`` is None and ``reasoning_content``
      is a non-empty string: a shallow copy is made with
      ``reasoning`` set from ``reasoning_content``. The caller's dict
      is NOT mutated.
    * Assistant role, both fields non-None and **unequal**: raises
      :class:`ReasoningFieldAmbiguityError`.
    * Assistant role, ``reasoning_content`` is present but not a
      string (or not a non-None value the upstream function could
      itself have accepted in the ``reasoning`` slot): raises
      :class:`ReasoningFieldAmbiguityError`. Silent coercion is not
      an acceptable outcome.
    """
    # Duck-typed ``.get`` check: the real type is ``TypedDict`` at
    # type-check time and ``dict`` at runtime. We accept any mapping
    # that exposes ``.get`` and ``["role"]``; anything else falls
    # through to the original function, which will raise its own
    # clear error.
    if not isinstance(message, dict):
        return message

    if message.get("role") != "assistant":
        return message

    reasoning = message.get("reasoning")
    reasoning_content = message.get("reasoning_content")

    # Happy path #1: neither field set. No-op.
    if reasoning is None and reasoning_content is None:
        return message

    # Happy path #2: ``reasoning`` is set, ``reasoning_content`` is
    # absent or identical. Upstream already handles this correctly.
    if reasoning is not None and reasoning_content is None:
        return message

    if reasoning is not None and reasoning_content is not None:
        if reasoning == reasoning_content:
            # Client sent both with identical values. Not ambiguous;
            # just redundant. Upstream happens to already populate
            # both output slots from ``reasoning`` alone, so we can
            # delegate unchanged.
            return message
        raise ReasoningFieldAmbiguityError(
            "assistant message specifies both 'reasoning' and "
            "'reasoning_content' with different values; the client "
            "must send at most one, or identical values in both. "
            f"reasoning type={type(reasoning).__name__!r}, "
            f"reasoning_content type={type(reasoning_content).__name__!r}."
        )

    # Remaining case: ``reasoning is None`` and
    # ``reasoning_content is not None``. This is the class of request
    # upstream silently drops. Normalize before delegating.
    if not isinstance(reasoning_content, str):
        raise ReasoningFieldAmbiguityError(
            "assistant message 'reasoning_content' is not a string "
            f"(got type {type(reasoning_content).__name__!r}); the "
            "downstream chat template expects a string and this "
            "patch refuses to coerce silently."
        )

    synthesized = dict(message)
    synthesized["reasoning"] = reasoning_content
    return synthesized


def _parse_chat_message_content_reasoning_normalized(
    message: Any,
    mm_tracker: Any,
    content_format: Any,
    interleave_strings: bool,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Ingest wrapper around ``_parse_chat_message_content``.

    Contract:

    * Normalizes ``message`` via :func:`_normalize_assistant_reasoning`
      (see that function for the full policy matrix).
    * Calls the original ``_parse_chat_message_content`` with the
      normalized dict and the remaining arguments forwarded verbatim.
      The original call is **NOT** inside a try/except; any exception
      propagates unchanged.
    * Returns the original's return value unchanged.
    """
    normalized = _normalize_assistant_reasoning(message)
    return _original(
        normalized,
        mm_tracker,
        content_format,
        interleave_strings,
        mm_processor_kwargs,
    )


_parse_chat_message_content_reasoning_normalized.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_parse_chat_message_content_reasoning_normalized.__wrapped_original__ = _original  # type: ignore[attr-defined]
_parse_chat_message_content_reasoning_normalized.__name__ = (
    "_parse_chat_message_content"
)
_parse_chat_message_content_reasoning_normalized.__qualname__ = (
    "_parse_chat_message_content"
)
_parse_chat_message_content_reasoning_normalized.__module__ = (
    _chat_utils_mod.__name__
)


# --------------------------------------------------------------------
# Phase 5: Install and verify.
# --------------------------------------------------------------------

_chat_utils_mod._parse_chat_message_content = (
    _parse_chat_message_content_reasoning_normalized
)

_installed = getattr(_chat_utils_mod, "_parse_chat_message_content")
_require(
    getattr(_installed, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install verification failed: "
    "chat_utils._parse_chat_message_content does not bear the "
    "expected patch tag. A concurrent monkey-patch has clobbered "
    "ours.",
)

# Second-order verification: ``inspect.getattr_static`` bypasses
# descriptor protocol and module-level ``__getattr__`` hooks. A module
# that has installed a ``__getattr__`` (PEP 562) could in principle
# return a different object via normal attribute access than the one
# sitting in ``__dict__``. We require both views to agree.
_resolved_static = inspect.getattr_static(
    _chat_utils_mod, "_parse_chat_message_content"
)
_require(
    getattr(_resolved_static, "__qwen36_patch__", None) == _PATCH_TAG,
    "static-lookup verification failed: "
    "inspect.getattr_static sees a different "
    "_parse_chat_message_content than normal attribute access. "
    "A module-level __getattr__ or similar is shadowing our "
    "assignment; refusing to proceed.",
)


_logger.info(
    "[%s] applied: wrapped %s._parse_chat_message_content for vLLM "
    "commit %s (ingest normalization; accepts 'reasoning' or "
    "'reasoning_content' on assistant messages; refuses ambiguous "
    "client input).",
    _PATCH_TAG,
    _chat_utils_mod.__name__,
    _PINNED_VLLM_COMMIT,
)
