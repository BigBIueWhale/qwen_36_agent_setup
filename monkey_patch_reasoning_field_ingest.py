"""Strict, fail-loud ingest patch for the ``reasoning_content`` field on
inbound assistant messages.

Why this patch must exist
-------------------------

vLLM's chat-completion ingest at ``vllm/entrypoints/chat_utils.py``
line 1519 reads ``message.get("reasoning")`` and silently drops
``reasoning_content``. The chat template at ``chat_template.jinja``
lines 91-92 reads ``message.reasoning_content`` to render historical
``<think>`` blocks under ``preserve_thinking=true``. **vLLM is feeding
its own template a field its own ingest discards.** The bug is purely
upstream: the ingest path's "what counts as reasoning" disagrees with
the template's "what counts as reasoning" — independent of any
client-side standard. Without this patch every multi-turn agent loop
loses prior reasoning on replay; the model — RL-trained to expect
prior-turn reasoning — re-derives context from scratch and tool-arg
correctness degrades after 2-3 turns (``badlogic/pi-mono#3325``).

Companion patch: ``monkey_patch_reasoning_field_egress.py`` (egress half
of the same ``reasoning`` / ``reasoning_content`` mismatch). The two
together close the round-trip silently broken by upstream.

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce``.

Patch-discipline contract
-------------------------

This file is a patch, not a library. At import it:

1. Imports vLLM and ``vllm.entrypoints.chat_utils``; either failing is
   a hard ImportError.
2. Verifies ``_parse_chat_message_content`` exists, is callable, and
   has the expected signature (parameter names + the trailing-default
   shape the wrapper relies on).
3. Reads the function's source via ``inspect.getsource`` and verifies
   two landmarks: the bug line ``reasoning = message.get("reasoning")``
   and the assistant-role gate ``if role == "assistant":``. Both
   absent → upstream restructured the path → refuse.
4. Installs the wrapper, tags it with ``__qwen36_patch__``, and verifies
   both ``getattr`` and ``inspect.getattr_static`` resolve to the
   tagged wrapper.

Any step failing raises :class:`IngestPatchRefusedError` and the
interpreter does not continue. No ``SystemExit(0)``, no
``try/except: pass``, no silent fallback.

Critical correctness invariants
-------------------------------

* **Input normalization only.** The wrapper NEVER mutates the caller's
  ``message`` dict; it returns a shallow copy when synthesis is
  needed. Callers may hold references for logging / audit / retry.
* **No silent resolution of client ambiguity.** Both ``reasoning`` and
  ``reasoning_content`` set with **different** non-None values raises
  :class:`ReasoningFieldAmbiguityError` (a ``ValueError`` subclass →
  HTTP 400). Choosing one and dropping the other is the silent-
  degradation class this patch exists to remove.
* **Role-gated.** Synthesis only runs on ``role == "assistant"``; the
  template reads ``reasoning_content`` only from assistant turns.
* **Non-string ``reasoning_content`` is refused, not coerced.** A dict
  or list stringified via ``str(...)`` would corrupt downstream
  template rendering.
* **Pass-through on happy paths.** Neither set, or only ``reasoning``
  set (upstream's existing behavior) → the original is called with
  the unchanged ``message`` object — no allocation, no identity change.
* **No try/except on the delegate call.** Exceptions from the upstream
  function propagate unchanged.
"""

from __future__ import annotations

import inspect
from typing import Any


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-reasoning-ingest-v1"


# Source landmarks proving the ingest shape is the one we expect.
_BUG_LANDMARK: str = 'reasoning = message.get("reasoning")'
_ROLE_GATE_LANDMARK: str = 'if role == "assistant":'


# Exact parameter list we verified against the pinned commit. Any drift
# means the wrapper's positional/keyword contract with the original has
# broken and silent arg-shifts become possible.
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

    See module docstring for the full policy matrix.
    """
    # Duck-typed ``.get`` check: real type is ``TypedDict`` at type-
    # check time and ``dict`` at runtime. Anything else falls through
    # to the original function, which raises its own clear error.
    if not isinstance(message, dict):
        return message

    if message.get("role") != "assistant":
        return message

    reasoning = message.get("reasoning")
    reasoning_content = message.get("reasoning_content")

    # Happy path #1: neither field set. No-op.
    if reasoning is None and reasoning_content is None:
        return message

    # Happy path #2: only ``reasoning`` set. Upstream already handles it.
    if reasoning is not None and reasoning_content is None:
        return message

    if reasoning is not None and reasoning_content is not None:
        if reasoning == reasoning_content:
            # Redundant but not ambiguous; upstream populates both
            # output slots from ``reasoning`` alone, so delegate.
            return message
        raise ReasoningFieldAmbiguityError(
            "assistant message specifies both 'reasoning' and "
            "'reasoning_content' with different values; the client "
            "must send at most one, or identical values in both. "
            f"reasoning type={type(reasoning).__name__!r}, "
            f"reasoning_content type={type(reasoning_content).__name__!r}."
        )

    # Remaining case: ``reasoning is None`` and
    # ``reasoning_content is not None``. The class of request upstream
    # silently drops. Normalize before delegating.
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
    """Ingest wrapper around ``_parse_chat_message_content``."""
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

# Static-lookup verification: ``inspect.getattr_static`` bypasses
# descriptor protocol and module-level ``__getattr__`` hooks. Both
# views must agree.
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
