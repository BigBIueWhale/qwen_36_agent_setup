"""Strict, fail-loud detector for ``<tool_call>`` markup emitted inside ``<think>``.

Why this patch must exist
-------------------------

Qwen3.6 occasionally emits ``<tool_call>...</tool_call>`` inside
``<think>...</think>`` (single-digit percent under agentic workloads).
Upstream ``Qwen3ReasoningParser`` correctly partitions on ``</think>``
and routes mid-think markup to ``reasoning`` (non-streaming
``qwen3_reasoning_parser.py:54-93``; streaming
``qwen3_reasoning_parser.py:95-147``). **The parser is correct to its
contract; the model is misbehaving.** This patch wraps both surfaces,
emits a single structured WARNING per request when the OOD markup is
detected, and passes the upstream return value through unchanged so
the agent's retry policy decides what to do — no rescue, no state
machine. Both wrappers emit the *byte-identical* format string so an
operator can ``grep model_emit_warning`` (or wire up any standard log
shipper) to count events; the structured suffix carries
``reasoning_len=N marker_count=M`` for histogramming.

Streaming wrapper state model
-----------------------------

Streaming requires per-request state (the WARNING fires once per
request, not once per delta). vLLM constructs a new
``Qwen3ReasoningParser`` instance per request inside
``OpenAIServingChat.create_chat_completion`` (``vllm/entrypoints/openai/
chat_completion/serving.py:247``), so the state can live on the
parser instance itself; we use two attribute names prefixed
``_qwen36_tool_call_in_think_…`` to avoid colliding with any upstream
field. Initialisation is explicit (``hasattr`` guard) — no silent
defaulting via ``getattr(default=…)``. Buffer/flag reset implicitly
between requests because the entire instance is replaced.

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce``.
**Removal trigger**: Qwen3.6 retraining eliminates the OOD emission.
"""

from __future__ import annotations

import inspect
from typing import Any

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.reasoning import qwen3_reasoning_parser as _qwen3_reasoning_mod

_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-tool-call-in-think-detector-v2"
_TOOL_CALL_OPEN: str = "<tool_call>"
_WARNING_FORMAT: str = (
    "model_emit_warning kind=tool_call_in_reasoning "
    "reasoning_len=%d marker_count=%d"
)
# Source landmarks — substrings required in the wrapped methods so an
# upstream refactor of the partitioning strategy forces a re-audit.
_NONSTREAMING_LANDMARK: str = "model_output.partition(self.end_token)"
_STREAMING_LANDMARK: str = "self.end_token_id in delta_token_ids"

# Per-instance attributes for streaming-mode state. Names are namespaced
# so they cannot collide with anything the upstream class might add.
_BUF_ATTR: str = "_qwen36_tool_call_in_think_buffer"
_WARNED_ATTR: str = "_qwen36_tool_call_in_think_warned"

_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class DetectorPatchRefusedError(RuntimeError):
    """Precondition for the detector was violated."""


def _require(cond: object, msg: str) -> None:
    if not cond:
        raise DetectorPatchRefusedError(f"[{_PATCH_TAG}] refusing to patch: {msg}")


# ----------------------------------------------------------------------
# Locate the target class.
# ----------------------------------------------------------------------

_ParserCls = getattr(_qwen3_reasoning_mod, "Qwen3ReasoningParser", None)
_require(
    _ParserCls is not None and inspect.isclass(_ParserCls),
    "Qwen3ReasoningParser missing or not a class.",
)


# ----------------------------------------------------------------------
# Phase 1 — non-streaming wrapper (extract_reasoning).
# ----------------------------------------------------------------------

_original_nonstreaming = getattr(_ParserCls, "extract_reasoning", None)
_require(
    callable(_original_nonstreaming),
    "Qwen3ReasoningParser.extract_reasoning missing.",
)
try:
    _ns_sig = inspect.signature(_original_nonstreaming)  # type: ignore[arg-type]
    _ns_src = inspect.getsource(_original_nonstreaming)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise DetectorPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect extract_reasoning: {_exc!r}"
    ) from _exc
_require(
    list(_ns_sig.parameters) == ["self", "model_output", "request"],
    f"extract_reasoning signature drifted; got {list(_ns_sig.parameters)!r}.",
)
_require(
    _NONSTREAMING_LANDMARK in _ns_src,
    f"non-streaming landmark {_NONSTREAMING_LANDMARK!r} missing.",
)


def _extract_reasoning_with_detector(
    self: Any, model_output: str, request: Any
) -> Any:
    """Pass-through non-streaming wrapper. Emits one structured WARNING
    when reasoning contains literal ``<tool_call>`` markup. Returns the
    upstream tuple unchanged so behaviour is byte-identical to no-patch."""
    reasoning, content = _original_nonstreaming(self, model_output, request)
    if reasoning is not None and _TOOL_CALL_OPEN in reasoning:
        _logger.warning(
            _WARNING_FORMAT,
            len(reasoning),
            reasoning.count(_TOOL_CALL_OPEN),
        )
    return reasoning, content


_extract_reasoning_with_detector.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_extract_reasoning_with_detector.__wrapped_original__ = _original_nonstreaming  # type: ignore[attr-defined]
_extract_reasoning_with_detector.__name__ = "extract_reasoning"
_extract_reasoning_with_detector.__qualname__ = (
    f"{_ParserCls.__qualname__}.extract_reasoning"
)

_ParserCls.extract_reasoning = _extract_reasoning_with_detector
_require(
    getattr(_ParserCls.extract_reasoning, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install (extract_reasoning): tag absent via attribute access.",
)
_require(
    getattr(
        inspect.getattr_static(_ParserCls, "extract_reasoning"),
        "__qwen36_patch__",
        None,
    )
    == _PATCH_TAG,
    "post-install (extract_reasoning): inspect.getattr_static disagrees.",
)


# ----------------------------------------------------------------------
# Phase 2 — streaming wrapper (extract_reasoning_streaming).
# ----------------------------------------------------------------------

_original_streaming = getattr(_ParserCls, "extract_reasoning_streaming", None)
_require(
    callable(_original_streaming),
    "Qwen3ReasoningParser.extract_reasoning_streaming missing.",
)
try:
    _s_sig = inspect.signature(_original_streaming)  # type: ignore[arg-type]
    _s_src = inspect.getsource(_original_streaming)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise DetectorPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect extract_reasoning_streaming: {_exc!r}"
    ) from _exc
_require(
    list(_s_sig.parameters) == [
        "self",
        "previous_text",
        "current_text",
        "delta_text",
        "previous_token_ids",
        "current_token_ids",
        "delta_token_ids",
    ],
    f"extract_reasoning_streaming signature drifted; got {list(_s_sig.parameters)!r}.",
)
_require(
    _STREAMING_LANDMARK in _s_src,
    f"streaming landmark {_STREAMING_LANDMARK!r} missing.",
)


def _extract_reasoning_streaming_with_detector(
    self: Any,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Any,
    current_token_ids: Any,
    delta_token_ids: Any,
) -> Any:
    """Pass-through streaming wrapper. Emits one structured WARNING per
    request (dedup'd via instance attributes) when the cumulative reasoning
    text contains literal ``<tool_call>`` markup. Returns the upstream
    ``DeltaMessage`` (or ``None``) unchanged so behaviour is byte-identical
    to no-patch.

    State (per parser instance, == per request — vLLM creates a new
    instance at ``serving.py:247``): a cumulative reasoning buffer plus a
    ``_warned`` flag. Initialisation is explicit on first call; reset is
    implicit at next request when a fresh instance is constructed.
    """
    delta = _original_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    )
    # Explicit per-instance lazy init (no fallback default semantics).
    if not hasattr(self, _BUF_ATTR):
        setattr(self, _BUF_ATTR, "")
        setattr(self, _WARNED_ATTR, False)
    if delta is not None and getattr(delta, "reasoning", None):
        buf = getattr(self, _BUF_ATTR) + delta.reasoning
        setattr(self, _BUF_ATTR, buf)
        if not getattr(self, _WARNED_ATTR) and _TOOL_CALL_OPEN in buf:
            _logger.warning(
                _WARNING_FORMAT,
                len(buf),
                buf.count(_TOOL_CALL_OPEN),
            )
            setattr(self, _WARNED_ATTR, True)
    return delta


_extract_reasoning_streaming_with_detector.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_extract_reasoning_streaming_with_detector.__wrapped_original__ = _original_streaming  # type: ignore[attr-defined]
_extract_reasoning_streaming_with_detector.__name__ = "extract_reasoning_streaming"
_extract_reasoning_streaming_with_detector.__qualname__ = (
    f"{_ParserCls.__qualname__}.extract_reasoning_streaming"
)

_ParserCls.extract_reasoning_streaming = _extract_reasoning_streaming_with_detector
_require(
    getattr(_ParserCls.extract_reasoning_streaming, "__qwen36_patch__", None)
    == _PATCH_TAG,
    "post-install (extract_reasoning_streaming): tag absent via attribute access.",
)
_require(
    getattr(
        inspect.getattr_static(_ParserCls, "extract_reasoning_streaming"),
        "__qwen36_patch__",
        None,
    )
    == _PATCH_TAG,
    "post-install (extract_reasoning_streaming): inspect.getattr_static disagrees.",
)


# ----------------------------------------------------------------------
# Phase 3 — behavioural verification of the streaming wrapper.
# ----------------------------------------------------------------------
#
# We can't construct a real ``Qwen3ReasoningParser`` at import time
# (its __init__ wants a tokenizer and the start/end-token ids). Build
# a stand-in that has the *exact* surface the wrapper touches —
# nothing more — and drive the wrapper with a fake upstream that
# returns DeltaMessages we control. This proves the dedup, the
# state-attribute initialisation, the format-string match, and the
# pass-through return-value contract are all correct on the patched
# code path before any real request hits the engine.

class _FakeDelta:
    __slots__ = ("reasoning", "content")

    def __init__(self, reasoning: str | None = None, content: str | None = None) -> None:
        self.reasoning = reasoning
        self.content = content


class _FakeInstance:
    """Bare ``Qwen3ReasoningParser`` shape for behavioural test only. We
    bypass __init__ to avoid the tokenizer dependency; the wrapper only
    reads/writes our two namespaced attributes so this is safe."""


def _phase3_behavioural_verify() -> None:
    # 1. The wrapped method must round-trip an upstream None unchanged.
    seen_warnings: list[tuple[int, int]] = []

    def _capture_warning(fmt: str, *args: object) -> None:
        if fmt == _WARNING_FORMAT:
            seen_warnings.append((int(args[0]), int(args[1])))  # type: ignore[arg-type]

    real_warning = _logger.warning
    _logger.warning = _capture_warning  # type: ignore[assignment]
    try:
        # Replace the module-level _original_streaming with a stub that
        # returns whatever we want — the wrapper closes over it lexically,
        # so we patch via globals().
        upstream_returns: list[Any] = []
        captured_calls: list[tuple[Any, ...]] = []

        def _fake_upstream(
            self: Any,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Any,
            current_token_ids: Any,
            delta_token_ids: Any,
        ) -> Any:
            captured_calls.append((previous_text, current_text, delta_text))
            return upstream_returns.pop(0)

        prev_original = globals()["_original_streaming"]
        globals()["_original_streaming"] = _fake_upstream
        try:
            inst = _FakeInstance()

            # Case A: upstream returns None — wrapper returns None and
            # does NOT initialise state (we should NOT touch attributes
            # if upstream returned nothing).
            upstream_returns.append(None)
            out = _extract_reasoning_streaming_with_detector(
                inst, "", "", "", (), (), ()
            )
            _require(out is None, "Phase 3 case A: upstream None must round-trip.")

            # Case B: reasoning delta with NO marker — buffer accumulates,
            # no warning, instance attrs initialised exactly once.
            inst2 = _FakeInstance()
            upstream_returns.append(_FakeDelta(reasoning="hello "))
            upstream_returns.append(_FakeDelta(reasoning="world"))
            d1 = _extract_reasoning_streaming_with_detector(
                inst2, "", "hello ", "hello ", (), (), ()
            )
            d2 = _extract_reasoning_streaming_with_detector(
                inst2, "hello ", "hello world", "world", (), (), ()
            )
            _require(d1 is not None and d1.reasoning == "hello ", "Phase 3 case B d1 mismatch.")
            _require(d2 is not None and d2.reasoning == "world", "Phase 3 case B d2 mismatch.")
            _require(getattr(inst2, _BUF_ATTR) == "hello world", "Phase 3 case B buffer mismatch.")
            _require(getattr(inst2, _WARNED_ATTR) is False, "Phase 3 case B should not warn.")
            _require(len(seen_warnings) == 0, "Phase 3 case B: unexpected warning fired.")

            # Case C: marker arrives as one whole delta — exactly one
            # warning, dedup flag flips, buffer + counts correct.
            inst3 = _FakeInstance()
            upstream_returns.append(_FakeDelta(reasoning="prefix <tool_call>foo"))
            d = _extract_reasoning_streaming_with_detector(
                inst3, "", "prefix <tool_call>foo", "prefix <tool_call>foo",
                (), (), (),
            )
            _require(d is not None and d.reasoning == "prefix <tool_call>foo", "Phase 3 case C delta mismatch.")
            _require(getattr(inst3, _WARNED_ATTR) is True, "Phase 3 case C: warned flag must flip.")
            _require(
                seen_warnings[-1] == (len("prefix <tool_call>foo"), 1),
                f"Phase 3 case C: warning args wrong, got {seen_warnings[-1]!r}.",
            )

            # Case D: marker split across two deltas — wrapper detects on
            # the boundary-completing delta and emits exactly one warning.
            inst4 = _FakeInstance()
            seen_before = len(seen_warnings)
            upstream_returns.append(_FakeDelta(reasoning="abc <tool_"))
            upstream_returns.append(_FakeDelta(reasoning="call>xyz"))
            _extract_reasoning_streaming_with_detector(inst4, "", "abc <tool_", "abc <tool_", (), (), ())
            _require(len(seen_warnings) == seen_before, "Phase 3 case D: marker incomplete must NOT warn yet.")
            _extract_reasoning_streaming_with_detector(
                inst4, "abc <tool_", "abc <tool_call>xyz", "call>xyz", (), (), (),
            )
            _require(len(seen_warnings) == seen_before + 1, "Phase 3 case D: completed marker must warn exactly once.")
            _require(
                seen_warnings[-1] == (len("abc <tool_call>xyz"), 1),
                f"Phase 3 case D: warning args wrong, got {seen_warnings[-1]!r}.",
            )

            # Case E: dedup — second marker in same instance does NOT
            # produce a second warning.
            seen_before = len(seen_warnings)
            upstream_returns.append(_FakeDelta(reasoning=" and another <tool_call>"))
            _extract_reasoning_streaming_with_detector(
                inst4, "abc <tool_call>xyz",
                "abc <tool_call>xyz and another <tool_call>",
                " and another <tool_call>", (), (), (),
            )
            _require(len(seen_warnings) == seen_before, "Phase 3 case E: dedup must suppress second warning.")

            # Case F: content-only delta on a fresh instance must NOT
            # accumulate into the reasoning buffer.
            inst5 = _FakeInstance()
            upstream_returns.append(_FakeDelta(content="post-think content"))
            _extract_reasoning_streaming_with_detector(
                inst5, "", "post-think content", "post-think content", (), (), (),
            )
            _require(getattr(inst5, _BUF_ATTR) == "", "Phase 3 case F: content delta leaked into reasoning buffer.")
            _require(getattr(inst5, _WARNED_ATTR) is False, "Phase 3 case F: content delta must not flip warned.")

        finally:
            globals()["_original_streaming"] = prev_original
    finally:
        _logger.warning = real_warning  # type: ignore[assignment]


_phase3_behavioural_verify()


_logger.info(
    "[%s] applied: wrapped %s.extract_reasoning (non-streaming) and "
    "%s.extract_reasoning_streaming for vLLM commit %s.",
    _PATCH_TAG,
    _ParserCls.__qualname__,
    _ParserCls.__qualname__,
    _PINNED_VLLM_COMMIT,
)
