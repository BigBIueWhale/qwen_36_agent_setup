"""Strict, fail-loud detector for ``<tool_call>`` markup emitted inside ``<think>``.

Why this patch must exist
-------------------------

Qwen3.6 occasionally emits ``<tool_call>...</tool_call>`` inside
``<think>...</think>`` (single-digit percent under agentic workloads).
Upstream ``Qwen3ReasoningParser.extract_reasoning`` correctly partitions
on ``</think>`` first (``qwen3_reasoning_parser.py:142-144``) and routes
mid-think markup to ``reasoning``. **The parser is correct to its
contract; the model is misbehaving.** This patch detects the OOD
emission, emits a single structured WARNING (greppable, alertable),
and passes the upstream tuple through unchanged so the agent's retry
policy decides what to do — no rescue, no state machine. Streaming
path is intentionally unwrapped: the rate is a model-side property,
not per-modality, so a single non-streaming wrapper suffices.

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
_PATCH_TAG: str = "qwen36-agent-setup-tool-call-in-think-detector-v1"
_TOOL_CALL_OPEN: str = "<tool_call>"
# Source landmark — substring required in the wrapped method body so an
# upstream refactor of the partitioning strategy forces a re-audit.
_NONSTREAMING_LANDMARK: str = "model_output.partition(self.end_token)"

_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class DetectorPatchRefusedError(RuntimeError):
    """Precondition for the detector was violated."""


def _require(cond: object, msg: str) -> None:
    if not cond:
        raise DetectorPatchRefusedError(f"[{_PATCH_TAG}] refusing to patch: {msg}")


_ParserCls = getattr(_qwen3_reasoning_mod, "Qwen3ReasoningParser", None)
_require(
    _ParserCls is not None and inspect.isclass(_ParserCls),
    "Qwen3ReasoningParser missing or not a class.",
)
_original = getattr(_ParserCls, "extract_reasoning", None)
_require(callable(_original), "Qwen3ReasoningParser.extract_reasoning missing.")
try:
    _sig = inspect.signature(_original)  # type: ignore[arg-type]
    _src = inspect.getsource(_original)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise DetectorPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect extract_reasoning: {_exc!r}"
    ) from _exc
_require(
    list(_sig.parameters) == ["self", "model_output", "request"],
    f"extract_reasoning signature drifted; got {list(_sig.parameters)!r}.",
)
_require(
    _NONSTREAMING_LANDMARK in _src,
    f"non-streaming landmark {_NONSTREAMING_LANDMARK!r} missing.",
)


def _extract_reasoning_with_detector(
    self: Any, model_output: str, request: Any
) -> Any:
    """Pass-through; emits a single structured WARNING when reasoning
    contains literal ``<tool_call>`` markup. Returns upstream tuple
    unchanged so behavior is byte-identical to no-patch."""
    reasoning, content = _original(self, model_output, request)
    if reasoning is not None and _TOOL_CALL_OPEN in reasoning:
        _logger.warning(
            "model_emit_warning kind=tool_call_in_reasoning "
            "reasoning_len=%d marker_count=%d",
            len(reasoning),
            reasoning.count(_TOOL_CALL_OPEN),
        )
    return reasoning, content


_extract_reasoning_with_detector.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_extract_reasoning_with_detector.__wrapped_original__ = _original  # type: ignore[attr-defined]
_extract_reasoning_with_detector.__name__ = "extract_reasoning"
_extract_reasoning_with_detector.__qualname__ = (
    f"{_ParserCls.__qualname__}.extract_reasoning"
)

_ParserCls.extract_reasoning = _extract_reasoning_with_detector
_require(
    getattr(_ParserCls.extract_reasoning, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install: tag absent via attribute access.",
)
_require(
    getattr(
        inspect.getattr_static(_ParserCls, "extract_reasoning"),
        "__qwen36_patch__",
        None,
    )
    == _PATCH_TAG,
    "post-install: inspect.getattr_static disagrees.",
)

_logger.info(
    "[%s] applied: wrapped %s.extract_reasoning (non-streaming) for vLLM "
    "commit %s.",
    _PATCH_TAG,
    _ParserCls.__qualname__,
    _PINNED_VLLM_COMMIT,
)
