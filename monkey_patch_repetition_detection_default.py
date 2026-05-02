"""Strict, fail-loud server-side default for vLLM ``repetition_detection``.

Why this patch exists
---------------------

Qwen3.6-27B-AWQ — like the rest of the Qwen3.x family — has a
documented propensity to enter degenerate single-token emission loops
mid-generation (Qwen3.6 issue #88: 17.4% of 1,400 LiveCodeBench outputs
truncated by repetition; Qwen3.6 issue #145; cyankiwi HF discussion #2;
vLLM issue #39348 reproducing the same wall-of-tokens at full BF16).
Alibaba's own Qwen3.6 model card acknowledges "endless repetitions"
and recommends ``presence_penalty`` 0–2 as the only inference-side
defense — but their published *thinking-coding* preset (which we use)
sets ``presence_penalty=0.0``, leaving the deployment without a
sampling-side defense.

vLLM 0.x (PR #35451, merged 2026-03-03) provides a first-class
defense: ``RepetitionDetectionParams``. When the same ``pattern_size``
-token sequence appears ``min_count`` consecutive times at the output
tail, generation terminates with ``stop_reason="repetition_detected"``
(``vllm/v1/core/sched/utils.py:check_sequence_repetition`` and
``check_stop``). It is **opt-in**: the field defaults to ``None`` on
every request, so a client that does not explicitly set it gets no
protection.

Concrete failure observed in the agent_service deployment (May 2026):
the model tried to encode ``2·10^50`` as a string literal inside a
``write_file`` tool call, falling into a degenerate ``0``-token
attractor and emitting 32,768 literal ``0`` characters before
``max_tokens`` hit. The qwen-code CLI rejected the truncated tool
call but retained the failed ``tool_use`` block in conversation
history, so a retry inflated the prompt by the same 32K of garbage
and a third request overflowed ``max_model_len=152000`` by exactly
one token. Two retries, one session aborted, no usable artifact.

This patch enforces ``repetition_detection`` SERVER-SIDE for any
client that does not explicitly send the field, just like
``monkey_patch_default_sampling_params`` enforces the seven Qwen3.6
sampling Best-Practices fields. The values:

  * ``max_pattern_size=8``    — catches loops up to 8-token cycles
  * ``min_pattern_size=1``    — catches the single-token wall-of-zeros
  * ``min_count=24``          — lets legitimate ≤23-digit decimal
                                literals complete (int64 boundaries
                                near 9.2×10¹⁸ are 19 digits) while
                                bounding the worst-case degenerate
                                burst to ~24 wasted tokens instead
                                of ``max_tokens``

This is intentionally a SEPARATE patch from
``monkey_patch_default_sampling_params``: that patch enforces
*Alibaba-published* Qwen3.6 Best Practices. ``repetition_detection``
is an *operational layered defense* — the model's training does not
give us this knob; vLLM's runtime does. The two have independent
removal triggers (vendor changes its Best Practices vs. vLLM gains a
non-opt-in default), so they live in distinct files.

Composition with ``monkey_patch_default_sampling_params``: the install
order in ``sitecustomize.py:_PATCH_MODULES`` places this patch AFTER
the seven-field-defaults patch, so this wrapper is the OUTERMOST
``to_sampling_params``. Each call: outer (this) → inner (Qwen3.6
seven-field defaults) → original. Both sets of defaults apply; both
respect ``model_fields_set`` so explicit client values are preserved.

Target: ``vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionRequest.to_sampling_params``
(same target as patch §7.6). The wrapper delegates to whatever is
currently bound (the seven-field-defaults wrapper or, in test
isolation, the unpatched original), then sets
``SamplingParams.repetition_detection`` iff the request did not carry
``repetition_detection`` in ``model_fields_set``.

**Removal trigger**: vLLM ships ``repetition_detection`` as a
non-opt-in default (e.g., enabled by ``--override-generation-config``
honouring nested objects, or a dedicated CLI flag), OR Alibaba adds
``repetition_detection`` to the published Qwen3.6 Best Practices block
(at which point the §7.6 patch absorbs it).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.entrypoints.openai.chat_completion import protocol as _protocol_mod
from vllm.sampling_params import RepetitionDetectionParams, SamplingParams


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-repetition-detection-default-v1"

# The defense values. See module docstring for rationale of each.
_RD_MAX_PATTERN_SIZE: int = 8
_RD_MIN_PATTERN_SIZE: int = 1
_RD_MIN_COUNT: int = 24


_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class RepetitionDetectionPatchRefusedError(RuntimeError):
    """A precondition for the ``repetition_detection`` server-side
    default wrapper was violated. Raised at import time only; the patch
    either applies cleanly or the process does not come up."""


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise RepetitionDetectionPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {msg}"
        )


# --------------------------------------------------------------------
# Phase 1: Locate the target class and currently-bound method.
# --------------------------------------------------------------------
# This patch composes with monkey_patch_default_sampling_params, which
# wraps the same method. Whichever loaded earlier installed its wrapper
# at this attribute; we wrap WHATEVER IS CURRENTLY BOUND. Walking the
# `__wrapped_original__` chain is the launcher's job — each wrapper
# stamps its own ``__qwen36_patch__`` and the launcher's chain walker
# (``_find_in_patch_chain``) finds them by tag.

_RequestCls = getattr(_protocol_mod, "ChatCompletionRequest", None)
_require(
    _RequestCls is not None and inspect.isclass(_RequestCls),
    "vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionRequest "
    "is missing or not a class.",
)

_currently_bound = getattr(_RequestCls, "to_sampling_params", None)
_require(
    _currently_bound is not None and callable(_currently_bound),
    "ChatCompletionRequest.to_sampling_params missing or not callable.",
)
_require(
    getattr(_currently_bound, "__qwen36_patch__", None) != _PATCH_TAG,
    f"ChatCompletionRequest.to_sampling_params already carries our tag "
    f"{_PATCH_TAG!r}; refusing to double-install.",
)


# --------------------------------------------------------------------
# Phase 2: Verify the request schema still carries ``repetition_detection``
# as a field whose presence in ``model_fields_set`` we rely on.
# --------------------------------------------------------------------

_model_fields = getattr(_RequestCls, "model_fields", None)
_require(
    isinstance(_model_fields, dict) and _model_fields,
    "ChatCompletionRequest.model_fields is missing or empty; not a "
    "Pydantic v2 BaseModel any more.",
)
_require(
    "repetition_detection" in _model_fields,
    "ChatCompletionRequest.model_fields no longer contains "
    "'repetition_detection' — vLLM rolled back PR #35451, OR a request-"
    "schema refactor renamed the field. The patch's notion of 'did the "
    "client send X?' is no longer valid against this vLLM build.",
)


# --------------------------------------------------------------------
# Phase 3: Verify ``RepetitionDetectionParams`` validates our values
# at construction time. ``__post_init__`` raises ``ValueError`` if
# (max_pattern_size < 0) | (min_pattern_size < 0) |
# (min_pattern_size > max_pattern_size) | (min_count < 2). Construct
# eagerly so the launcher fails at import, not at first request.
# --------------------------------------------------------------------

try:
    _construction_probe = RepetitionDetectionParams(
        max_pattern_size=_RD_MAX_PATTERN_SIZE,
        min_pattern_size=_RD_MIN_PATTERN_SIZE,
        min_count=_RD_MIN_COUNT,
    )
except (TypeError, ValueError) as _exc:
    raise RepetitionDetectionPatchRefusedError(
        f"[{_PATCH_TAG}] RepetitionDetectionParams("
        f"max_pattern_size={_RD_MAX_PATTERN_SIZE}, "
        f"min_pattern_size={_RD_MIN_PATTERN_SIZE}, "
        f"min_count={_RD_MIN_COUNT}) failed validation: {_exc!r}"
    ) from _exc

_require(
    isinstance(_construction_probe, RepetitionDetectionParams),
    f"RepetitionDetectionParams constructor returned a non-instance "
    f"({type(_construction_probe).__name__!r}).",
)
del _construction_probe  # do NOT share a single instance across requests


# --------------------------------------------------------------------
# Phase 4: Verify ``SamplingParams.repetition_detection`` is the
# attribute we will assign to. If vLLM renamed the field on the
# server-side type, our setattr would silently land on the instance's
# __dict__ instead of the documented attribute.
# --------------------------------------------------------------------

_sampling_attrs = {
    name for name in dir(SamplingParams) if not name.startswith("_")
}
_require(
    "repetition_detection" in _sampling_attrs,
    "SamplingParams has no attribute 'repetition_detection'; the wrapper's "
    "setattr would silently land on the instance dict instead of the "
    "scheduler-readable field.",
)


# --------------------------------------------------------------------
# Phase 5: Verify the load-bearing primitive on a real instance —
# ``BaseModel.model_fields_set`` correctly excludes the unset
# ``repetition_detection`` field. If a future Pydantic / vLLM combo
# changed the semantics (e.g., ``model_validator(mode='before')``
# pre-populating the field), our "client did not send X" check would
# silently lie.
# --------------------------------------------------------------------

_unset_probe = _RequestCls(model="probe", messages=[])
_require(
    "repetition_detection" not in _unset_probe.model_fields_set,
    f"primitive failure: 'repetition_detection' unexpectedly in "
    f"model_fields_set of a freshly-built request with no sampling "
    f"fields. model_fields_set={_unset_probe.model_fields_set!r}.",
)

_set_probe = _RequestCls(
    model="probe",
    messages=[],
    repetition_detection=RepetitionDetectionParams(
        max_pattern_size=4, min_pattern_size=1, min_count=8
    ),
)
_require(
    "repetition_detection" in _set_probe.model_fields_set,
    f"primitive failure: 'repetition_detection' missing from "
    f"model_fields_set of a request constructed with an explicit value. "
    f"model_fields_set={_set_probe.model_fields_set!r}.",
)


# --------------------------------------------------------------------
# Phase 6: The wrapper.
# --------------------------------------------------------------------


@functools.wraps(_currently_bound)
def _to_sampling_params_with_repetition_detection(
    self: Any,
    max_tokens: int,
    default_sampling_params: dict,
) -> SamplingParams:
    """Run whatever is currently bound to ``to_sampling_params`` (which
    may itself be a previous patch's wrapper, e.g. patch §7.6's seven-
    field-defaults wrapper), then set
    ``SamplingParams.repetition_detection`` iff the client did NOT
    explicitly send the field on this request.

    Construct a FRESH ``RepetitionDetectionParams`` per call — never
    share a module-level instance, since the dataclass is not frozen
    and a downstream consumer could mutate it.
    """
    sampling = _currently_bound(self, max_tokens, default_sampling_params)

    if "repetition_detection" not in self.model_fields_set:
        sampling.repetition_detection = RepetitionDetectionParams(
            max_pattern_size=_RD_MAX_PATTERN_SIZE,
            min_pattern_size=_RD_MIN_PATTERN_SIZE,
            min_count=_RD_MIN_COUNT,
        )

    return sampling


_to_sampling_params_with_repetition_detection.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_to_sampling_params_with_repetition_detection.__wrapped_original__ = _currently_bound  # type: ignore[attr-defined]


# --------------------------------------------------------------------
# Phase 7: Install and verify both dynamic and static lookups resolve.
# --------------------------------------------------------------------

_RequestCls.to_sampling_params = _to_sampling_params_with_repetition_detection

_installed = getattr(_RequestCls, "to_sampling_params")
_require(
    getattr(_installed, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install: tag absent via attribute access on "
    "ChatCompletionRequest.to_sampling_params.",
)
_resolved_static = inspect.getattr_static(_RequestCls, "to_sampling_params")
_require(
    getattr(_resolved_static, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install: inspect.getattr_static disagrees with attribute access; "
    "a metaclass shim or descriptor is shadowing the install.",
)


# --------------------------------------------------------------------
# Phase 8: Behavioural verification on real ChatCompletionRequest
# instances. Tag-only checks pass even when the wrapper silently
# miscomputes its overrides, so exercise both the apply-default and
# preserve-explicit branches.
# --------------------------------------------------------------------

# Case 1: client sent NO sampling fields — repetition_detection must be
# our defense triple.
_req_unset = _RequestCls(model="probe", messages=[])
_sp_unset = _req_unset.to_sampling_params(
    max_tokens=4096,
    default_sampling_params={},
)
_require(
    isinstance(_sp_unset, SamplingParams),
    f"Phase 8 case 1: to_sampling_params returned non-SamplingParams "
    f"({type(_sp_unset).__name__!r}).",
)
_rd_actual = _sp_unset.repetition_detection
_require(
    isinstance(_rd_actual, RepetitionDetectionParams),
    f"Phase 8 case 1: SamplingParams.repetition_detection is "
    f"{type(_rd_actual).__name__!r}, expected RepetitionDetectionParams.",
)
_require(
    _rd_actual.max_pattern_size == _RD_MAX_PATTERN_SIZE
    and _rd_actual.min_pattern_size == _RD_MIN_PATTERN_SIZE
    and _rd_actual.min_count == _RD_MIN_COUNT,
    f"Phase 8 case 1: RepetitionDetectionParams fields "
    f"({_rd_actual.max_pattern_size}, {_rd_actual.min_pattern_size}, "
    f"{_rd_actual.min_count}) != expected "
    f"({_RD_MAX_PATTERN_SIZE}, {_RD_MIN_PATTERN_SIZE}, {_RD_MIN_COUNT}).",
)

# Case 2: client sent EXPLICIT repetition_detection. It MUST be
# preserved (negative control proving we don't override unconditionally).
_explicit_rd = RepetitionDetectionParams(
    max_pattern_size=2, min_pattern_size=1, min_count=4
)
_req_explicit = _RequestCls(
    model="probe", messages=[], repetition_detection=_explicit_rd
)
_sp_explicit = _req_explicit.to_sampling_params(
    max_tokens=4096,
    default_sampling_params={},
)
_rd_preserved = _sp_explicit.repetition_detection
_require(
    isinstance(_rd_preserved, RepetitionDetectionParams),
    f"Phase 8 case 2: explicit repetition_detection became "
    f"{type(_rd_preserved).__name__!r}.",
)
_require(
    _rd_preserved.max_pattern_size == 2
    and _rd_preserved.min_pattern_size == 1
    and _rd_preserved.min_count == 4,
    f"Phase 8 case 2: explicit repetition_detection was overridden — "
    f"got ({_rd_preserved.max_pattern_size}, "
    f"{_rd_preserved.min_pattern_size}, {_rd_preserved.min_count}) "
    f"instead of preserved (2, 1, 4). The wrapper is rewriting client "
    f"intent.",
)

# Case 3: each fresh call must produce a fresh
# RepetitionDetectionParams instance. If we shared a module-level
# instance across requests, mutation by one consumer would leak into
# others. Two separate calls, each with no fields set, must NOT alias.
_req_fresh_a = _RequestCls(model="probe", messages=[])
_req_fresh_b = _RequestCls(model="probe", messages=[])
_sp_a = _req_fresh_a.to_sampling_params(
    max_tokens=4096, default_sampling_params={}
)
_sp_b = _req_fresh_b.to_sampling_params(
    max_tokens=4096, default_sampling_params={}
)
_require(
    _sp_a.repetition_detection is not _sp_b.repetition_detection,
    "Phase 8 case 3: two separate requests share the same "
    "RepetitionDetectionParams instance — mutation by one consumer "
    "would leak across requests.",
)


_logger.info(
    "[%s] applied: wrapped %s.ChatCompletionRequest.to_sampling_params "
    "for vLLM commit %s (server-side repetition_detection default; only "
    "fills the field when the client did not explicitly send it).",
    _PATCH_TAG,
    _protocol_mod.__name__,
    _PINNED_VLLM_COMMIT,
)
