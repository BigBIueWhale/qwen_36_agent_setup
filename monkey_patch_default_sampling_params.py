"""Strict, fail-loud server-side enforcement of Qwen3.6 sampling defaults.

Why this patch must exist
-------------------------

vLLM honours whatever sampling fields the client sends. Qwen3.6 has a
"Best Practices" sampling block on its model card whose values matter
load-bearingly: at the vLLM defaults (``temperature=1.0`` only when
nothing is supplied, ``presence_penalty=0.0`` always, ``top_p=1.0``,
``top_k=0``, ``min_p=0.0``) the model exhibits a thinking-mode loop
pathology measured at 17.4% truncation on Qwen3.5-35B-A3B's
LiveCodeBench (27.5% on hard problems). Alibaba ships
``presence_penalty=1.5`` as the documented mitigation, plus the
sampling-shape combo (``temperature=1.0, top_p=0.95, top_k=20,
min_p=0.0, repetition_penalty=1.0, max_tokens=16384``). README §5.6.

Default-clients (OpenAI Python SDK, OpenAI-compatible TypeScript SDKs,
the Qwen Code CLI, Qwen-Agent) do not send Qwen3.6's recommended
sampling defaults. Without this patch, the agent backend operates in
the broken-by-default region every time a client omits the fields.

This patch enforces the Qwen3.6 thinking-mode defaults SERVER-SIDE
for fields the client did NOT explicitly send. **Only fills defaults
for unset fields** — if the client EXPLICITLY sends ``temperature=0.6``
("precise coding mode"), 0.6 is preserved. The detection primitive is
Pydantic v2's ``BaseModel.model_fields_set`` (set of field names that
were assigned during validation; excludes class defaults), which is
load-bearing-verified at import time on a real
``ChatCompletionRequest`` instance constructed with no sampling fields.

Target: ``vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionRequest.to_sampling_params``
at vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce``. The wrapper
delegates to the original, then rewrites the returned ``SamplingParams``
fields. The same method is invoked from
``chat_completion/serving.py:300`` (online) and
``chat_completion/batch_serving.py:162`` (batched); both paths benefit.

**Removal trigger**: vLLM gains a first-class "model-recommended sampling
defaults" mechanism (e.g. ``--default-sampling-params`` flag widened to
respect ``model_fields_set``) and the deployment's launch flags adopt
it.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.entrypoints.openai.chat_completion import protocol as _protocol_mod
from vllm.sampling_params import SamplingParams


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-default-sampling-params-v1"

# Qwen3.6 Best Practices, thinking-mode / agentic-correct defaults.
# README §5.6. Mirrors the model-card "Best Practices" block.
QWEN36_DEFAULTS: dict[str, float | int] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 16384,
}

# Source landmark — substring required in the wrapped method body so
# an upstream refactor of how the request constructs its SamplingParams
# forces a re-audit before this patch silently rewrites a different
# code path's outputs.
_FROM_OPTIONAL_LANDMARK: str = "SamplingParams.from_optional("

_EXPECTED_PARAMS: list[str] = ["self", "max_tokens", "default_sampling_params"]

# Field names the request carries that we need present BEFORE we trust
# `model_fields_set`. If any of these vanished upstream, the request
# protocol has been redesigned and our "did the client send it?" probe
# is no longer meaningful.
_REQUEST_FIELDS_REQUIRED: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "max_tokens",
    "max_completion_tokens",
    "messages",
    "model",
)


_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class DefaultSamplingParamsPatchRefusedError(RuntimeError):
    """A precondition for the Qwen3.6 default-sampling-params wrapper was
    violated. Raised at import time only; the patch either applies
    cleanly or the process does not come up."""


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise DefaultSamplingParamsPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {msg}"
        )


# --------------------------------------------------------------------
# Phase 1: Locate the target class and method.
# --------------------------------------------------------------------

_RequestCls = getattr(_protocol_mod, "ChatCompletionRequest", None)
_require(
    _RequestCls is not None and inspect.isclass(_RequestCls),
    "vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionRequest "
    "is missing or not a class.",
)

_original = getattr(_RequestCls, "to_sampling_params", None)
_require(
    _original is not None and callable(_original),
    "ChatCompletionRequest.to_sampling_params missing or not callable.",
)
_require(
    getattr(_original, "__qwen36_patch__", None) is None,
    f"ChatCompletionRequest.to_sampling_params already carries a "
    f"__qwen36_patch__ tag {getattr(_original, '__qwen36_patch__', None)!r}; "
    f"refusing to double-install.",
)


# --------------------------------------------------------------------
# Phase 2: Verify signature and source landmark.
# --------------------------------------------------------------------

try:
    _sig = inspect.signature(_original)  # type: ignore[arg-type]
    _src = inspect.getsource(_original)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise DefaultSamplingParamsPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect ChatCompletionRequest."
        f"to_sampling_params: {_exc!r}"
    ) from _exc

_param_names = list(_sig.parameters)
_require(
    _param_names == _EXPECTED_PARAMS,
    f"to_sampling_params signature changed; expected {_EXPECTED_PARAMS!r}, "
    f"got {_param_names!r}.",
)

_require(
    _FROM_OPTIONAL_LANDMARK in _src,
    f"landmark {_FROM_OPTIONAL_LANDMARK!r} missing from to_sampling_params "
    f"source — upstream restructured the SamplingParams construction; the "
    f"wrapper's "
    f"'delegate-then-overwrite-fields' contract is no longer safe. "
    f"Re-audit before bumping the pinned commit.",
)


# --------------------------------------------------------------------
# Phase 3: Verify the request schema still carries the fields whose
# presence in `model_fields_set` we rely on.
# --------------------------------------------------------------------

_model_fields = getattr(_RequestCls, "model_fields", None)
_require(
    isinstance(_model_fields, dict) and _model_fields,
    "ChatCompletionRequest.model_fields is missing or empty; not a "
    "Pydantic v2 BaseModel any more.",
)
for _required in _REQUEST_FIELDS_REQUIRED:
    _require(
        _required in _model_fields,
        f"ChatCompletionRequest.model_fields no longer contains "
        f"{_required!r}; the patch's notion of 'did the client send X?' "
        f"is no longer valid.",
    )

# Verify each Qwen3.6 default field is also a field on SamplingParams
# itself (we mutate by setattr after construction).
_sampling_attrs = {
    name
    for name in dir(SamplingParams)
    if not name.startswith("_")
}
for _name in QWEN36_DEFAULTS:
    _require(
        _name in _sampling_attrs,
        f"SamplingParams has no attribute {_name!r}; the wrapper's "
        f"setattr-after-construction would silently no-op.",
    )


# --------------------------------------------------------------------
# Phase 4: Verify the load-bearing primitive on a real instance.
# --------------------------------------------------------------------
# `BaseModel.model_fields_set` is the load-bearing primitive. If a
# future Pydantic / vLLM combination changed its semantics (e.g.,
# because of a `model_validator(mode='before')` that explicitly sets
# defaults), our "client did not send X" check would silently lie.
# Build a real ChatCompletionRequest with NO sampling fields and assert
# none of the Qwen3.6 keys ended up in `model_fields_set`.

_unset_probe = _RequestCls(model="probe", messages=[])
for _name in QWEN36_DEFAULTS:
    _require(
        _name not in _unset_probe.model_fields_set,
        f"primitive failure: {_name!r} unexpectedly in model_fields_set "
        f"of a freshly-built request with no sampling fields. "
        f"model_fields_set={_unset_probe.model_fields_set!r}.",
    )

# Mirror probe: a request with `temperature=0.5` MUST have it in the set.
_set_probe = _RequestCls(model="probe", messages=[], temperature=0.5)
_require(
    "temperature" in _set_probe.model_fields_set,
    f"primitive failure: 'temperature' missing from model_fields_set of "
    f"a request constructed with temperature=0.5. "
    f"model_fields_set={_set_probe.model_fields_set!r}.",
)


# --------------------------------------------------------------------
# Phase 5: The wrapper.
# --------------------------------------------------------------------


@functools.wraps(_original)
def _to_sampling_params_with_qwen36_defaults(
    self: Any,
    max_tokens: int,
    default_sampling_params: dict,
) -> SamplingParams:
    """Run the original ``to_sampling_params`` then, for each Qwen3.6
    Best-Practices default, override the corresponding SamplingParams
    field iff the client did NOT explicitly send it on this request.

    Special-case ``max_tokens``: the request carries it via TWO fields
    (``max_tokens`` and ``max_completion_tokens``). We treat either
    explicit setting as "client expressed an opinion". The original
    method receives an already-bounded ``max_tokens`` arg (capped at
    ``model_max_len - input_len`` by the serving layer's
    ``get_max_tokens``); to never EXCEED that cap, we apply the Qwen3.6
    16384 default as ``min(qwen_default, current)``, never as a raw
    overwrite.
    """
    sampling = _original(self, max_tokens, default_sampling_params)
    fields_set = self.model_fields_set

    for name, qwen_default in QWEN36_DEFAULTS.items():
        if name == "max_tokens":
            continue  # handled below — capped, not overwritten
        if name in fields_set:
            continue
        setattr(sampling, name, qwen_default)

    if (
        "max_tokens" not in fields_set
        and "max_completion_tokens" not in fields_set
    ):
        current_max = sampling.max_tokens
        qwen_max = QWEN36_DEFAULTS["max_tokens"]
        if current_max is None:
            sampling.max_tokens = qwen_max
        else:
            sampling.max_tokens = min(current_max, qwen_max)

    return sampling


_to_sampling_params_with_qwen36_defaults.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_to_sampling_params_with_qwen36_defaults.__wrapped_original__ = _original  # type: ignore[attr-defined]


# --------------------------------------------------------------------
# Phase 6: Install and verify.
# --------------------------------------------------------------------

_RequestCls.to_sampling_params = _to_sampling_params_with_qwen36_defaults

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
# Phase 7: Behavioural verification on real ChatCompletionRequest
# instances. Load-bearing — a tag-only check passes even when the
# wrapper silently miscomputes its overrides.
# --------------------------------------------------------------------

# Case 1: client sent NO sampling fields. All Qwen3.6 defaults must apply.
_req_unset = _RequestCls(model="probe", messages=[])
_sp_unset = _req_unset.to_sampling_params(
    max_tokens=200_000,  # large; verifies the max_tokens cap to 16384 fires
    default_sampling_params={},
)
_require(
    isinstance(_sp_unset, SamplingParams),
    f"Phase 7 case 1: to_sampling_params returned non-SamplingParams "
    f"({type(_sp_unset).__name__!r}).",
)
for _name, _expected in QWEN36_DEFAULTS.items():
    _actual = getattr(_sp_unset, _name)
    _require(
        _actual == _expected,
        f"Phase 7 case 1 ({_name}): expected {_expected!r}, got {_actual!r} "
        f"on a request with NO sampling fields set.",
    )

# Case 2: client sent EXPLICIT temperature=0.5. It MUST be preserved
# (negative control proving we don't override unconditionally).
_req_explicit = _RequestCls(model="probe", messages=[], temperature=0.5)
_sp_explicit = _req_explicit.to_sampling_params(
    max_tokens=4096,
    default_sampling_params={},
)
_require(
    _sp_explicit.temperature == 0.5,
    f"Phase 7 case 2: explicit temperature=0.5 was overridden to "
    f"{_sp_explicit.temperature!r}; the patch is rewriting client intent.",
)
# All OTHER Qwen3.6 defaults still apply (only temperature was explicit).
for _name, _expected in QWEN36_DEFAULTS.items():
    if _name in ("temperature", "max_tokens"):
        continue
    _actual = getattr(_sp_explicit, _name)
    _require(
        _actual == _expected,
        f"Phase 7 case 2 ({_name}): expected {_expected!r}, got {_actual!r} "
        f"when only 'temperature' was explicitly set.",
    )

# Case 3: max_tokens cap is min(qwen_default, current), never an
# unconditional overwrite. Use a small `max_tokens` arg to prove we do
# not raise it from 1024 to 16384 when the model_max_len cap is small.
_req_unset_2 = _RequestCls(model="probe", messages=[])
_sp_capped = _req_unset_2.to_sampling_params(
    max_tokens=1024,
    default_sampling_params={},
)
_require(
    _sp_capped.max_tokens == 1024,
    f"Phase 7 case 3: small max_tokens=1024 cap was overridden to "
    f"{_sp_capped.max_tokens!r}; the wrapper is exceeding the serving "
    f"layer's already-applied cap.",
)

# Case 4: explicit presence_penalty=0.0 (i.e. precise-coding mode) is
# preserved — pydantic field's class default is 0.0 but explicit
# assignment still appears in model_fields_set.
_req_pp_zero = _RequestCls(model="probe", messages=[], presence_penalty=0.0)
_sp_pp_zero = _req_pp_zero.to_sampling_params(
    max_tokens=4096,
    default_sampling_params={},
)
_require(
    _sp_pp_zero.presence_penalty == 0.0,
    f"Phase 7 case 4: explicit presence_penalty=0.0 was overridden to "
    f"{_sp_pp_zero.presence_penalty!r}; the patch is ignoring "
    f"`model_fields_set` for non-None-default fields.",
)

# Case 5: explicit max_completion_tokens (the OpenAI-recommended field
# name) is also recognized as "client expressed an opinion" so we do
# not impose the 16384 cap.
_req_mct = _RequestCls(model="probe", messages=[], max_completion_tokens=2048)
_sp_mct = _req_mct.to_sampling_params(
    max_tokens=2048,
    default_sampling_params={},
)
_require(
    _sp_mct.max_tokens == 2048,
    f"Phase 7 case 5: explicit max_completion_tokens=2048 yielded "
    f"SamplingParams.max_tokens={_sp_mct.max_tokens!r}; the patch is not "
    f"recognising max_completion_tokens as 'client expressed an opinion'.",
)


_logger.info(
    "[%s] applied: wrapped %s.ChatCompletionRequest.to_sampling_params for "
    "vLLM commit %s (server-side Qwen3.6 sampling defaults; only fills "
    "fields the client did not explicitly send).",
    _PATCH_TAG,
    _protocol_mod.__name__,
    _PINNED_VLLM_COMMIT,
)
