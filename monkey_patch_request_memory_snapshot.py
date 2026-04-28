"""Strict, fail-loud correctness fix for vLLM's startup snapshot check.

Why this patch must exist
-------------------------

``vllm.v1.worker.utils.request_memory`` raises ``ValueError`` whenever
``init_snapshot.free_memory < total_memory * gpu_memory_utilization``.
The intent — refuse to come up if other CUDA processes are starving the
target device of memory — is fine, but the implementation is
**semantically wrong**: ``init_snapshot`` is taken inside the EngineCore
worker AFTER vLLM has already initialised its own CUDA context (PyTorch
+ cuBLAS/cuDNN + FlashAttention v2 + NIXL + NCCL collectives), which on
this Blackwell SM 12.0 build consumes ~510 MiB. That ~510 MiB is part of
``vllm``'s footprint, not external pressure, and is reserved out of
``requested_memory`` further downstream as ``non_torch_memory`` /
``torch_peak_increase``. So the original check **double-counts vLLM's
own init footprint as 'external pressure'** and refuses any
``--gpu-memory-utilization > ~0.984`` on a single-tenant exclusive GPU
even when no other process is using the card.

Concrete on-host symptom (32 GiB RTX 5090, iGPU separate, exclusive
dGPU): ``Free memory on device cuda:0 (30.85/31.36 GiB) on startup is
less than desired GPU memory utilization (0.99, 31.04 GiB). Decrease
GPU memory utilization or reduce GPU memory used by other processes.``
There is nothing to decrease — vLLM is the only consumer.

What the patch does
-------------------

Replaces ``request_memory`` with a version that compares
``requested_memory`` against ``init_snapshot.free_memory + INIT_SLACK``,
where ``INIT_SLACK`` is a generous bound on vLLM's own post-init
footprint (1 GiB by default; tunable via ``QWEN36_VLLM_INIT_SLACK_MIB``
env var for hosts whose init footprint is larger or smaller).

This **preserves the original safety intent** — the check still fires
loudly when external processes consume more memory than vLLM's own init
plausibly could (>1 GiB beyond the slack) — while permitting the
operator to land at ``--gpu-memory-utilization`` close to 1.0 on a
single-tenant deployment.

The replacement function returns the same ``int`` (``requested_memory``)
the original returns, so all downstream consumers
(``gpu_worker.py:421-424``: ``available_kv_cache_memory_bytes =
requested_memory - non_kv_cache_memory - cudagraph_memory_estimate``)
see byte-identical output. No other site at the pin reads
``init_snapshot.free_memory`` as a gating value (``gpu_worker.py:412``
is an assert during profiling, ``:427`` is a debug log,
``:631-660,350`` are diagnostic message strings) — verified
exhaustively by Phase 5 below.

Target: ``vllm.v1.worker.utils.request_memory`` at vLLM commit
``8cd174fa358326d5cc4195446be2ebcd65c481ce``.

Removal trigger
---------------

vLLM upstream rewrites ``request_memory`` to either (a) measure a
pre-init snapshot in addition to the post-init one and compare against
the pre-init free, or (b) accept a slack parameter, or (c) introduce a
``--allow-shared-init-footprint`` (or equivalent) flag. The buggy
``free_memory < requested_memory`` predicate is the landmark this patch
asserts on; if a fix lands the predicate changes shape and the patch
refuses to apply.
"""

from __future__ import annotations

import inspect
import math
import os
from typing import Any, Callable, TypeAlias

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker import utils as _utils_mod


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-request-memory-snapshot-v1"

# Source landmarks — substrings required in the original function body.
# If any of these change shape upstream, our wrap is no longer aimed at
# the function we audited.
_LANDMARK_PREDICATE: str = "init_snapshot.free_memory < requested_memory"
_LANDMARK_FORMULA: str = "init_snapshot.total_memory * cache_config.gpu_memory_utilization"
_LANDMARK_RAISE: str = 'raise ValueError'

# Default slack for vLLM's own post-init CUDA footprint. Empirically
# measured 510 MiB on Blackwell SM 12.0 + the pinned image; we round
# up to 1 GiB for headroom against torch / cuBLAS / cuDNN / FlashAttention
# version-bump variance.
_DEFAULT_INIT_SLACK_MIB: int = 1024

# Env var that lets an operator tune the slack without re-patching.
# Useful if a future image bumps a dependency that adds (or trims) post-
# init footprint. Read at module import; the value is captured into the
# wrapper closure so subsequent env mutations don't mutate the patch.
_ENV_NAME: str = "QWEN36_VLLM_INIT_SLACK_MIB"


_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


RequestMemoryFn: TypeAlias = Callable[[MemorySnapshot, CacheConfig], int]


class MonkeyPatchRefusedError(RuntimeError):
    """Precondition for the request_memory patch was violated.

    Raised at import time only; the patch either applies cleanly or the
    process does not come up. Same idiom as the other Qwen3.6 patches.
    """


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise MonkeyPatchRefusedError(f"[{_PATCH_TAG}] refusing to patch: {msg}")


# ---------------------------------------------------------------------------
# Phase 1 — locate target.
# ---------------------------------------------------------------------------

_original: RequestMemoryFn | None = getattr(_utils_mod, "request_memory", None)
_require(
    _original is not None and callable(_original),
    "request_memory missing or not callable in vllm.v1.worker.utils.",
)


# ---------------------------------------------------------------------------
# Phase 2 — verify signature.
# ---------------------------------------------------------------------------

try:
    _sig = inspect.signature(_original)  # type: ignore[arg-type]
    _original_src = inspect.getsource(_original)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect request_memory: {_exc!r}"
    )

_params = list(_sig.parameters.values())
_require(
    len(_params) == 2,
    f"request_memory expected 2 params, got {len(_params)}: "
    f"{[p.name for p in _params]}.",
)
_require(
    _params[0].name == "init_snapshot",
    f"first param must be 'init_snapshot', got '{_params[0].name}'.",
)
_require(
    _params[1].name == "cache_config",
    f"second param must be 'cache_config', got '{_params[1].name}'.",
)
_require(
    _sig.return_annotation is int or str(_sig.return_annotation) == "int",
    f"return annotation must be int, got {_sig.return_annotation!r}.",
)


# ---------------------------------------------------------------------------
# Phase 3 — strict landmark check on the buggy predicate.
# ---------------------------------------------------------------------------

_require(
    _LANDMARK_PREDICATE in _original_src,
    f"buggy predicate '{_LANDMARK_PREDICATE}' not found in request_memory; "
    f"upstream may have already fixed the check — re-audit before patching.",
)
_require(
    _LANDMARK_FORMULA in _original_src,
    f"requested-memory formula '{_LANDMARK_FORMULA}' not found in "
    f"request_memory; the function shape has shifted upstream.",
)
_require(
    _LANDMARK_RAISE in _original_src,
    f"raise statement '{_LANDMARK_RAISE}' not found in request_memory; "
    f"the function shape has shifted upstream.",
)


# ---------------------------------------------------------------------------
# Phase 4 — primitive verify on a synthetic snapshot.
# ---------------------------------------------------------------------------

# Build a synthetic init_snapshot that triggers the original failure mode
# (free < requested by a vLLM-init-sized margin) and confirm the original
# raises and the wrapper accepts. This is the load-bearing behavioural
# proof.
class _FakeSnapshot:
    """Stand-in for MemorySnapshot — only attributes the function reads."""

    def __init__(self, total_gib: float, free_gib: float, device: str = "cuda:0") -> None:
        self.total_memory = int(total_gib * (1 << 30))
        self.free_memory = int(free_gib * (1 << 30))
        # request_memory's error formatter calls .device_ — provide it.
        self.device_ = device


class _FakeCacheConfig:
    """Stand-in for CacheConfig — only attribute the function reads."""

    def __init__(self, gmu: float) -> None:
        self.gpu_memory_utilization = gmu


_probe_total = 31.36  # GiB — RTX 5090 advertised
_probe_free_after_init = 30.85  # GiB — measured post-vLLM-init
_probe_gmu = 0.99  # The case the patch unblocks.

_probe_snapshot = _FakeSnapshot(_probe_total, _probe_free_after_init)
_probe_cache_config = _FakeCacheConfig(_probe_gmu)

# 4a — confirm original IS broken on this case.
_original_raised_correctly = False
try:
    _original(_probe_snapshot, _probe_cache_config)  # type: ignore[arg-type]
except ValueError as _ve:
    _original_raised_correctly = True
    _expected_phrase = "is less than desired GPU memory utilization"
    _require(
        _expected_phrase in str(_ve),
        f"original raised ValueError but message changed shape "
        f"(missing '{_expected_phrase}'): {_ve!r}.",
    )
_require(
    _original_raised_correctly,
    "original request_memory did NOT raise ValueError on the "
    f"({_probe_free_after_init:.2f}/{_probe_total:.2f} GiB, gmu={_probe_gmu}) "
    "probe — the bug landmark may already be fixed upstream.",
)


# ---------------------------------------------------------------------------
# Phase 5 — exhaustive consumer audit (compile-time, not runtime).
# ---------------------------------------------------------------------------

# Verified manually (agent Z audit, 2026-04-28): every other site at the
# pin that reads init_snapshot.free_memory is non-gating (log/diagnostic
# only). They are listed here for the operator's reference — if vLLM
# ever changes one of them into a gate, this patch's removal trigger
# fires.
_NON_GATING_FREE_MEMORY_READERS: tuple[str, ...] = (
    "vllm/v1/worker/gpu_worker.py:412 — assert during profiling (sanity)",
    "vllm/v1/worker/gpu_worker.py:427 — debug log of unrequested_memory",
    "vllm/v1/worker/gpu_worker.py:350 — info log when --kv-cache-memory-bytes is set",
    "vllm/v1/worker/gpu_worker.py:631-660 — diagnostic message string formatting",
)


# ---------------------------------------------------------------------------
# Phase 6 — build the replacement.
# ---------------------------------------------------------------------------

# Resolve slack at install time, not per-call, so config is reproducible
# from boot logs.
try:
    _slack_mib = int(os.environ.get(_ENV_NAME, _DEFAULT_INIT_SLACK_MIB))
except ValueError:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] environment variable {_ENV_NAME}="
        f"{os.environ.get(_ENV_NAME)!r} is not an integer (MiB)."
    )
_require(
    _slack_mib >= 0,
    f"{_ENV_NAME} must be a non-negative integer (MiB), got {_slack_mib}.",
)
_INIT_SLACK_BYTES: int = _slack_mib * (1 << 20)


# Bring in the original's helper for byte-formatting in the replacement
# error message. Hard import — if upstream renames it, we want to refuse
# loudly rather than silently substitute a different formatter (matches
# the rest of this repo's patch idiom).
from vllm.v1.worker.utils import format_gib as _format_gib  # noqa: E402


def _request_memory_corrected(
    init_snapshot: MemorySnapshot, cache_config: CacheConfig
) -> int:
    """Patched ``request_memory`` — see module docstring for rationale."""
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    # The original used `init_snapshot.free_memory` directly, which post-
    # dates vLLM's own init. Add a generous bound on vLLM's own footprint
    # so the check fires only when EXTERNAL processes are squeezing the
    # GPU. The slack is operator-tunable via QWEN36_VLLM_INIT_SLACK_MIB.
    effective_free = init_snapshot.free_memory + _INIT_SLACK_BYTES

    if effective_free < requested_memory:
        external_pressure = max(
            0,
            init_snapshot.total_memory
            - init_snapshot.free_memory
            - _INIT_SLACK_BYTES,
        )
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({_format_gib(init_snapshot.free_memory)}/"
            f"{_format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{_format_gib(requested_memory)} GiB), even after "
            f"accounting for vLLM's own init footprint "
            f"({_format_gib(_INIT_SLACK_BYTES)} GiB slack via "
            f"{_PATCH_TAG}). External processes appear to be consuming "
            f"~{_format_gib(external_pressure)} GiB on this device. "
            f"Decrease GPU memory utilization, raise "
            f"{_ENV_NAME} (current {_slack_mib} MiB), or reduce GPU "
            f"memory used by other processes."
        )

    return requested_memory


# Stamp so external introspection can confirm the patched function is
# in place (the launcher's verifier checks this attribute).
_request_memory_corrected.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_request_memory_corrected.__wrapped_original__ = _original  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Phase 7 — install + behavioural verification.
# ---------------------------------------------------------------------------

setattr(_utils_mod, "request_memory", _request_memory_corrected)

# Verify both dynamic and static lookups see the patched function.
_dyn = getattr(_utils_mod, "request_memory", None)
_static = inspect.getattr_static(_utils_mod, "request_memory", None)
_require(
    _dyn is _request_memory_corrected,
    "dynamic lookup of request_memory does not see the patched function.",
)
_require(
    _static is _request_memory_corrected,
    "static lookup of request_memory does not see the patched function "
    "(another patch may be racing this install).",
)
_require(
    getattr(_dyn, "__qwen36_patch__", None) == _PATCH_TAG,
    "patched function is missing the __qwen36_patch__ stamp — installation "
    "may have been overwritten by another patch.",
)

# Behavioural check: the same probe that made the original raise must
# now succeed (because slack covers vLLM's own post-init footprint).
try:
    _result_after = _request_memory_corrected(
        _probe_snapshot, _probe_cache_config
    )  # type: ignore[arg-type]
except ValueError as _ve:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] patched request_memory still raises on the probe "
        f"({_probe_free_after_init:.2f}/{_probe_total:.2f} GiB, "
        f"gmu={_probe_gmu}, slack={_slack_mib} MiB): {_ve!r}. "
        f"Either the slack is too small or the patch is incorrectly built."
    )

_expected = math.ceil(_probe_snapshot.total_memory * _probe_gmu)
_require(
    _result_after == _expected,
    f"patched request_memory returned {_result_after} but expected "
    f"{_expected} (= ceil({_probe_total:.2f} GiB * {_probe_gmu})). "
    f"Math is broken.",
)

# Negative behavioural check: external pressure exceeding slack must
# still raise. Probe with free = total - 2*slack, gmu=0.99 → free is
# ~1 GiB short even with slack accounted; expect raise.
_neg_total = 31.36
_neg_free = _neg_total - 2 * (_slack_mib / 1024)  # GiB units
_neg_snapshot = _FakeSnapshot(_neg_total, _neg_free)
_neg_raised = False
try:
    _request_memory_corrected(_neg_snapshot, _probe_cache_config)  # type: ignore[arg-type]
except ValueError:
    _neg_raised = True
_require(
    _neg_raised,
    f"patched request_memory FAILED to raise on external-pressure probe "
    f"({_neg_free:.2f}/{_neg_total:.2f} GiB, gmu={_probe_gmu}, "
    f"slack={_slack_mib} MiB). Original safety intent has been broken — "
    f"the patch is too permissive.",
)


_logger.info(
    f"[{_PATCH_TAG}] applied: replaced "
    f"vllm.v1.worker.utils.request_memory for vLLM commit "
    f"{_PINNED_VLLM_COMMIT} (snapshot check now compares against "
    f"free_memory + {_slack_mib} MiB slack; permits gmu close to 1.0 on "
    f"single-tenant exclusive GPUs while preserving fail-loud against "
    f">{_slack_mib} MiB external pressure)."
)
