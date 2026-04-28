"""Strict, fail-loud runtime patch for vLLM hybrid-KV log under-reporting (#37121).

Why this patch must exist. Two log functions in
``vllm/v1/core/kv_cache_utils.py`` divide by
``len(kv_cache_groups)`` instead of by the count of token-capacity-
contributing groups — ``get_max_concurrency_for_kv_cache_config``
at lines 802-820 and ``_report_kv_cache_config`` at lines
1305-1346. For Qwen3.6-27B (1 full attn group + 3 GDN-mamba groups
= 4 groups; the GDN groups are O(1) per request under
``mamba_cache_mode != "all"``) this makes the boot-log lines
``GPU KV cache size: X tokens`` and ``Maximum concurrency: X.XXx``
~4× understated — at the §8.2 production flags, ``~37K tokens`` /
``0.28x`` displayed vs. ``~149K tokens`` / ``1.13x`` after this
patch. Operators sizing ``--max-model-len`` against the displayed
numbers under-utilize the card. The math is verifiably wrong;
the upstream issue is #37121 and PRs #40384 / #40694 are the open
candidate fixes.

Where these functions actually sit in the pinned commit. Both are
reached only from the boot-time log path:
``unify_kv_cache_configs:1629`` → ``_report_kv_cache_config:1305`` →
``get_max_concurrency_for_kv_cache_config:802``. Neither is wired
into the request-admission gate — that is
``KVCacheManager.can_fit_full_sequence`` at ``kv_cache_manager.py:218``,
which uses ``block_pool.get_num_free_blocks()`` on the byte-correct
shared pool and is independent of these reporters. So the
user-visible effect of the unpatched bug is one wrong number per
boot — not OOM, not wrong admission, not wrong allocation.

Scope vs PR #40384. PR #40384 in master fixes
``_report_kv_cache_config`` and a ``Scheduler.__init__`` site that
computes ``max_num_kv_tokens`` for the routed-experts capturer
buffer. This patch backports ``_report_kv_cache_config`` *and*
additionally fixes ``get_max_concurrency_for_kv_cache_config``
(same divisor pattern, second log line at boot, untouched by PR
#40384). It deliberately does NOT backport the scheduler.py site:
that site is gated on ``model_config.enable_return_routed_experts``
(default ``False`` in ``ModelConfig``) and only fires for MoE
deployments with routed-expert capture — Qwen3.6-27B is dense and
we never set the flag. Porting code under an off-by-default gate
we never trip is churn for no benefit and would only widen the
patch's removal surface. Backport semantics, not literal port —
the pinned commit's ``MambaSpec.max_memory_usage_bytes`` signature
is ``(self, vllm_config)``, not the master signature PR #40384
targets, so the helper is rebuilt locally rather than imported.

Defensive shape validation in ``_validate_kv_cache_config_shape``
is kept anyway: cheap, and refuses on a malformed KVCacheConfig
rather than emit a plausible-but-wrong float if a future upstream
change ever wires either function into a non-log consumer.

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce``.
**Removal trigger**: PR #40384 or #40694 merges. **CRITICAL**: remove
this patch BEFORE pulling an image containing PR #37429 (the broader
byte-level redesign) — that fix changes the tensor layout and this
patch's reporting view would no longer be coherent.
"""

from __future__ import annotations

import dataclasses as _dataclasses
import inspect
from typing import Any

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.v1.core import kv_cache_utils as _kv_cache_utils_mod
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
)
from vllm.config import VllmConfig
from vllm.config.scheduler import SchedulerConfig

_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-pr40384-backport"

# Multi-line landmarks: their absence at master means PR #40384 (or its
# broader sibling #37429) has landed and this patch should be deleted.
# Both are EXACTLY the buggy expressions. The shape is load-bearing —
# narrow strings prevent coincidental matches.
_CONCURRENCY_BUGGY_LANDMARK: str = (
    "num_layer_per_group = max(\n"
    "        len(group.layer_names) for group in kv_cache_config.kv_cache_groups\n"
    "    )"
)
_REPORT_BUGGY_LANDMARK: str = (
    "num_tokens = (\n"
    "        kv_cache_config.num_blocks\n"
    "        // len(kv_cache_config.kv_cache_groups)\n"
    "        * min_block_size\n"
    "    )"
)

_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class HybridKvPatchRefusedError(RuntimeError):
    """Precondition for the hybrid-KV patch was violated.

    Raised at import time only. The patched functions are log-only in
    the pinned commit (call graph in the module docstring), so a wrong
    value would corrupt the boot-log numbers operators rely on for
    capacity planning rather than OOM at request time — but a silently
    wrong boot log is exactly what makes this class of bug hard to
    notice. Refusing to boot on any landmark drift surfaces the
    divergence loudly, and forward-protects against the case where a
    future upstream change wires either function into a non-log path.
    """


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise HybridKvPatchRefusedError(f"[{_PATCH_TAG}] refusing to patch: {msg}")


# Spec-class hierarchy: replacement uses ``isinstance(spec, AttentionSpec)``
# to classify groups; a hierarchy drift would mis-classify silently.
_require(
    inspect.isclass(AttentionSpec) and inspect.isclass(MambaSpec),
    "AttentionSpec/MambaSpec no longer classes.",
)
_require(
    issubclass(FullAttentionSpec, AttentionSpec),
    "FullAttentionSpec is no longer a subclass of AttentionSpec.",
)
_require(
    not issubclass(MambaSpec, AttentionSpec),
    "MambaSpec is now a subclass of AttentionSpec — replacement would OVER-CREDIT.",
)

# KVCacheConfig / KVCacheGroupSpec field shape.
_require(
    _dataclasses.is_dataclass(KVCacheConfig)
    and _dataclasses.is_dataclass(KVCacheGroupSpec),
    "KVCacheConfig/KVCacheGroupSpec no longer dataclasses.",
)
_require(
    {"kv_cache_groups", "num_blocks"}
    <= {f.name for f in _dataclasses.fields(KVCacheConfig)},
    "KVCacheConfig fields drifted.",
)
_require(
    {"kv_cache_spec", "layer_names"}
    <= {f.name for f in _dataclasses.fields(KVCacheGroupSpec)},
    "KVCacheGroupSpec fields drifted.",
)

# max_memory_usage_bytes signature on both spec sides.
for _cls, _label in ((MambaSpec, "MambaSpec"), (FullAttentionSpec, "FullAttentionSpec")):
    _fn = getattr(_cls, "max_memory_usage_bytes", None)
    _require(callable(_fn), f"{_label}.max_memory_usage_bytes missing.")
    try:
        _params = list(inspect.signature(_fn).parameters)  # type: ignore[arg-type]
    except (TypeError, ValueError) as _exc:
        raise HybridKvPatchRefusedError(
            f"[{_PATCH_TAG}] cannot introspect {_label}.max_memory_usage_bytes: {_exc!r}"
        ) from _exc
    _require(
        _params == ["self", "vllm_config"],
        f"{_label}.max_memory_usage_bytes signature drifted; got {_params!r}.",
    )

# MambaCacheMode "all"/"none" literals — sentinel for "Mamba scales with context".
try:
    from vllm.config.cache import MambaCacheMode  # type: ignore[attr-defined]
except ImportError as _exc:
    raise HybridKvPatchRefusedError(
        f"[{_PATCH_TAG}] cannot import vllm.config.cache.MambaCacheMode: {_exc!r}"
    ) from _exc
_mamba_mode_values: tuple[Any, ...] = getattr(MambaCacheMode, "__args__", ())
_require(
    "all" in _mamba_mode_values and "none" in _mamba_mode_values,
    f"MambaCacheMode literals drifted; got {_mamba_mode_values!r}.",
)

# SchedulerConfig.max_num_seqs (refactor canary).
if _dataclasses.is_dataclass(SchedulerConfig):
    _sched_fields: set[str] = {f.name for f in _dataclasses.fields(SchedulerConfig)}
elif hasattr(SchedulerConfig, "model_fields"):
    _sched_fields = set(SchedulerConfig.model_fields)
else:
    _sched_fields = set()
_require("max_num_seqs" in _sched_fields, "SchedulerConfig.max_num_seqs not declared.")

# Landmark the two targeted functions.
_orig_get_max_concurrency = getattr(
    _kv_cache_utils_mod, "get_max_concurrency_for_kv_cache_config", None
)
_orig_report_kv_cache_config = getattr(
    _kv_cache_utils_mod, "_report_kv_cache_config", None
)
_require(
    callable(_orig_get_max_concurrency) and callable(_orig_report_kv_cache_config),
    "target functions missing or not callable.",
)
try:
    _gmc_sig = inspect.signature(_orig_get_max_concurrency)
    _gmc_src = inspect.getsource(_orig_get_max_concurrency)
    _rkc_sig = inspect.signature(_orig_report_kv_cache_config)
    _rkc_src = inspect.getsource(_orig_report_kv_cache_config)
except (TypeError, ValueError, OSError) as _exc:
    raise HybridKvPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect target functions: {_exc!r}"
    ) from _exc

_require(
    list(_gmc_sig.parameters) == ["vllm_config", "kv_cache_config"]
    and list(_rkc_sig.parameters) == ["vllm_config", "kv_cache_config"],
    "target signatures drifted.",
)
_require(
    _CONCURRENCY_BUGGY_LANDMARK in _gmc_src,
    "concurrency buggy landmark missing — PR #40384 likely landed; delete this patch.",
)
_require(
    _REPORT_BUGGY_LANDMARK in _rkc_src,
    "report buggy landmark missing — PR #40384 likely landed; delete this patch.",
)
# Coordinated: _report still calls get_max_concurrency.
_require(
    "get_max_concurrency_for_kv_cache_config" in _rkc_src,
    "_report_kv_cache_config no longer calls get_max_concurrency; coordination broken.",
)


# Phase 7: Helper — token-capacity classifier.
def _group_contributes_to_token_capacity(
    group: KVCacheGroupSpec, vllm_config: VllmConfig
) -> bool:
    """True iff this group's KV state grows with context length.

    AttentionSpec → always; MambaSpec → only when
    ``cache_config.mamba_cache_mode == "all"`` (per
    ``MambaSpec.max_memory_usage_bytes`` branch at lines 403-405);
    otherwise Mamba is O(1) per request and excluded. Unknown spec
    subclass → conservative True (under-report the boot-log capacity
    rather than over-report it).
    """
    spec = group.kv_cache_spec
    if isinstance(spec, AttentionSpec):
        return True
    if isinstance(spec, MambaSpec):
        return (
            getattr(
                getattr(vllm_config, "cache_config", None),
                "mamba_cache_mode",
                "none",
            )
            == "all"
        )
    return True


def _validate_kv_cache_config_shape(kv_cache_config: object, fn: str) -> KVCacheConfig:
    """Per-call defensive shape guard. In the pinned commit both targets
    are reached only from the boot-time log path, so this validation is
    future-proofing — refuse on a malformed KVCacheConfig rather than
    return a plausible-but-wrong float if a future upstream change ever
    routes either function into a non-log consumer.
    """
    if not isinstance(kv_cache_config, KVCacheConfig):
        raise TypeError(
            f"[{_PATCH_TAG}] {fn}: expected KVCacheConfig, got "
            f"{type(kv_cache_config).__name__!r}"
        )
    if not isinstance(kv_cache_config.kv_cache_groups, list):
        raise TypeError(f"[{_PATCH_TAG}] {fn}: kv_cache_groups not a list")
    if not isinstance(kv_cache_config.num_blocks, int) or kv_cache_config.num_blocks < 0:
        raise TypeError(
            f"[{_PATCH_TAG}] {fn}: num_blocks invalid ({kv_cache_config.num_blocks!r})"
        )
    for i, group in enumerate(kv_cache_config.kv_cache_groups):
        if not isinstance(group, KVCacheGroupSpec):
            raise TypeError(f"[{_PATCH_TAG}] {fn}: kv_cache_groups[{i}] not a group")
        if not isinstance(group.kv_cache_spec, KVCacheSpec):
            raise TypeError(f"[{_PATCH_TAG}] {fn}: groups[{i}].kv_cache_spec invalid")
        if not isinstance(group.layer_names, list):
            raise TypeError(f"[{_PATCH_TAG}] {fn}: groups[{i}].layer_names not a list")
    return kv_cache_config


def _capacity_groups(
    kv_cache_config: KVCacheConfig, vllm_config: VllmConfig
) -> list[KVCacheGroupSpec]:
    """Filter to token-capacity-contributing groups; pathological-empty
    falls back to all groups (matches PR #40384)."""
    groups = [
        g
        for g in kv_cache_config.kv_cache_groups
        if _group_contributes_to_token_capacity(g, vllm_config)
    ]
    return groups or list(kv_cache_config.kv_cache_groups)


def _get_max_concurrency_patched(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """Count only token-capacity-contributing groups for per-request
    memory and the pool-capacity divisor. O(1) Mamba groups excluded."""
    _validate_kv_cache_config_shape(
        kv_cache_config, "get_max_concurrency_for_kv_cache_config"
    )
    capacity_groups = _capacity_groups(kv_cache_config, vllm_config)
    if not capacity_groups:
        return 0.0
    num_layer_per_group = max(len(g.layer_names) for g in capacity_groups)
    if num_layer_per_group == 0:
        return 0.0
    max_memory_usage_per_request = num_layer_per_group * sum(
        g.kv_cache_spec.max_memory_usage_bytes(vllm_config) for g in capacity_groups
    )
    memory_per_block = (
        capacity_groups[0].kv_cache_spec.page_size_bytes * num_layer_per_group
    )
    if memory_per_block <= 0:
        raise TypeError(
            f"[{_PATCH_TAG}] memory_per_block non-positive ({memory_per_block!r})"
        )
    from vllm.utils.math_utils import cdiv

    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    if num_block_per_request <= 0:
        return 0.0
    return float(kv_cache_config.num_blocks / num_block_per_request)


def _report_kv_cache_config_patched(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Divide ``num_blocks`` by the count of token-capacity-contributing
    groups, not by ``len(kv_cache_groups)``. No per-call shape validation:
    this function only logs and the original would have crashed too.
    """
    if not kv_cache_config.kv_cache_groups:
        return
    capacity_groups = _capacity_groups(kv_cache_config, vllm_config)
    min_block_size = min(
        g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
    )
    if min_block_size <= 0:
        raise TypeError(
            f"[{_PATCH_TAG}] min_block_size non-positive ({min_block_size!r})"
        )
    if not capacity_groups:
        return

    num_tokens = (kv_cache_config.num_blocks // len(capacity_groups)) * min_block_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size * dcp_size > 1:
        num_tokens *= pcp_size * dcp_size
        _kv_cache_utils_mod.logger.info(
            "Multiplying the GPU KV cache size by the cp_world_size %d "
            "(pcp_world_size %d * dcp_world_size %d).",
            pcp_size * dcp_size,
            pcp_size,
            dcp_size,
        )
    _kv_cache_utils_mod.logger.info_once(
        "GPU KV cache size: %s tokens", f"{num_tokens:,}", scope="local"
    )
    # Re-resolve from the module so a patch composed on top of ours
    # picks up the latest version rather than capturing a stale closure.
    max_concurrency = _kv_cache_utils_mod.get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    _kv_cache_utils_mod.logger.info_once(
        "Maximum concurrency for %s tokens per request: %.2fx",
        f"{vllm_config.model_config.max_model_len:,}",
        max_concurrency,
        scope="local",
    )


for _fn, _name, _orig in (
    (
        _get_max_concurrency_patched,
        "get_max_concurrency_for_kv_cache_config",
        _orig_get_max_concurrency,
    ),
    (_report_kv_cache_config_patched, "_report_kv_cache_config", _orig_report_kv_cache_config),
):
    _fn.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
    _fn.__wrapped_original__ = _orig  # type: ignore[attr-defined]
    _fn.__name__ = _name
    _fn.__qualname__ = _name
    _fn.__module__ = _kv_cache_utils_mod.__name__


# Install and verify both replacements via dynamic and static lookup.
_kv_cache_utils_mod.get_max_concurrency_for_kv_cache_config = _get_max_concurrency_patched
_kv_cache_utils_mod._report_kv_cache_config = _report_kv_cache_config_patched

for _attr in ("get_max_concurrency_for_kv_cache_config", "_report_kv_cache_config"):
    _require(
        getattr(getattr(_kv_cache_utils_mod, _attr), "__qwen36_patch__", None)
        == _PATCH_TAG,
        f"post-install: {_attr} tag absent via attribute access.",
    )
    _require(
        getattr(
            inspect.getattr_static(_kv_cache_utils_mod, _attr), "__qwen36_patch__", None
        )
        == _PATCH_TAG,
        f"post-install: {_attr} inspect.getattr_static disagrees.",
    )


_logger.info(
    "[%s] applied: O(1) Mamba groups excluded from per-token capacity "
    "(vLLM commit %s); 'GPU KV cache size' boot-line and 'Maximum "
    "concurrency' now honest. Byte-level over-reservation per #37121 / "
    "PR #37429 is NOT fixed by this patch.",
    _PATCH_TAG,
    _PINNED_VLLM_COMMIT,
)
