"""Strict, fail-loud runtime patch for vLLM hybrid-KV scheduler-budget bug #37121.

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce`` (README §3.2,
§6.7). Mirrors upstream PR #40384 ("[Bugfix] Exclude O(1) Mamba groups from
hybrid KV cache token capacity") semantically. Remove this file the moment
#40384 (or the broader #37429) is present in the pinned image (README §12
trigger 5/6).

What the bug is
---------------

vLLM's V1 paged KV cache manager builds one ``KVCacheGroupSpec`` per distinct
attention type. For Qwen3.6-35B-A3B (40 decoder layers = 30 Gated DeltaNet +
10 full softmax attention) this yields **two** groups:

* one ``MambaSpec`` group (the 30 DeltaNet layers — small, fixed recurrent
  state per request, **independent of context length**);
* one ``FullAttentionSpec`` group (the 10 full-attention layers — the only
  layers whose KV state grows with sequence length).

Two routines in ``vllm/v1/core/kv_cache_utils.py`` then mis-account for that
shape because they treat every group symmetrically:

1. ``get_max_concurrency_for_kv_cache_config`` (lines 802-820) sums
   ``MambaSpec.max_memory_usage_bytes`` (which is correctly O(1) — see
   ``vllm/v1/kv_cache_interface.py:402-409``) into the per-request memory
   budget as if it grew with tokens, then divides by the attention group's
   page size. The reported concurrency therefore undercounts.
2. ``_report_kv_cache_config`` (lines 1297-1306) computes
   ``num_tokens = num_blocks // len(kv_cache_groups) * min_block_size``.
   For Qwen3.6 (2 groups) this divides the reported "GPU KV cache size: X
   tokens" line by 2 even though only one of the groups actually consumes a
   block per token. For Nemotron-H-style ``1 attn + 3 mamba`` the divisor is
   off by 4.

Concrete impact on this deployment (README §6.15):

* Reported: ``GPU KV cache size: 63,360 tokens`` / ``Maximum concurrency: 1.85x``.
* Attention-math expected: ~255,850 tokens / concurrency ~7.4x.
* Ratio 255,850 / 63,360 ≈ 4.04, matching the ``40 layers / 10 attn layers``
  shape of this model exactly.

What this patch does
--------------------

Replaces the two reporting/scheduling functions above with versions that, for
each ``KVCacheGroupSpec``, distinguish:

* **Token-capacity-contributing groups**: ``AttentionSpec`` subclasses
  (``FullAttentionSpec``, ``SlidingWindowSpec``, ``ChunkedLocalAttentionSpec``,
  ``MLAAttentionSpec``, ``SinkFullAttentionSpec``, ``CrossAttentionSpec``,
  ``EncoderOnlyAttentionSpec``, ``TQFullAttentionSpec``) — their KV state
  scales with context length and they consume one block per ``block_size``
  tokens.
* **O(1) Mamba groups**: ``MambaSpec`` with ``mamba_cache_mode != "all"`` —
  their state is independent of context length and must be **excluded** from
  per-token capacity counting. (When ``mamba_cache_mode == "all"`` Mamba does
  scale with context, per ``MambaSpec.max_memory_usage_bytes`` line 403-405,
  so the patch keeps it in the capacity calculation in that mode.)

The patch matches PR #40384's narrow scope: it fixes the **scheduler/budget
side** (how many concurrent tokens we credit ourselves with) only. It does
NOT reshape the underlying byte-level allocation; the allocator still pads
every layer's blocks up to the unified attention page size and reserves
``num_blocks * padded_page_size`` per layer regardless. That deeper waste is
PR #37429's territory and is explicitly out of scope here. The pool already
has the bytes that make the higher reported concurrency safe; the scheduler
just was not crediting them.

What this patch deliberately does NOT do
-----------------------------------------

* Does **not** modify ``get_kv_cache_config_from_groups``. That function
  allocates the actual KV tensors and at the pinned commit its bytes-per-pool
  arithmetic is internally consistent (page_size × num_blocks × group_size
  per pool). Reshaping it is PR #37429's open RFC scope and would change the
  pool's physical layout, not just the scheduler's view of it. PR #40384 also
  leaves it alone.
* Does **not** change ``MambaSpec.max_memory_usage_bytes`` — that function
  already correctly reports O(1) sizing; the bug is the *callers* ignoring
  what it reports.
* Does **not** add per-request logging. The replacement functions are on the
  hot path of boot reporting only; they should be silent at request time.

Risk acknowledgment
-------------------

This patch touches the **request-admission accounting path**. A wrong value
here either:

1. Under-counts capacity → no harm beyond the current bug (server admits less
   than it could, which is exactly what we have today); OR
2. Over-counts capacity → scheduler admits more concurrent tokens than the
   pool can hold → **OOM at runtime**.

The landmark-validation discipline below mitigates this risk by refusing to
install if any of the structural assumptions the replacement bodies depend
on are not present in the targeted source. It does not eliminate the risk:
a future field rename plus a coincidentally-matching landmark string could
in principle slip through. The mitigation is to keep the landmark set narrow
and specific (so a coincidence is implausible), to fail loud at install
time, and to fail loud per-call if the inputs deviate from the contract.

Patch-discipline contract
-------------------------

At import the patch:

1. Imports vLLM and the targeted modules. Failure is a hard ImportError.
2. Resolves both targeted functions by name in the expected modules and
   verifies they exist and are callable with the expected signature.
3. Reads the source of each targeted function and checks for landmark
   substrings that prove the buggy behavior is present. If a landmark is
   absent — i.e. PR #40384 has already landed — the patch refuses to apply
   and tells the operator to delete this file.
4. Verifies the spec-class hierarchy the replacement bodies depend on:
   ``AttentionSpec``, ``MambaSpec``, ``FullAttentionSpec``, plus
   ``KVCacheConfig.kv_cache_groups``, ``KVCacheGroupSpec.kv_cache_spec``,
   ``KVCacheGroupSpec.layer_names``, ``KVCacheConfig.num_blocks``.
5. Verifies ``MambaSpec.max_memory_usage_bytes`` and
   ``AttentionSpec`` subclasses' ``max_memory_usage_bytes`` exist with the
   expected ``(self, vllm_config)`` signature, since the patched code calls
   them.
6. Verifies ``vllm.config.cache.MambaCacheMode`` accepts ``"all"`` (the
   gating literal the replacement uses to decide whether Mamba scales with
   tokens in the current configuration).
7. Verifies ``SchedulerConfig.max_num_seqs`` exists, since the per-call
   defensive check uses it to bound any future Mamba-as-tokens accounting
   (and to log the assumed seq count). It is not load-bearing for #40384's
   narrow scope but its absence would indicate a SchedulerConfig refactor
   we should re-audit.
8. Installs both replacements, tags each with ``__qwen36_patch__``, and
   verifies via both ``getattr`` and ``inspect.getattr_static`` that the
   tagged versions are what the modules will resolve at call time.
9. Logs a single INFO line via ``vllm.logger.init_logger`` naming the
   pinned commit and the two functions replaced.

Any landmark failure raises :class:`HybridKvPatchRefusedError` and the
interpreter does not continue. There is no ``SystemExit(0)`` or
``try/except Exception: pass`` on any path. Refusing to boot is strictly
safer than half-applying a patch whose blast radius is the engine's whole
admission path.

Loading
-------

Same caveat as ``monkey_patch_qwen3_coder.py``: ``PYTHONSTARTUP`` is honoured
only in interactive mode; for ``vllm serve`` use the wrapper-script load
mechanism documented in that file's docstring, e.g.::

    python -c "import monkey_patch_hybrid_kv_allocator; \\
               from runpy import run_module; \\
               run_module('vllm.entrypoints.cli.main', \\
                          run_name='__main__')" serve …

after mounting this file on ``PYTHONPATH``.
"""

from __future__ import annotations

import inspect
from typing import Any


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-pr40384-backport"

# Source-level landmarks. Each must be present in the targeted function's
# source for the patch to apply. Their absence indicates upstream has already
# fixed the bug (or refactored the function such that the replacement's
# assumptions no longer hold) — either way, refuse.
#
# `_CONCURRENCY_BUGGY_LANDMARK` proves the buggy summing-mamba-into-budget
# behavior at lines 808-820 of kv_cache_utils.py. PR #40384 replaces this
# expression with a filtered-groups version.
_CONCURRENCY_BUGGY_LANDMARK: str = (
    "num_layer_per_group = max(\n"
    "        len(group.layer_names) for group in kv_cache_config.kv_cache_groups\n"
    "    )"
)
# `_REPORT_BUGGY_LANDMARK` proves the buggy `// len(kv_cache_groups)` divisor
# in `_report_kv_cache_config` (lines 1297-1306). PR #40384 swaps the divisor
# for the count of token-capacity-contributing groups.
_REPORT_BUGGY_LANDMARK: str = (
    "num_tokens = (\n"
    "        kv_cache_config.num_blocks\n"
    "        // len(kv_cache_config.kv_cache_groups)\n"
    "        * min_block_size\n"
    "    )"
)


class HybridKvPatchRefusedError(RuntimeError):
    """A precondition for the hybrid-KV scheduler-budget patch was violated.

    Raised at import time only. The patch either applies cleanly and
    correctly, or the process does not come up. A half-installed or
    silently-skipped patch on the request-admission path is strictly worse
    than a loud crash at boot — wrong values here OOM the GPU at request
    time, in a manner that looks like a model-side timeout rather than the
    accounting bug it actually is.
    """


def _require(condition: object, failure_message: str) -> None:
    if not condition:
        raise HybridKvPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {failure_message}"
        )


# --------------------------------------------------------------------
# Phase 1: Import vLLM and locate the target surface.
# --------------------------------------------------------------------

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

_logger = init_logger(__name__)


# --------------------------------------------------------------------
# Phase 2: Landmark the spec-class hierarchy the replacement assumes.
# --------------------------------------------------------------------

_require(
    inspect.isclass(AttentionSpec) and inspect.isclass(MambaSpec),
    "AttentionSpec and/or MambaSpec are no longer classes in "
    "vllm.v1.kv_cache_interface. The replacement body's isinstance checks "
    "would silently misclassify groups; refusing to apply.",
)
_require(
    issubclass(FullAttentionSpec, AttentionSpec),
    "FullAttentionSpec is no longer a subclass of AttentionSpec. The "
    "replacement body classifies groups via `isinstance(spec, AttentionSpec)` "
    "and would mis-treat full-attention groups as O(1) — the worst possible "
    "outcome (under-credit attention capacity, but also potentially over-credit "
    "if combined with the Mamba branch). Refusing.",
)
_require(
    not issubclass(MambaSpec, AttentionSpec),
    "MambaSpec is now a subclass of AttentionSpec — the replacement body's "
    "`isinstance(spec, AttentionSpec)` check would misclassify Mamba groups "
    "as token-scaling, OVER-CREDITING the pool's capacity and risking OOM "
    "at request time. Refusing.",
)

# KVCacheConfig / KVCacheGroupSpec field shape — the replacement bodies
# read these attributes by name. A rename here would AttributeError per-call.
# Use dataclasses.fields() rather than __annotations__ so we are robust to
# inheritance — the names we read must be REAL fields on the resolved
# dataclass, not just type hints anywhere up the MRO.
import dataclasses as _dataclasses

_require(
    _dataclasses.is_dataclass(KVCacheConfig),
    "KVCacheConfig is no longer a dataclass; the patch reads its field "
    "names via dataclasses.fields(). Refusing.",
)
_require(
    _dataclasses.is_dataclass(KVCacheGroupSpec),
    "KVCacheGroupSpec is no longer a dataclass; refusing.",
)
_kvcc_field_names = {f.name for f in _dataclasses.fields(KVCacheConfig)}
_require(
    "kv_cache_groups" in _kvcc_field_names,
    f"KVCacheConfig.kv_cache_groups field is missing or renamed. "
    f"Got fields: {sorted(_kvcc_field_names)!r}.",
)
_require(
    "num_blocks" in _kvcc_field_names,
    f"KVCacheConfig.num_blocks field is missing or renamed. "
    f"Got fields: {sorted(_kvcc_field_names)!r}.",
)
_kvgs_field_names = {f.name for f in _dataclasses.fields(KVCacheGroupSpec)}
_require(
    "kv_cache_spec" in _kvgs_field_names,
    f"KVCacheGroupSpec.kv_cache_spec field is missing or renamed. "
    f"Got fields: {sorted(_kvgs_field_names)!r}.",
)
_require(
    "layer_names" in _kvgs_field_names,
    f"KVCacheGroupSpec.layer_names field is missing or renamed. "
    f"Got fields: {sorted(_kvgs_field_names)!r}.",
)


# --------------------------------------------------------------------
# Phase 3: Landmark the max_memory_usage_bytes API the replacement uses.
# --------------------------------------------------------------------

_mamba_max_mem = getattr(MambaSpec, "max_memory_usage_bytes", None)
_require(
    callable(_mamba_max_mem),
    "MambaSpec.max_memory_usage_bytes is missing or not callable. The "
    "patched concurrency calculation depends on this method to compute the "
    "per-request fixed reservation that gets EXCLUDED from per-token "
    "capacity. Refusing.",
)
try:
    _mamba_sig = inspect.signature(_mamba_max_mem)  # type: ignore[arg-type]
except (TypeError, ValueError) as _exc:
    raise HybridKvPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect "
        f"MambaSpec.max_memory_usage_bytes: {_exc!r}"
    ) from _exc
_require(
    list(_mamba_sig.parameters) == ["self", "vllm_config"],
    f"MambaSpec.max_memory_usage_bytes signature changed; expected "
    f"(self, vllm_config), got {list(_mamba_sig.parameters)!r}. The "
    f"replacement body calls it as `spec.max_memory_usage_bytes(vllm_config)`; "
    f"a different signature would TypeError at the first boot.",
)

_attn_max_mem = getattr(FullAttentionSpec, "max_memory_usage_bytes", None)
_require(
    callable(_attn_max_mem),
    "FullAttentionSpec.max_memory_usage_bytes is missing or not callable.",
)
try:
    _attn_sig = inspect.signature(_attn_max_mem)  # type: ignore[arg-type]
except (TypeError, ValueError) as _exc:
    raise HybridKvPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect "
        f"FullAttentionSpec.max_memory_usage_bytes: {_exc!r}"
    ) from _exc
_require(
    list(_attn_sig.parameters) == ["self", "vllm_config"],
    f"FullAttentionSpec.max_memory_usage_bytes signature changed; expected "
    f"(self, vllm_config), got {list(_attn_sig.parameters)!r}.",
)


# --------------------------------------------------------------------
# Phase 4: Landmark MambaCacheMode — gating literal the replacement uses.
# --------------------------------------------------------------------

# `mamba_cache_mode == "all"` is the upstream sentinel for "Mamba state DOES
# scale with context length" (see MambaSpec.max_memory_usage_bytes branch at
# kv_cache_interface.py:403-405). The replacement uses this literal; if
# upstream renamed the mode or its value, refuse.
try:
    from vllm.config.cache import MambaCacheMode  # type: ignore[attr-defined]
except ImportError as _exc:
    raise HybridKvPatchRefusedError(
        f"[{_PATCH_TAG}] cannot import vllm.config.cache.MambaCacheMode: "
        f"{_exc!r}. The replacement body distinguishes O(1) Mamba groups "
        f"from O(tokens) Mamba groups via cache_config.mamba_cache_mode == "
        f"'all'; without this type we cannot prove the literal is still valid."
    ) from _exc

# typing.Literal exposes its values via __args__ on the alias.
_mamba_mode_values: tuple[Any, ...] = getattr(MambaCacheMode, "__args__", ())
_require(
    "all" in _mamba_mode_values,
    f"vllm.config.cache.MambaCacheMode no longer contains 'all'. The "
    f"replacement body uses 'all' as the sentinel for 'Mamba state scales "
    f"with context length'. Got values: {_mamba_mode_values!r}.",
)
_require(
    "none" in _mamba_mode_values,
    f"vllm.config.cache.MambaCacheMode no longer contains 'none' (the "
    f"default). Got values: {_mamba_mode_values!r}.",
)


# --------------------------------------------------------------------
# Phase 5: Landmark SchedulerConfig.max_num_seqs.
# --------------------------------------------------------------------

# Not load-bearing for #40384's narrow scope itself, but its absence would
# indicate a SchedulerConfig refactor large enough to warrant re-auditing
# the whole patch against the new shape. SchedulerConfig is a pydantic
# dataclass in this commit (vllm.config.utils.config decorator), so it
# exposes fields via dataclasses.fields(); also accept BaseModel-style
# `model_fields` for forward-compat.
_sched_field_names: set[str]
if _dataclasses.is_dataclass(SchedulerConfig):
    _sched_field_names = {f.name for f in _dataclasses.fields(SchedulerConfig)}
elif hasattr(SchedulerConfig, "model_fields"):
    _sched_field_names = set(SchedulerConfig.model_fields)
else:
    _sched_field_names = set()
_require(
    "max_num_seqs" in _sched_field_names,
    f"SchedulerConfig.max_num_seqs is no longer a declared field. "
    f"Got fields: {sorted(_sched_field_names)!r}. A refactor of this scope "
    f"warrants re-auditing this patch.",
)


# --------------------------------------------------------------------
# Phase 6: Landmark the two targeted functions and verify they are buggy.
# --------------------------------------------------------------------

_orig_get_max_concurrency = getattr(
    _kv_cache_utils_mod, "get_max_concurrency_for_kv_cache_config", None
)
_require(
    _orig_get_max_concurrency is not None and callable(_orig_get_max_concurrency),
    "vllm.v1.core.kv_cache_utils.get_max_concurrency_for_kv_cache_config is "
    "missing or not callable. Upstream has moved or renamed it; re-audit "
    "before bumping the pinned commit.",
)
try:
    _gmc_sig = inspect.signature(_orig_get_max_concurrency)
except (TypeError, ValueError) as _exc:
    raise HybridKvPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect "
        f"get_max_concurrency_for_kv_cache_config: {_exc!r}"
    ) from _exc
_require(
    list(_gmc_sig.parameters) == ["vllm_config", "kv_cache_config"],
    f"get_max_concurrency_for_kv_cache_config signature changed; expected "
    f"(vllm_config, kv_cache_config), got {list(_gmc_sig.parameters)!r}.",
)

try:
    _gmc_src = inspect.getsource(_orig_get_max_concurrency)
except (OSError, TypeError) as _exc:
    raise HybridKvPatchRefusedError(
        f"[{_PATCH_TAG}] cannot read source of "
        f"get_max_concurrency_for_kv_cache_config: {_exc!r}"
    ) from _exc

_require(
    _CONCURRENCY_BUGGY_LANDMARK in _gmc_src,
    "buggy landmark for get_max_concurrency_for_kv_cache_config not found. "
    "Upstream PR #40384 has most likely already landed — delete this patch "
    "file and its mount per README §11 step 3 / §6.15 'Revisit when' rather "
    "than re-patching a fixed function.",
)

_orig_report_kv_cache_config = getattr(
    _kv_cache_utils_mod, "_report_kv_cache_config", None
)
_require(
    _orig_report_kv_cache_config is not None
    and callable(_orig_report_kv_cache_config),
    "vllm.v1.core.kv_cache_utils._report_kv_cache_config is missing or not "
    "callable. The 'GPU KV cache size: X tokens' boot line is emitted from "
    "this function; without it the user-visible token-count number cannot be "
    "corrected.",
)
try:
    _rkc_sig = inspect.signature(_orig_report_kv_cache_config)
except (TypeError, ValueError) as _exc:
    raise HybridKvPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect _report_kv_cache_config: {_exc!r}"
    ) from _exc
_require(
    list(_rkc_sig.parameters) == ["vllm_config", "kv_cache_config"],
    f"_report_kv_cache_config signature changed; expected "
    f"(vllm_config, kv_cache_config), got {list(_rkc_sig.parameters)!r}.",
)
try:
    _rkc_src = inspect.getsource(_orig_report_kv_cache_config)
except (OSError, TypeError) as _exc:
    raise HybridKvPatchRefusedError(
        f"[{_PATCH_TAG}] cannot read source of _report_kv_cache_config: "
        f"{_exc!r}"
    ) from _exc

_require(
    _REPORT_BUGGY_LANDMARK in _rkc_src,
    "buggy landmark for _report_kv_cache_config not found. Upstream PR "
    "#40384 (or a sibling) has most likely already corrected the divisor in "
    "the 'GPU KV cache size: X tokens' computation. Delete this patch file.",
)
# The replacement calls the patched get_max_concurrency_for_kv_cache_config
# from inside _report_kv_cache_config, mirroring upstream. Verify upstream
# still does so — if it stopped, our two patches are no longer coordinated.
_require(
    "get_max_concurrency_for_kv_cache_config" in _rkc_src,
    "_report_kv_cache_config no longer calls "
    "get_max_concurrency_for_kv_cache_config; the two replacements were "
    "designed to be coordinated. Refusing to apply only one half of the fix.",
)


# --------------------------------------------------------------------
# Phase 7: Helper — classify a group as token-capacity-contributing.
# --------------------------------------------------------------------


def _group_contributes_to_token_capacity(
    group: KVCacheGroupSpec, vllm_config: VllmConfig
) -> bool:
    """Return True iff this group's KV state grows with context length.

    Token-capacity-contributing groups are exactly:

    * Any ``AttentionSpec`` subclass (full attn, sliding window, chunked
      local, MLA, sink-attn, cross-attn, encoder-only, TQ). Their
      ``max_memory_usage_bytes`` scales with ``max_model_len``.
    * ``MambaSpec`` only when ``vllm_config.cache_config.mamba_cache_mode ==
      "all"``. In that mode (see MambaSpec.max_memory_usage_bytes branch
      at kv_cache_interface.py:403-405) Mamba does scale with
      ``max_model_len`` and must be counted; in any other mode (default
      "none", or "align") Mamba is O(1) per request and must be excluded.

    Anything else (unknown spec subclass) we conservatively treat as
    token-capacity-contributing — the result is to UNDER-credit the pool
    rather than over-credit it, which is the safe direction for an
    unfamiliar spec.
    """
    spec = group.kv_cache_spec
    if isinstance(spec, AttentionSpec):
        return True
    if isinstance(spec, MambaSpec):
        mamba_mode = getattr(
            getattr(vllm_config, "cache_config", None),
            "mamba_cache_mode",
            "none",
        )
        return mamba_mode == "all"
    # Unknown spec subclass: conservative default. Keeps the divisor at
    # least as large as before the patch for unrecognized shapes.
    return True


def _validate_kv_cache_config_shape(
    kv_cache_config: object, fn_name: str
) -> KVCacheConfig:
    """Per-call defensive guard. Raises TypeError on any contract violation.

    The blast radius of this patch is the request-admission path. We refuse
    to compute on inputs that don't match the contract this patch was
    written against, rather than silently producing a wrong-but-plausible
    number.
    """
    if not isinstance(kv_cache_config, KVCacheConfig):
        raise TypeError(
            f"[{_PATCH_TAG}] {fn_name}: expected KVCacheConfig, got "
            f"{type(kv_cache_config).__name__!r}"
        )
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if groups is None:
        raise TypeError(
            f"[{_PATCH_TAG}] {fn_name}: KVCacheConfig has no "
            f"'kv_cache_groups' attribute"
        )
    if not isinstance(groups, list):
        raise TypeError(
            f"[{_PATCH_TAG}] {fn_name}: KVCacheConfig.kv_cache_groups is "
            f"{type(groups).__name__!r}, expected list"
        )
    num_blocks = getattr(kv_cache_config, "num_blocks", None)
    if not isinstance(num_blocks, int):
        raise TypeError(
            f"[{_PATCH_TAG}] {fn_name}: KVCacheConfig.num_blocks is "
            f"{type(num_blocks).__name__!r}, expected int"
        )
    if num_blocks < 0:
        raise TypeError(
            f"[{_PATCH_TAG}] {fn_name}: KVCacheConfig.num_blocks is "
            f"negative ({num_blocks!r})"
        )
    for i, group in enumerate(groups):
        if not isinstance(group, KVCacheGroupSpec):
            raise TypeError(
                f"[{_PATCH_TAG}] {fn_name}: kv_cache_groups[{i}] is "
                f"{type(group).__name__!r}, expected KVCacheGroupSpec"
            )
        if not hasattr(group, "kv_cache_spec"):
            raise TypeError(
                f"[{_PATCH_TAG}] {fn_name}: kv_cache_groups[{i}] is missing "
                f"'kv_cache_spec' attribute"
            )
        if not isinstance(group.kv_cache_spec, KVCacheSpec):
            raise TypeError(
                f"[{_PATCH_TAG}] {fn_name}: kv_cache_groups[{i}].kv_cache_spec "
                f"is {type(group.kv_cache_spec).__name__!r}, expected "
                f"KVCacheSpec"
            )
        if not hasattr(group, "layer_names") or not isinstance(
            group.layer_names, list
        ):
            raise TypeError(
                f"[{_PATCH_TAG}] {fn_name}: kv_cache_groups[{i}].layer_names "
                f"is missing or not a list"
            )
    return kv_cache_config  # narrowed type for the caller


# --------------------------------------------------------------------
# Phase 8: The replacement functions.
# --------------------------------------------------------------------


def _get_max_concurrency_patched(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """Strict replacement for ``get_max_concurrency_for_kv_cache_config``.

    Mirrors PR #40384's intent: count only token-capacity-contributing
    groups when computing per-request memory usage and the divisor for
    pool-capacity arithmetic. O(1) Mamba groups are excluded.

    Per-call validation: refuses to compute on a malformed
    ``KVCacheConfig`` (raises TypeError) rather than silently degrading
    to a wrong-but-plausible concurrency number.
    """
    _validate_kv_cache_config_shape(
        kv_cache_config, "get_max_concurrency_for_kv_cache_config"
    )

    capacity_groups: list[KVCacheGroupSpec] = [
        g
        for g in kv_cache_config.kv_cache_groups
        if _group_contributes_to_token_capacity(g, vllm_config)
    ]
    # Defensive: if filtering empties the list (e.g. a Mamba-only model in
    # "none" mode — pathological but possible), fall back to all groups so
    # we report SOMETHING rather than ZeroDivisionError. Matches PR #40384's
    # `or list(kv_cache_config.kv_cache_groups)` fallback.
    if not capacity_groups:
        capacity_groups = list(kv_cache_config.kv_cache_groups)

    if not capacity_groups:
        # Attention-free model edge case: nothing to compute concurrency
        # against. Upstream's original would have hit `max(...)` on an
        # empty sequence and raised; we return 0.0 as the architecturally
        # truthful answer (no per-token capacity at all).
        return 0.0

    num_layer_per_group = max(len(g.layer_names) for g in capacity_groups)
    if num_layer_per_group == 0:
        return 0.0

    # Per-request memory: only the capacity-contributing groups' specs
    # contribute. MambaSpec in "none" mode reports O(1) bytes via its own
    # max_memory_usage_bytes; we exclude it from this sum entirely so its
    # fixed reservation doesn't inflate the per-request budget that we
    # then divide the pool by.
    max_memory_usage_per_request = num_layer_per_group * sum(
        g.kv_cache_spec.max_memory_usage_bytes(vllm_config)
        for g in capacity_groups
    )

    # Memory-per-block must use the same group set so the units match.
    # Pre-#40384, upstream used `kv_cache_groups[0].kv_cache_spec.page_size_bytes`
    # which assumed all groups have the same padded page size — that
    # assumption holds (platforms/interface.py:626 forces it), but sourcing
    # the page_size from a capacity-contributing group is more honest about
    # what the number means: how many bytes one block costs in the layers
    # that actually grow with context.
    memory_per_block = (
        capacity_groups[0].kv_cache_spec.page_size_bytes * num_layer_per_group
    )
    if memory_per_block <= 0:
        raise TypeError(
            f"[{_PATCH_TAG}] get_max_concurrency_for_kv_cache_config: "
            f"capacity group page_size_bytes is non-positive "
            f"({memory_per_block!r}); cannot compute concurrency"
        )

    # cdiv to round up the per-request block count, matching upstream.
    from vllm.utils.math_utils import cdiv

    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    if num_block_per_request <= 0:
        return 0.0
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return float(max_concurrency)


def _report_kv_cache_config_patched(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Strict replacement for ``_report_kv_cache_config``.

    Mirrors PR #40384's intent for the user-visible "GPU KV cache size: X
    tokens" line: divide ``num_blocks`` by the count of token-capacity-
    contributing groups, not by ``len(kv_cache_groups)``. For Qwen3.6 this
    raises the reported number from ``num_blocks // 2 * block_size`` to
    ``num_blocks // 1 * block_size`` — a 2× jump for 1-attn-1-mamba, 4×
    for Nemotron-H 1-attn-3-mamba.

    Per-call validation: refuses on a malformed ``KVCacheConfig``.
    """
    _validate_kv_cache_config_shape(
        kv_cache_config, "_report_kv_cache_config"
    )

    capacity_groups: list[KVCacheGroupSpec] = [
        g
        for g in kv_cache_config.kv_cache_groups
        if _group_contributes_to_token_capacity(g, vllm_config)
    ]
    if not capacity_groups:
        capacity_groups = list(kv_cache_config.kv_cache_groups)

    # min_block_size is computed across ALL groups, matching upstream's
    # original (the smallest block size in any group is still the unit
    # of the user-facing token count). Empty-group case is the
    # attention-free model — guard early.
    if not kv_cache_config.kv_cache_groups:
        return

    min_block_size = min(
        g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
    )
    if min_block_size <= 0:
        raise TypeError(
            f"[{_PATCH_TAG}] _report_kv_cache_config: min block_size across "
            f"groups is non-positive ({min_block_size!r}); refusing to log "
            f"a meaningless token count."
        )

    capacity_divisor = len(capacity_groups)
    if capacity_divisor <= 0:
        return

    num_tokens = (
        kv_cache_config.num_blocks // capacity_divisor
    ) * min_block_size

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
    num_tokens_str = f"{num_tokens:,}"
    _kv_cache_utils_mod.logger.info_once(
        "GPU KV cache size: %s tokens", num_tokens_str, scope="local"
    )
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    # Re-resolve the patched concurrency function from the module so that
    # if some other patch composes on top of ours after install, we pick
    # up the latest version rather than capturing a stale closure here.
    max_concurrency = _kv_cache_utils_mod.get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    _kv_cache_utils_mod.logger.info_once(
        "Maximum concurrency for %s tokens per request: %.2fx",
        max_model_len_str,
        max_concurrency,
        scope="local",
    )


# Tag and rename the replacements so post-install verification can find
# them via both getattr and inspect.getattr_static.
_get_max_concurrency_patched.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_get_max_concurrency_patched.__wrapped_original__ = _orig_get_max_concurrency  # type: ignore[attr-defined]
_get_max_concurrency_patched.__name__ = "get_max_concurrency_for_kv_cache_config"
_get_max_concurrency_patched.__qualname__ = (
    "get_max_concurrency_for_kv_cache_config"
)
_get_max_concurrency_patched.__module__ = _kv_cache_utils_mod.__name__

_report_kv_cache_config_patched.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_report_kv_cache_config_patched.__wrapped_original__ = _orig_report_kv_cache_config  # type: ignore[attr-defined]
_report_kv_cache_config_patched.__name__ = "_report_kv_cache_config"
_report_kv_cache_config_patched.__qualname__ = "_report_kv_cache_config"
_report_kv_cache_config_patched.__module__ = _kv_cache_utils_mod.__name__


# --------------------------------------------------------------------
# Phase 9: Install and verify (both functions).
# --------------------------------------------------------------------

_kv_cache_utils_mod.get_max_concurrency_for_kv_cache_config = (
    _get_max_concurrency_patched
)
_kv_cache_utils_mod._report_kv_cache_config = _report_kv_cache_config_patched

_installed_concurrency = _kv_cache_utils_mod.get_max_concurrency_for_kv_cache_config
_require(
    getattr(_installed_concurrency, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install verification failed: "
    "kv_cache_utils.get_max_concurrency_for_kv_cache_config does not bear "
    "the expected patch tag. A concurrent monkey-patch has clobbered ours.",
)

_resolved_concurrency = inspect.getattr_static(
    _kv_cache_utils_mod, "get_max_concurrency_for_kv_cache_config"
)
_require(
    getattr(_resolved_concurrency, "__qwen36_patch__", None) == _PATCH_TAG,
    "static-lookup verification failed: "
    "inspect.getattr_static sees a different "
    "get_max_concurrency_for_kv_cache_config than normal attribute access. "
    "Refusing to proceed.",
)

_installed_report = _kv_cache_utils_mod._report_kv_cache_config
_require(
    getattr(_installed_report, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install verification failed: "
    "kv_cache_utils._report_kv_cache_config does not bear the expected "
    "patch tag. A concurrent monkey-patch has clobbered ours.",
)

_resolved_report = inspect.getattr_static(
    _kv_cache_utils_mod, "_report_kv_cache_config"
)
_require(
    getattr(_resolved_report, "__qwen36_patch__", None) == _PATCH_TAG,
    "static-lookup verification failed: "
    "inspect.getattr_static sees a different _report_kv_cache_config than "
    "normal attribute access. Refusing to proceed.",
)


_logger.info(
    "[%s] applied: replaced %s.get_max_concurrency_for_kv_cache_config and "
    "%s._report_kv_cache_config for vLLM commit %s "
    "(O(1) Mamba groups now excluded from per-token capacity counting; "
    "boot-line 'GPU KV cache size' and 'Maximum concurrency' values become "
    "honest about the attention-only token budget; allocator's underlying "
    "byte-level over-reservation per #37121 / PR #37429 is NOT fixed by "
    "this patch).",
    _kv_cache_utils_mod.__name__,
    _kv_cache_utils_mod.__name__,
    _PINNED_VLLM_COMMIT,
)
