"""sitecustomize.py — strict, fail-loud auto-load of every patch in
EVERY Python interpreter that has /opt/patches on its sys.path.

Why this file exists
--------------------

vLLM v1 spawns its EngineCore as a separate ``multiprocessing`` child
process. On Linux with CUDA the start method is ``spawn`` (CUDA forbids
``fork`` after init, enforced upstream by
``vllm/utils/system_utils.py:_maybe_force_spawn``). ``spawn`` means
EngineCore is a fresh Python interpreter that inherits the parent's
environment but **not** the parent's ``sys.modules``.

The launcher (``launch_with_patches.py``) imports each patch in PID 1.
Patches targeting API-server-resident code become live there directly.
Patch ``monkey_patch_hybrid_kv_allocator`` targets two functions in
``vllm.v1.core.kv_cache_utils`` called by the EngineCore subprocess at
boot reporting time — never by the API server. Without re-installing
in EngineCore, that patch is silently dead: the launcher's PID-1
verifier reports success, but the boot-log "GPU KV cache size" line
emitted from EngineCore shows the unpatched value.

CPython's standard library ``site.py`` auto-imports a top-level
``sitecustomize`` from ``sys.path`` at every interpreter startup —
including spawned multiprocessing children — before any user code
runs. With ``PYTHONPATH=/opt/patches`` set in the docker run command,
this file is what ``site.py`` finds, so each patch's module-level
install code runs in EngineCore's interpreter too.

Idempotency: each patch is imported exactly once per Python process
via ``__import__``'s ``sys.modules`` cache. The ``builtins`` sentinel
below blocks the install loop from re-firing on a second sitecustomize
import. A forced re-install would still fail loud — patches refuse on
already-patched state.

Standards: no try/except suppression of patch install errors. A patch
that refuses to install means the interpreter does NOT come up.
"""

from __future__ import annotations

# Order MUST match launch_with_patches.py:_PATCH_MODULES (the launcher
# checks for drift at boot). Load-bearing ordering: reasoning_field_egress
# rebuilds the DeltaMessage Pydantic schema, so any patch that constructs
# DeltaMessage at request time should come after it. The detector wraps
# extract_reasoning; the rebuild does not affect that surface.
_PATCH_MODULES: tuple[str, ...] = (
    "monkey_patch_qwen3_coder",
    "monkey_patch_hybrid_kv_allocator",
    "monkey_patch_reasoning_field_egress",
    "monkey_patch_reasoning_field_ingest",
    "monkey_patch_tool_call_in_think_detector",
    "monkey_patch_default_sampling_params",
    "monkey_patch_repetition_detection_default",
    "monkey_patch_qwen3_coder_grammar",
    "monkey_patch_request_memory_snapshot",
    "monkey_patch_tool_role_media_preserve",
    "monkey_patch_mm_cache_validator_eviction",
    "monkey_patch_qwen3_coder_streaming_truncation",
)


def _install_all_patches() -> None:
    """Import each patch in launcher order. Refuses to come up on any
    patch's typed install exception (no try/except suppression)."""
    for module_name in _PATCH_MODULES:
        __import__(module_name)


# Detect repeated load (defensive — sitecustomize is normally loaded
# once per interpreter, but importing manually from user code or a test
# harness would double-fire). The `sys.modules` cache makes the second
# call a no-op for the patches; this sentinel marks the run for an
# external observer.
if not getattr(__import__("builtins"), "_qwen36_sitecustomize_loaded", False):
    setattr(__import__("builtins"), "_qwen36_sitecustomize_loaded", True)
    _install_all_patches()
