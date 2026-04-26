"""sitecustomize.py — strict, fail-loud auto-load of every patch in
EVERY Python interpreter that has /opt/patches on its sys.path.

Why this file exists
--------------------

vLLM v1 spawns its EngineCore as a separate ``multiprocessing`` child
process. On Linux with CUDA the start method is ``spawn`` (CUDA
forbids ``fork`` after initialisation, enforced upstream by
``vllm/utils/system_utils.py:_maybe_force_spawn``). ``spawn`` means the
EngineCore is a fresh Python interpreter that inherits the parent's
environment but *not* the parent's ``sys.modules``.

The launcher (``launch_with_patches.py``) imports each of the seven
monkey-patches in the API-server process, runs each per-patch verifier
in that process, and then hands off to ``vllm.entrypoints.cli.main``.
Patches that target API-server-resident code (the tool parser,
the reasoning parser, the chat-utils ingest, the Pydantic response
schema) become live in the API server because that is where the patch
ran. **Patch 2** (``monkey_patch_hybrid_kv_allocator``) targets two
functions in ``vllm.v1.core.kv_cache_utils`` that are called by the
EngineCore subprocess at boot reporting time — never by the API
server. Without re-installing patch 2 in EngineCore, its install in
PID 1 is silently dead: the launcher's verifier reports success
(checked in PID 1), the boot log line is wrong (emitted in EngineCore),
and operators see the unpatched ``GPU KV cache size: X tokens`` value.

Empirical proof of the regression: running the unpatched-against-
EngineCore configuration emits

    (EngineCore pid=NNN) INFO TIME [kv_cache_utils.py:1337]
        GPU KV cache size: 29,008 tokens

with the upstream filename annotation ``[kv_cache_utils.py:1337]``. The
patched function lives in ``monkey_patch_hybrid_kv_allocator.py:~744``;
when the patch is active the annotation switches to
``[monkey_patch_hybrid_kv_allocator.py:744]``. That filename
substitution is the cleanest pass/fail discriminator and is what
``tests/test_patches_in_enginecore.py`` (or equivalent) should assert.

How this file fixes the regression
----------------------------------

Python's ``site.py`` (CPython standard library) looks for a top-level
``sitecustomize`` module on ``sys.path`` at every interpreter startup
and imports it before any user code runs. With
``PYTHONPATH=/opt/patches`` set in the docker run command,
``/opt/patches`` is prepended to ``sys.path``, so this file is the
``sitecustomize`` module Python finds.

The flow in EngineCore subprocess startup:

1. ``multiprocessing.spawn`` creates a fresh ``python3`` process and
   passes ``PYTHONPATH=/opt/patches`` (env vars are inherited).
2. The fresh interpreter runs ``site.py``.
3. ``site.py`` finds ``/opt/patches/sitecustomize.py`` and imports
   it.
4. This file imports each of the seven patches in launcher order.
5. Each patch's module-level install code runs: it imports vLLM,
   validates landmarks, and installs the replacement. Any landmark
   failure raises a typed ``*PatchRefusedError`` — propagated, the
   subprocess startup aborts, and EngineCore fails to boot loudly.
6. EngineCore's main code (``VllmEngineCoreProc.run_engine_core``)
   then proceeds. Its imports of ``vllm.v1.core.kv_cache_utils``,
   ``Qwen3CoderToolParser``, ``Qwen3ReasoningParser``, etc. all hit
   the now-cached, patched modules in ``sys.modules`` of THIS
   interpreter.

The same flow applies to PID 1 (the launcher's own interpreter):
``site.py`` runs first, this file installs the patches, then
``launch.py`` is loaded and its ``importlib.import_module(...)`` calls
return cached modules. The launcher's verifiers still run and confirm
the install. There is no double-install: each patch's module-level
install code only fires once per Python process (the second
``importlib.import_module`` call hits ``sys.modules`` and is a no-op).

Idempotency contract
--------------------

* Each patch is imported exactly once per Python process. The second
  ``importlib.import_module(...)`` for the same name (whether by the
  launcher or by any other code path) returns the cached module
  without re-running its install code.
* If somehow the patches are re-imported (``importlib.reload(...)``,
  cleared ``sys.modules``, etc.), each patch's import-time landmark
  check would refuse on already-patched state — for example, the
  egress patch's Phase 4 audit refuses if
  ``model_config["serialize_by_alias"]`` is already ``True``. So
  even a forced re-install fails loudly rather than silently
  double-applying.

Standards
---------

* No try/except suppression of patch install errors. A patch that
  refuses to install means the interpreter does NOT come up.
* No fallback path. The patches are required, not optional.
* No logs in inner loops — this file runs ONCE per interpreter
  startup, not on any hot path.
"""

from __future__ import annotations

# Order MUST match launch_with_patches.py:_PATCH_MODULES (the
# launcher's docstring documents which orderings are load-bearing —
# specifically, ``reasoning_field_egress`` MUST come before
# ``tool_call_in_think_rescue``, since the rescue patch constructs
# DeltaMessage instances and the egress patch rebuilds DeltaMessage's
# Pydantic schema).
_PATCH_MODULES: tuple[str, ...] = (
    "monkey_patch_qwen3_coder",
    "monkey_patch_hybrid_kv_allocator",
    "monkey_patch_extract_tool_calls_metrics",
    "monkey_patch_extract_tool_calls_streaming_metrics",
    "monkey_patch_reasoning_field_egress",
    "monkey_patch_reasoning_field_ingest",
    "monkey_patch_tool_call_in_think_rescue",
)


def _install_all_patches() -> None:
    """Import each patch, in launcher order. Refuses to come up on any
    patch's typed install exception.

    This function is intentionally a closure — its locals do not
    leak into the importing interpreter's globals beyond what
    ``__import__`` itself caches in ``sys.modules``.
    """
    for module_name in _PATCH_MODULES:
        # ``__import__`` (vs. ``importlib.import_module``) is preferred
        # here because we are in sitecustomize, which runs before user
        # code: avoiding even importlib's overhead keeps interpreter
        # startup latency bounded. Either form is sufficient for
        # caching in sys.modules.
        __import__(module_name)


# Detect repeated load (defensive — sitecustomize is normally loaded
# exactly once per interpreter, but importing this module manually
# from user code or some test harness would double-fire). A second
# call is a no-op for the patches (sys.modules cache) but we still
# want to mark this module so an external observer can see it ran.
if not getattr(__import__("builtins"), "_qwen36_sitecustomize_loaded", False):
    setattr(__import__("builtins"), "_qwen36_sitecustomize_loaded", True)
    _install_all_patches()
