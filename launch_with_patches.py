#!/usr/bin/env python3
"""launch_with_patches.py — strict, fail-loud entrypoint that installs
server-side runtime patches BEFORE handing off to vLLM's CLI.

Why this file exists
--------------------

README §8.2 currently mounts ``monkey_patch_qwen3_coder.py`` into the
container and sets ``PYTHONSTARTUP=/opt/patches/monkey_patch.py`` to
load it before ``vllm serve`` imports the parser. **This does not
work.** CPython only honours ``PYTHONSTARTUP`` in *interactive* mode
(see ``Lib/site.py`` / ``Modules/main.c``: ``Py_InspectFlag`` / the
``-i`` switch); the container's ``ENTRYPOINT ["vllm", "serve"]`` is
the non-interactive console-script path, so ``PYTHONSTARTUP`` is
never read and the patch is never imported. The vLLM server boots
with the *unpatched* ``Qwen3CoderToolParser`` and the bug
``monkey_patch_qwen3_coder.py`` exists to eliminate is fully present
at request time. The docstring of ``monkey_patch_qwen3_coder.py``
("Loading" section near the bottom) names this gap explicitly and
sketches the wrapper-script approach this file implements.

What this file does
-------------------

Replaces the container entrypoint. Where the README §8.2 invocation
was effectively ``vllm serve <args>``, the corrected invocation is::

    python /opt/patches/launch.py serve <args>

The launcher:

1. Imports each module in :data:`_PATCH_MODULES` in declared order.
   Each patch is itself strict: at import time it validates landmarks
   and raises a typed exception (e.g.
   :class:`monkey_patch_qwen3_coder.MonkeyPatchRefusedError`) on any
   mismatch. The launcher does **not** catch those exceptions — they
   propagate and the container exits non-zero with the patch's
   specific refusal message intact.
2. After each import, runs a per-module verifier from
   :data:`_PATCH_VERIFICATION` that re-checks the patch's installed
   landmark from the launcher's vantage point. This is defense in
   depth above the patch's own internal post-install check: if a
   later patch in the registry, an import side-effect, or a vLLM
   subsystem somehow clobbers the surface between patch install and
   handoff, the verifier raises :class:`PatchVerificationError`
   before the server starts taking traffic.
3. Hands off to ``vllm.entrypoints.cli.main`` via
   :func:`runpy.run_module` with ``run_name="__main__"`` and
   ``alter_sys=True``, after munging :data:`sys.argv` so argv[0] is
   ``"vllm"`` (matching the entry-point script the
   ``vllm/vllm-openai`` image installs at ``/usr/local/bin/vllm``)
   and argv[1:] are the subcommand args this launcher received.

The launcher adds nothing to the request path. Once
``runpy.run_module`` fires, vLLM's argparse, dispatch, and serving
machinery are unmodified.

The new docker run command
--------------------------

The docker run in README §8.2 must become (only the bind-mount key,
the env line, and the entrypoint override change — every other flag
is identical to the existing command)::

    docker run --rm -d --name qwen36 --gpus all \\
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
      -p 8000:8000 \\
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
      -v "$PWD/monkey_patch_qwen3_coder.py:/opt/patches/monkey_patch_qwen3_coder.py:ro" \\
      -v "$PWD/launch_with_patches.py:/opt/patches/launch.py:ro" \\
      -e HF_HUB_ENABLE_HF_TRANSFER=1 \\
      -e VLLM_USE_V1=1 \\
      -e PYTHONPATH=/opt/patches \\
      --entrypoint python \\
      vllm/vllm-openai@sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c \\
      /opt/patches/launch.py serve \\
      --model RedHatAI/Qwen3.6-35B-A3B-NVFP4 \\
      --revision e850c696e6d75f965367e816c16bc7dacd955ffa \\
      ...  # remaining --served-model-name / --host / ... flags unchanged

Two structural changes from the old command:

* The bind-mount for ``monkey_patch_qwen3_coder.py`` keeps its
  filename (so ``import monkey_patch_qwen3_coder`` resolves under
  ``PYTHONPATH=/opt/patches``) instead of being renamed to
  ``monkey_patch.py``.
* ``PYTHONSTARTUP`` is dropped entirely. The ``PYTHONPATH``
  prepend and the ``--entrypoint python /opt/patches/launch.py``
  override replace it.

Adding a new server-side patch
------------------------------

Each new patch is a standalone module with the same discipline as
``monkey_patch_qwen3_coder``: strict landmark validation, hard
raise on mismatch, ``__qwen36_patch__`` tag stamped on every
target it modifies. To register it with the launcher:

1. Make the module importable from ``PYTHONPATH=/opt/patches``
   (bind-mount it alongside the existing patch).
2. Append its module name to :data:`_PATCH_MODULES`. Order matters
   — modules earlier in the tuple are imported first. If patch B
   patches a class that patch A also touches, B must come after A.
3. Append a verifier callable to :data:`_PATCH_VERIFICATION`. The
   verifier receives the imported module object and must raise
   :class:`PatchVerificationError` if the patch's intended target
   does not bear the expected ``__qwen36_patch__`` tag (or whatever
   the patch's post-install landmark is). This is independent of
   the patch's own internal post-install check; both must agree.

Patch-discipline contract
-------------------------

The launcher will not:

* Catch ``MonkeyPatchRefusedError`` or any patch's typed install
  exception. A failed patch install is a hard boot failure.
* Continue to handoff if any verifier reports failure. A
  half-installed patch returns the server to the silent-failure
  mode the patches exist to eliminate; that outcome is strictly
  worse than a loud crash at boot.
* Catch ``runpy``'s exceptions on handoff. If
  ``vllm.entrypoints.cli.main`` raises during argparse,
  subcommand dispatch, or server startup, the traceback is the
  user's diagnostic — wrapping it would only hide it.
* Validate vLLM subcommand syntax. ``vllm`` has its own
  :class:`FlexibleArgumentParser` and emits a perfectly clear
  usage message; reproducing that here would drift.
"""

from __future__ import annotations

import importlib
import inspect
import runpy
import sys
from types import ModuleType
from typing import Callable, Iterable

_LAUNCHER_TAG: str = "qwen36-agent-setup-launcher-v1"

# The argv[0] we synthesize for the runpy'd vLLM CLI. Matches the
# entry-point script installed at /usr/local/bin/vllm by the
# vllm/vllm-openai image (the one a user would type by hand). vLLM
# uses sys.argv[0] in help-message paths and a few log lines; making
# it identical to the canonical invocation keeps those unchanged.
_VLLM_ARGV0: str = "vllm"

# The CLI module runpy targets. This is the exact module a normal
# `vllm ...` invocation runs under the hood (see
# /usr/local/bin/vllm: `from vllm.entrypoints.cli.main import main`).
_VLLM_CLI_MODULE: str = "vllm.entrypoints.cli.main"


class LauncherError(RuntimeError):
    """A precondition for the qwen3.6 launcher was violated.

    Raised before handoff to vLLM. The launcher either installs and
    verifies every registered patch and then hands off cleanly, or
    the process does not come up. There is no partial-install path.
    """


class PatchVerificationError(LauncherError):
    """A registered patch did not leave its expected landmark in place.

    Raised when the launcher's per-patch verifier (see
    :data:`_PATCH_VERIFICATION`) cannot find the
    ``__qwen36_patch__`` tag on the target the patch is supposed to
    have replaced. Distinguished from
    :class:`monkey_patch_qwen3_coder.MonkeyPatchRefusedError` —
    that one is raised by the patch *during* install if a precondition
    fails; this one is raised by the launcher *after* install if the
    patch's effect cannot be observed from the launcher's vantage point
    (e.g. clobbered by a sibling patch or a vLLM import side-effect).
    """


# --------------------------------------------------------------------
# Per-patch post-install verifiers.
# --------------------------------------------------------------------
#
# Each verifier takes the just-imported patch module and raises
# PatchVerificationError if the patch's intended target does not bear
# the expected __qwen36_patch__ tag. Each verifier re-imports vLLM
# internals from scratch (rather than trusting the patch module's
# cached references) so what gets checked is what `vllm serve` will
# actually see when it dispatches a request.
#
# Common shape: every patch in this project (a) stamps __qwen36_patch__
# on every target attribute it modifies and (b) stamps
# __wrapped_original__ on the same attribute. The shared helper
# _verify_target_carries_tag below probes via both getattr and
# inspect.getattr_static (defending against metaclass-level
# __getattribute__ overrides that could let normal lookup see a
# different object than the one bound in __dict__) and confirms both
# tags are present.


def _expected_tag_from(patch_module: ModuleType) -> str:
    """Read ``_PATCH_TAG`` off the patch module. Raises
    :class:`PatchVerificationError` if absent or non-string."""
    expected_tag = getattr(patch_module, "_PATCH_TAG", None)
    if not isinstance(expected_tag, str) or not expected_tag:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] patch module {patch_module.__name__!r} "
            f"does not export a non-empty _PATCH_TAG; cannot verify "
            f"post-install state."
        )
    return expected_tag


def _verify_target_carries_tag(
    target_obj: object,
    attr_name: str,
    expected_tag: str,
    *,
    patch_module_name: str,
    target_description: str,
) -> None:
    """Verify a class or module attribute bears the patch tag.

    Probes via both ``getattr`` and ``inspect.getattr_static``; the
    two MUST agree, or a metaclass shim is shadowing the install.
    Also verifies ``__wrapped_original__`` is present (every patch in
    this project sets both — missing one would indicate the tag was
    forged or the patch's contract has drifted in a way the launcher
    has not been audited against).
    """
    installed_dynamic = getattr(target_obj, attr_name, None)
    if installed_dynamic is None:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {target_description}.{attr_name} is "
            f"missing after import of {patch_module_name!r}. The patch "
            f"silently failed to install and its own post-install check "
            f"did not catch it."
        )
    try:
        installed_static = inspect.getattr_static(target_obj, attr_name)
    except AttributeError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] inspect.getattr_static cannot resolve "
            f"{target_description}.{attr_name}: {exc!r}. The attribute "
            f"machinery has been altered between patch install and "
            f"verification."
        ) from exc

    dynamic_tag = getattr(installed_dynamic, "__qwen36_patch__", None)
    static_tag = getattr(installed_static, "__qwen36_patch__", None)
    if dynamic_tag != expected_tag:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] post-install verification failed for "
            f"{patch_module_name!r}: "
            f"getattr({target_description}, {attr_name!r}) carries "
            f"__qwen36_patch__={dynamic_tag!r}, expected "
            f"{expected_tag!r}. A sibling patch or a vLLM import "
            f"side-effect has clobbered the install between patch "
            f"import and launcher handoff."
        )
    if static_tag != expected_tag:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] post-install static-lookup verification "
            f"failed for {patch_module_name!r}: inspect.getattr_static "
            f"sees __qwen36_patch__={static_tag!r}, expected "
            f"{expected_tag!r}. Normal attribute lookup and static "
            f"lookup disagree — a metaclass-level shim is shadowing the "
            f"install."
        )

    wrapped_original = getattr(installed_dynamic, "__wrapped_original__", None)
    if wrapped_original is None:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] post-install verification failed for "
            f"{patch_module_name!r}: {target_description}.{attr_name} "
            f"bears the patch tag but lacks __wrapped_original__. The "
            f"patch's published contract requires both; missing one "
            f"indicates the tag was forged or the patch was rewritten "
            f"in a way the launcher has not been audited against."
        )


def _verify_qwen3_coder(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_qwen3_coder`` actually replaced the parser.

    Re-imports ``Qwen3CoderToolParser`` from the same path vLLM's
    request handler uses and confirms ``_parse_xml_function_call``
    carries the patch tag.
    """
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.tool_parsers.qwen3coder_tool_parser import (
            Qwen3CoderToolParser,
        )
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import "
            f"vllm.tool_parsers.qwen3coder_tool_parser.Qwen3CoderToolParser "
            f"for verification: {exc!r}"
        ) from exc
    _verify_target_carries_tag(
        Qwen3CoderToolParser,
        "_parse_xml_function_call",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="Qwen3CoderToolParser",
    )


def _verify_hybrid_kv_allocator(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_hybrid_kv_allocator`` replaced both targets.

    The patch replaces TWO functions in
    ``vllm.v1.core.kv_cache_utils``; both must bear the tag, or the
    fix is half-applied and the boot-line / scheduler views disagree.
    """
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.v1.core import kv_cache_utils as _kv_cache_utils_mod
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import vllm.v1.core.kv_cache_utils "
            f"for verification: {exc!r}"
        ) from exc
    _verify_target_carries_tag(
        _kv_cache_utils_mod,
        "get_max_concurrency_for_kv_cache_config",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="vllm.v1.core.kv_cache_utils",
    )
    _verify_target_carries_tag(
        _kv_cache_utils_mod,
        "_report_kv_cache_config",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="vllm.v1.core.kv_cache_utils",
    )


def _verify_extract_tool_calls_metrics(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_extract_tool_calls_metrics`` wrapped the
    parser's ``extract_tool_calls`` method."""
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.tool_parsers.qwen3coder_tool_parser import (
            Qwen3CoderToolParser,
        )
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import "
            f"vllm.tool_parsers.qwen3coder_tool_parser.Qwen3CoderToolParser "
            f"for verification: {exc!r}"
        ) from exc
    _verify_target_carries_tag(
        Qwen3CoderToolParser,
        "extract_tool_calls",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="Qwen3CoderToolParser",
    )


# --------------------------------------------------------------------
# Registry. Order matters — see module docstring.
# --------------------------------------------------------------------
#
# Each entry is (importable module name, verifier callable). The
# module must resolve via PYTHONPATH at launcher-execution time;
# under the docker-run command in the module docstring this means
# the file must be bind-mounted into /opt/patches/.

_PatchVerifier = Callable[[ModuleType], None]

_PATCH_MODULES: tuple[str, ...] = (
    # Order matters. qwen3_coder first because it owns the
    # tool-parser surface; the metrics wrapper composes on top of it.
    # hybrid_kv_allocator touches a disjoint surface
    # (vllm.v1.core.kv_cache_utils) so its position is functionally
    # arbitrary, but conventionally we group server-side patches
    # by the area they touch.
    "monkey_patch_qwen3_coder",
    "monkey_patch_hybrid_kv_allocator",
    "monkey_patch_extract_tool_calls_metrics",
    # Append future entries here; do not re-order existing entries
    # without re-auditing the dependency graph between patches.
)

_PATCH_VERIFICATION: dict[str, _PatchVerifier] = {
    "monkey_patch_qwen3_coder": _verify_qwen3_coder,
    "monkey_patch_hybrid_kv_allocator": _verify_hybrid_kv_allocator,
    "monkey_patch_extract_tool_calls_metrics": _verify_extract_tool_calls_metrics,
}


# --------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------


def _import_and_verify(module_name: str) -> ModuleType:
    """Import a server-side patch module and verify its install stuck.

    Does not catch the patch's own typed install exceptions — they
    propagate so the container exits non-zero with the patch's
    specific refusal message. Then runs the launcher's own verifier
    for defense in depth; if that fails it raises
    :class:`PatchVerificationError`.

    Returns the imported module so the caller can keep a reference
    (preventing the module from being garbage-collected and re-imported
    on a later attribute lookup, which would re-run the install).
    """
    if module_name not in _PATCH_VERIFICATION:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] no verifier registered for patch module "
            f"{module_name!r}. Refusing to import a patch the launcher "
            f"cannot post-validate. Add an entry to _PATCH_VERIFICATION."
        )

    # The patch's own import-time validation runs here. Any
    # MonkeyPatchRefusedError (or analogous typed exception from a
    # future patch) propagates unchanged.
    patch_module: ModuleType = importlib.import_module(module_name)

    verifier: _PatchVerifier = _PATCH_VERIFICATION[module_name]
    verifier(patch_module)
    return patch_module


def _handoff_to_vllm_cli(argv: list[str]) -> None:
    """Hand off to vLLM's CLI exactly as `vllm <argv>` would.

    Munges :data:`sys.argv` so argv[0] is the canonical ``"vllm"``
    string the entry-point script uses, then runs
    ``vllm.entrypoints.cli.main`` under ``__main__``. ``alter_sys=True``
    ensures the runpy'd module sees the munged ``sys.argv`` (without
    it, the module would still see the launcher's original argv).

    Does not catch exceptions from the runpy'd module. A failure in
    vLLM's argparse, subcommand dispatch, or server startup must
    surface as the user's diagnostic, not be wrapped by the launcher.
    """
    sys.argv = [_VLLM_ARGV0, *argv]
    runpy.run_module(
        _VLLM_CLI_MODULE, run_name="__main__", alter_sys=True
    )


def _usage_error_and_exit() -> None:
    """Refuse to invoke vLLM with no subcommand and exit non-zero.

    Mirrors the shape of vLLM's own bare-`vllm` behaviour (which
    prints help and returns 0) but treats a bare launcher invocation
    as a deployment misconfiguration: the docker entrypoint is
    expected to pass ``serve <args>`` and a missing subcommand
    almost certainly means the docker command was edited wrong.
    Raising loudly here is more useful than printing help and
    returning 0 because the container would then exit 0 immediately,
    which a process supervisor reads as "served and finished".
    """
    sys.stderr.write(
        f"[{_LAUNCHER_TAG}] usage: python launch_with_patches.py "
        f"<vllm-subcommand> [args...]\n"
        f"  e.g. python launch_with_patches.py serve --model ... "
        f"--host 0.0.0.0 --port 8000\n"
        f"This launcher is a drop-in replacement for `vllm` that first "
        f"installs the server-side patches in {_PATCH_MODULES!r}. It "
        f"requires at least a subcommand argument; vLLM's own CLI will "
        f"validate the rest.\n"
    )
    raise SystemExit(2)


def main(argv: Iterable[str] | None = None) -> None:
    """Install registered patches in order, verify each, then exec vLLM.

    :param argv: subcommand args to forward to vLLM, defaulting to
        ``sys.argv[1:]``. argv[0] is *not* part of this list — the
        launcher synthesizes the canonical ``"vllm"`` argv[0] before
        handoff.
    """
    args: list[str] = list(argv if argv is not None else sys.argv[1:])
    if not args:
        _usage_error_and_exit()

    # Refuse to proceed if vLLM itself isn't importable. Without this
    # check the first patch import dies with a confusing ImportError
    # from inside the patch module's own `import vllm` statement; here
    # we surface it with a launcher-specific message that points at
    # the deployment misconfiguration (PYTHONPATH wrong, image broken,
    # vllm package absent).
    try:
        importlib.import_module("vllm")
    except ImportError as exc:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] vllm is not importable from the current "
            f"Python environment ({sys.executable!r}, sys.path={sys.path!r}). "
            f"The launcher cannot install server-side patches into a "
            f"runtime that does not have vLLM. Check the container image, "
            f"PYTHONPATH, and that the launcher is being run inside the "
            f"vllm/vllm-openai container."
        ) from exc

    # Strict ordered install. Each patch's own typed exceptions and
    # the launcher's PatchVerificationError propagate unchanged.
    installed: list[ModuleType] = []
    for module_name in _PATCH_MODULES:
        installed.append(_import_and_verify(module_name))

    # Hold a reference to every imported patch module for the lifetime
    # of the process so the install side-effects cannot be undone by
    # garbage collection of a stale module reference. This is paranoia,
    # but the cost is one tuple of N module objects.
    globals()["_INSTALLED_PATCHES"] = tuple(installed)

    _handoff_to_vllm_cli(args)


if __name__ == "__main__":
    main()
