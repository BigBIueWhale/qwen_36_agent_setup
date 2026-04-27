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
      --network host \\
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\
      -v "$PWD/sitecustomize.py:/opt/patches/sitecustomize.py:ro" \\
      -v "$PWD/monkey_patch_qwen3_coder.py:/opt/patches/monkey_patch_qwen3_coder.py:ro" \\
      -v "$PWD/monkey_patch_hybrid_kv_allocator.py:/opt/patches/monkey_patch_hybrid_kv_allocator.py:ro" \\
      -v "$PWD/monkey_patch_extract_tool_calls_metrics.py:/opt/patches/monkey_patch_extract_tool_calls_metrics.py:ro" \\
      -v "$PWD/monkey_patch_extract_tool_calls_streaming_metrics.py:/opt/patches/monkey_patch_extract_tool_calls_streaming_metrics.py:ro" \\
      -v "$PWD/monkey_patch_reasoning_field_egress.py:/opt/patches/monkey_patch_reasoning_field_egress.py:ro" \\
      -v "$PWD/monkey_patch_reasoning_field_ingest.py:/opt/patches/monkey_patch_reasoning_field_ingest.py:ro" \\
      -v "$PWD/monkey_patch_tool_call_in_think_rescue.py:/opt/patches/monkey_patch_tool_call_in_think_rescue.py:ro" \\
      -v "$PWD/launch_with_patches.py:/opt/patches/launch.py:ro" \\
      -e HF_HUB_ENABLE_HF_TRANSFER=1 \\
      -e VLLM_USE_V1=1 \\
      -e PYTHONPATH=/opt/patches \\
      --entrypoint python3 \\
      vllm/vllm-openai@sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba \\
      /opt/patches/launch.py serve \\
      --model QuantTrio/Qwen3.6-27B-AWQ \\
      --revision 9b507bdc9afafb87b7898700cc2a591aa6639461 \\
      --host 127.0.0.1 \\
      ...  # remaining --port / --served-model-name / ... flags unchanged

Three structural changes from the pre-launcher docker command:

* Each patch module is bind-mounted into ``/opt/patches/`` by its
  real filename so ``import monkey_patch_<n>`` resolves under
  ``PYTHONPATH=/opt/patches``. Do NOT rename them in the bind-mount.
* ``sitecustomize.py`` is bind-mounted at the same location. CPython's
  ``site.py`` auto-imports ``sitecustomize`` from ``sys.path`` at
  every interpreter startup, including the spawned EngineCore
  subprocess. This is load-bearing for ``monkey_patch_hybrid_kv_allocator``
  (patch 2), whose target functions are called only by EngineCore;
  without sitecustomize, patch 2's launcher install in PID 1 has no
  effect on EngineCore and the boot-log "GPU KV cache size" line
  shows the unpatched (~4× under-counted) value. See
  ``sitecustomize.py``'s docstring for the full rationale.
* ``PYTHONSTARTUP`` is dropped entirely. The ``PYTHONPATH``
  prepend and the ``--entrypoint python3 /opt/patches/launch.py``
  override replace it. (CPython only honors ``PYTHONSTARTUP`` in
  interactive mode; the container's non-interactive entrypoint
  never reads it. That is this launcher's reason for existing.)
* ``--network host`` plus ``--host 127.0.0.1`` binds vLLM to the
  host's loopback interface only. The previous ``-p 8000:8000`` shape
  defaults to ``0.0.0.0:8000`` on the host side, which on a
  publicly-routable IPv4 host with no firewall would expose the
  /v1/* endpoints (and the /metrics endpoint) to the internet. The
  loopback-only binding is the safe default; deployments that want
  remote access should add an authenticating reverse proxy (nginx,
  Caddy, ingress) bound to a controlled interface, NOT bind vLLM
  itself to a public IP.

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
import os
import runpy
import subprocess
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

    Three layers, each catching a regression class the others would miss:

    1. Tag presence on ``_parse_xml_function_call`` (the v1 verifier;
       kept as the cheapest pre-flight signal).
    2. **Behavioral**: instantiate ``Qwen3CoderToolParser`` with the
       cached Qwen3.6 tokenizer and a synthetic tool list, then call
       ``_parse_xml_function_call`` on a deliberately truncated input
       (``"calculator>\\n<parameter=a"``). The patch's whole purpose is
       to return ``None`` on that input rather than raise ``ValueError``;
       this asserts the function table actually carries the new
       behavior, defending against a regression where the tag is
       stamped but the function-table got swapped back, or where some
       sibling patch composed on top of ours.
    3. Behavioral negative-control on a well-formed input: confirm the
       same parser returns a non-None result (so we know the
       ``None``-on-truncation isn't a parser-completely-broken false
       positive).
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

    # Behavioral check: instantiate the parser and exercise the patched
    # method on the precise input the patch was written to handle.
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import behavioral-check dep "
            f"ChatCompletionToolsParam: {exc!r}"
        ) from exc

    # Decoupled from the served model. The patched ``_parse_xml_function_call``
    # only depends on ``self.tools``, ``self.tool_call_parameter_regex``
    # (compiled in ``__init__`` from a hard-coded pattern), and
    # ``self._convert_param_value`` (a method). It never tokenises anything,
    # so a real Qwen3.6 tokenizer is not required to exercise it.
    #
    # ``Qwen3CoderToolParser.__init__`` (vllm/tool_parsers/qwen3coder_tool_parser.py
    # lines 33-93 at the pinned commit) imposes only two requirements on
    # the tokenizer it is handed:
    #
    # 1. ``self.model_tokenizer`` must be truthy (``if not self.model_tokenizer:
    #    raise ValueError(...)`` at line 71-75).
    # 2. ``self.vocab.get("<tool_call>")`` and ``self.vocab.get("</tool_call>")``
    #    must each return a non-None integer (line 77-84), where ``self.vocab``
    #    is the ``ToolParser.vocab`` cached_property defined at
    #    ``vllm/tool_parsers/abstract_tool_parser.py:79-83`` returning
    #    ``self.model_tokenizer.get_vocab()``.
    #
    # The minimal mock below satisfies both. The integer IDs are arbitrary;
    # the verifier never tokenises anything, only calls
    # ``_parse_xml_function_call`` directly with text. This decoupling
    # means an operator can change the served ``--model`` argv without
    # editing this verifier — the previous version hard-coded
    # ``QuantTrio/Qwen3.6-27B-AWQ`` and would have hit the network or
    # spuriously failed on any other deployment target's cache.
    class _Qwen3CoderVerifierTokenizerMock:
        """Minimal mock satisfying ``Qwen3CoderToolParser.__init__``'s
        contract. NOT a substitute for a real tokenizer in any other
        context — only the two attributes above are populated."""

        # The IDs are arbitrary non-zero ints; they only need to be
        # distinct and non-None so the constructor's
        # ``if ... is None: raise RuntimeError`` guard does not trigger.
        _vocab: dict[str, int] = {
            "<tool_call>": 1_000_001,
            "</tool_call>": 1_000_002,
        }

        def get_vocab(self) -> dict[str, int]:
            return self._vocab

        # Truthy check (``if not self.model_tokenizer:``). A fresh
        # ``object`` instance is already truthy in Python; we set this
        # explicitly to make the contract obvious to a future reader.
        def __bool__(self) -> bool:
            return True

    tokenizer_mock = _Qwen3CoderVerifierTokenizerMock()

    tool_calc = ChatCompletionToolsParam.model_validate(
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "string"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    )
    parser = Qwen3CoderToolParser(tokenizer_mock, tools=[tool_calc])

    # Truncated probe — the exact bug class #39771.
    truncated = "calculator>\n<parameter=a"
    try:
        result = parser._parse_xml_function_call(truncated)
    except ValueError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] behavioral verification of "
            f"{patch_module.__name__!r} FAILED: "
            f"_parse_xml_function_call raised ValueError on truncated "
            f"input {truncated!r} — this is the exact upstream bug the "
            f"patch is supposed to eliminate. The patch tag is present "
            f"but the function table was reverted to the buggy version."
            f"\n  underlying: {exc!r}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 -- any raise is a hard refusal
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] behavioral verification of "
            f"{patch_module.__name__!r} FAILED: "
            f"_parse_xml_function_call raised "
            f"{type(exc).__name__} on truncated input {truncated!r}: "
            f"{exc!r}. The patch was supposed to return None silently."
        ) from exc
    if result is not None:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] behavioral verification of "
            f"{patch_module.__name__!r} FAILED: "
            f"_parse_xml_function_call({truncated!r}) returned "
            f"{result!r}, expected None. The patch's contract is to "
            f"return None on truncated parameter tags so the outer "
            f"loop can drop the call and keep its siblings."
        )

    # Behavioral negative control: well-formed input must still return
    # a real ToolCall. This catches the regression where the patched
    # function returns None on EVERYTHING (rather than only on the
    # truncated case), which would silently break tool calling.
    well_formed = (
        "calculator>\n<parameter=a>\n2\n</parameter>\n"
        "<parameter=b>\n3\n</parameter>\n"
    )
    try:
        good = parser._parse_xml_function_call(well_formed)
    except Exception as exc:  # noqa: BLE001
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] behavioral verification of "
            f"{patch_module.__name__!r} FAILED on negative control: "
            f"_parse_xml_function_call({well_formed!r}) raised "
            f"{type(exc).__name__}: {exc!r}. The patched function is "
            f"refusing on inputs the original would have accepted — "
            f"it has over-broadened the truncation guard."
        ) from exc
    if good is None:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] behavioral verification of "
            f"{patch_module.__name__!r} FAILED on negative control: "
            f"_parse_xml_function_call({well_formed!r}) returned None. "
            f"A well-formed tool call must produce a ToolCall, not "
            f"None. The patched function appears to be returning None "
            f"on all inputs."
        )


def _verify_hybrid_kv_allocator(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_hybrid_kv_allocator`` replaced both targets
    AND the replacements behave with the patched (filtered-groups)
    divisor instead of the unpatched ``len(kv_cache_groups)`` divisor.

    Three layers:

    1. Tag presence on both ``_report_kv_cache_config`` and
       ``get_max_concurrency_for_kv_cache_config``.
    2. **Behavioral**: construct a synthetic ``KVCacheConfig`` mirroring
       the Qwen3.6-27B layout (1 ``FullAttentionSpec`` group + 3
       ``MambaSpec`` groups in mode ``"none"``). Call the patched
       ``get_max_concurrency_for_kv_cache_config``. The patched divisor
       is 1 (only the attention group contributes per-token capacity);
       the unpatched divisor is 4 (all four groups). Compute the
       expected concurrency under each and assert the actual result
       matches the patched value.
    3. Behavioral negative-control: monkey_patch_hybrid_kv_allocator
       can be cleanly distinguished from upstream by the synthetic
       ``KVCacheConfig`` test above. Cosmetic: also exercise
       ``_report_kv_cache_config`` to confirm it doesn't raise on the
       same input (the patched body validates input shape).
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

    # Behavioral: build a synthetic KVCacheConfig and call the patched
    # function; assert the divisor used was the post-patch one.
    try:
        import torch

        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            MambaSpec,
        )
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import behavioral-check deps "
            f"(KVCacheConfig family, torch): {exc!r}"
        ) from exc

    # Qwen3.6-27B layout: 1 attention group of 16 layers (full attn at
    # indices 3, 7, ..., 63) + 3 disjoint MambaSpec groups (3 different
    # GDN shapes per attention block, 16 layers each), 47 GDN layers
    # total — but for the discriminator we only need DIFFERENT counts
    # in the two divisor regimes, not a faithful Qwen3.6 reproduction.
    # 1 attn vs 4 total (1+3) is a 4× ratio, which is unambiguous.
    block_size = 16
    num_kv_heads = 4
    head_size = 256
    num_full_attn_layers = 16
    num_mamba_layers = 16  # per group; identical so the test is symmetric

    full_attn_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )

    # The pinned signature of MambaSpec is (block_size, shapes, dtypes,
    # page_size_padded?, mamba_type?, mamba_cache_mode?, num_speculative_blocks?).
    # mamba_cache_mode="none" so the patch's filter excludes all three
    # Mamba groups (post-patch divisor = 1; pre-patch divisor = 4).
    #
    # We need each Mamba spec to contribute meaningful bytes so the
    # bytes-summed difference between (1 attn) and (1 attn + 3 mamba)
    # is large enough to escape rounding. Pad each Mamba page to 256 MB
    # so unpatched arithmetic sums 1024 MB total (256 MB attn + 3 *
    # 256 MB mamba), versus patched 256 MB. That's a 4x ratio in
    # max_memory_usage_per_request, mapping to a 4x ratio in concurrency.
    huge_padded_bytes = 256 * 1024 * 1024
    mamba_specs: list[MambaSpec] = []
    for shape_tuple in (
        ((128, 128),),
        ((256,),),
        ((1024,),),
    ):
        mamba_specs.append(
            MambaSpec(
                block_size=block_size,
                shapes=shape_tuple,
                dtypes=(torch.bfloat16,) * len(shape_tuple),
                page_size_padded=huge_padded_bytes,
                mamba_cache_mode="none",
            )
        )

    mamba_groups = [
        KVCacheGroupSpec(
            layer_names=[f"mamba.{idx}.{i}" for i in range(num_mamba_layers)],
            kv_cache_spec=spec,
        )
        for idx, spec in enumerate(mamba_specs)
    ]
    full_attn_group = KVCacheGroupSpec(
        layer_names=[f"attn.{i}" for i in range(num_full_attn_layers)],
        kv_cache_spec=full_attn_spec,
    )

    synthetic_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[full_attn_group, *mamba_groups],
    )

    # We need a VllmConfig-shaped object the patch can read. The patch
    # only touches three sub-attributes: cache_config.mamba_cache_mode,
    # parallel_config.{decode,prefill}_context_parallel_size, and
    # model_config.max_model_len. Use a stub via SimpleNamespace; the
    # filter helper does `getattr(getattr(cfg, "cache_config", None),
    # "mamba_cache_mode", "none")` so the stub need only expose those.
    from types import SimpleNamespace

    stub_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode="none",
            num_gpu_blocks_override=None,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            max_model_len=65536,
        ),
        scheduler_config=SimpleNamespace(
            max_num_encoder_input_tokens=0,
        ),
    )

    # Compute the expected post-patch concurrency, mirroring the
    # patched function's arithmetic. The patched divisor is 1 because
    # only the FullAttentionSpec group contributes (mamba_cache_mode
    # is "none"). Compute what the unpatched divisor (4 total groups)
    # would have produced, and assert the actual result matches the
    # patched value, NOT the unpatched value.
    from vllm.utils.math_utils import cdiv

    full_attn_max_bytes = full_attn_spec.max_memory_usage_bytes(
        stub_vllm_config
    )
    page_size_bytes = full_attn_spec.page_size_bytes

    num_layer_per_group_patched = num_full_attn_layers
    max_mem_per_request_patched = (
        num_layer_per_group_patched * full_attn_max_bytes
    )
    memory_per_block_patched = page_size_bytes * num_layer_per_group_patched
    num_blocks_per_request_patched = cdiv(
        max_mem_per_request_patched, memory_per_block_patched
    )
    expected_patched = (
        synthetic_config.num_blocks / num_blocks_per_request_patched
    )

    # The unpatched divisor would have summed all 4 groups' specs in
    # `max_memory_usage_per_request`. For our specs the Mamba groups
    # report O(1) bytes (mamba_cache_mode=none), so the per-request
    # bytes are roughly the same — but the unpatched arithmetic uses
    # `max(...)` over `len(group.layer_names)` across all groups, which
    # is identical to ours since all groups have 16 layers. The
    # discriminating term is the SUM of max_memory_usage_bytes:
    # patched sums only attn (= attn_bytes), unpatched sums attn + 3*mamba
    # (= attn_bytes + 3*mamba_O1_bytes). Even one byte of mamba O(1)
    # state in the sum changes the divisor.

    actual = _kv_cache_utils_mod.get_max_concurrency_for_kv_cache_config(
        stub_vllm_config, synthetic_config
    )

    # We allow a small relative tolerance for floating-point
    # round-trip; the value should be exactly the patched one.
    if not isinstance(actual, (int, float)):
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] behavioral verification of "
            f"{patch_module.__name__!r} FAILED: "
            f"get_max_concurrency_for_kv_cache_config returned "
            f"{actual!r} (type {type(actual).__name__}); expected float."
        )
    if abs(actual - expected_patched) > 1e-6 * max(
        abs(expected_patched), 1.0
    ):
        # Compute what unpatched would have produced for context.
        mamba_total_bytes = sum(
            s.max_memory_usage_bytes(stub_vllm_config) for s in mamba_specs
        )
        max_mem_per_request_unpatched = num_layer_per_group_patched * (
            full_attn_max_bytes + mamba_total_bytes
        )
        num_blocks_per_request_unpatched = cdiv(
            max_mem_per_request_unpatched, memory_per_block_patched
        )
        expected_unpatched = (
            synthetic_config.num_blocks
            / num_blocks_per_request_unpatched
        )
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] behavioral verification of "
            f"{patch_module.__name__!r} FAILED: "
            f"get_max_concurrency_for_kv_cache_config returned "
            f"{actual!r}, expected patched value {expected_patched!r} "
            f"(unpatched would be ~{expected_unpatched!r}). The "
            f"function table appears to be carrying the unpatched "
            f"arithmetic — the patch tag is present but the body has "
            f"been reverted."
        )

    # Cosmetic: exercise _report_kv_cache_config too. It only logs;
    # we assert it does NOT raise on the synthetic input. (The patched
    # body has _validate_kv_cache_config_shape guards that would refuse
    # on a malformed config, so a non-raise here proves the wiring.)
    try:
        _kv_cache_utils_mod._report_kv_cache_config(
            stub_vllm_config, synthetic_config
        )
    except Exception as exc:  # noqa: BLE001
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] behavioral verification of "
            f"{patch_module.__name__!r} FAILED: "
            f"_report_kv_cache_config raised "
            f"{type(exc).__name__}: {exc!r} on a valid synthetic "
            f"config. The patched body is refusing on input the "
            f"contract should accept."
        ) from exc


def _verify_extract_tool_calls_metrics(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_extract_tool_calls_metrics`` wrapped the
    parser's ``extract_tool_calls`` method (non-streaming entry)."""
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


def _verify_extract_tool_calls_streaming_metrics(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_extract_tool_calls_streaming_metrics``
    wrapped the parser's ``extract_tool_calls_streaming`` method
    (streaming entry — the one the serving layer calls per-chunk under
    ``stream=True``).
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
        "extract_tool_calls_streaming",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="Qwen3CoderToolParser",
    )


def _verify_reasoning_field_egress(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_reasoning_field_egress`` applied the alias
    end-to-end across every dump path vLLM actually serialises.

    Independent of the patch's own post-install checks. Three layers of
    verification, each catching a regression class the others would miss:

    1. **Tag presence on every target class.** The patch stamps
       ``__qwen36_egress_patch__`` on six classes:

       * ``ChatMessage``, ``DeltaMessage`` (leaves)
       * ``ChatCompletionResponseChoice``, ``ChatCompletionResponseStreamChoice``
       * ``ChatCompletionResponse``, ``ChatCompletionStreamResponse``

       Each must resolve via both ``getattr`` and
       ``inspect.getattr_static`` (defending against metaclass
       ``__getattribute__`` overrides).

    2. **Standalone leaf dump.** Constructs ``ChatMessage`` and
       ``DeltaMessage`` instances directly and asserts ``model_dump()``
       emits ``"reasoning_content"`` and not ``"reasoning"``. This was
       the only check v1's verifier ran; it passes even when the
       wrapper-rebuild step is missing, which is why v1's patch
       silently failed in production despite v1's verifier reporting
       success.

    3. **Nested wire dump.** Constructs a real
       ``ChatCompletionResponse`` (the non-streaming wire shape from
       ``api_router.py:70``) and a real ``ChatCompletionStreamResponse``
       (the streaming wire shape from
       ``serving.py:685/721/1208/1233``), serialises each via the
       exact API call vLLM uses (``model_dump_json()`` for
       non-streaming, ``model_dump_json(exclude_unset=True)`` for
       streaming), and substring-checks the wire bytes for
       ``"reasoning_content":"<probe>"`` presence and
       ``"reasoning":`` absence. THIS IS THE LOAD-BEARING CHECK
       v1's verifier missed.
    """
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatCompletionResponseStreamChoice,
            ChatCompletionStreamResponse,
            ChatMessage,
        )
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaMessage,
            UsageInfo,
        )
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import egress patch's six target "
            f"classes (ChatMessage, DeltaMessage, "
            f"ChatCompletionResponseChoice, "
            f"ChatCompletionResponseStreamChoice, "
            f"ChatCompletionResponse, ChatCompletionStreamResponse) "
            f"for verification: {exc!r}"
        ) from exc

    _probe_value = "__qwen36_launcher_egress_probe__"

    # --- Layer 1: tag presence on every target class.
    targets_with_names: tuple[tuple[type, str], ...] = (
        (ChatMessage, "ChatMessage"),
        (DeltaMessage, "DeltaMessage"),
        (ChatCompletionResponseChoice, "ChatCompletionResponseChoice"),
        (
            ChatCompletionResponseStreamChoice,
            "ChatCompletionResponseStreamChoice",
        ),
        (ChatCompletionResponse, "ChatCompletionResponse"),
        (ChatCompletionStreamResponse, "ChatCompletionStreamResponse"),
    )
    for target_cls, target_name in targets_with_names:
        dynamic = getattr(target_cls, "__qwen36_egress_patch__", None)
        if dynamic != expected_tag:
            raise PatchVerificationError(
                f"[{_LAUNCHER_TAG}] post-install tag verification "
                f"failed for {patch_module.__name__!r}: {target_name} "
                f"carries __qwen36_egress_patch__={dynamic!r}, expected "
                f"{expected_tag!r}. The egress patch did not stamp this "
                f"target — check that {target_name} is in its target list."
            )
        static = inspect.getattr_static(
            target_cls, "__qwen36_egress_patch__", None
        )
        if static != expected_tag:
            raise PatchVerificationError(
                f"[{_LAUNCHER_TAG}] static-lookup tag verification "
                f"failed for {patch_module.__name__!r}: "
                f"inspect.getattr_static({target_name}, "
                f"'__qwen36_egress_patch__')={static!r}, expected "
                f"{expected_tag!r}. Normal attribute lookup and static "
                f"lookup disagree — a metaclass-level shim is shadowing "
                f"the install."
            )

    # --- Layer 2: standalone leaf dump (the v1-style check, kept).
    for leaf_cls, leaf_name in (
        (ChatMessage, "ChatMessage"),
        (DeltaMessage, "DeltaMessage"),
    ):
        if "role" in leaf_cls.model_fields and leaf_cls.model_fields[
            "role"
        ].is_required():
            instance = leaf_cls(role="assistant", reasoning=_probe_value)
        else:
            instance = leaf_cls(reasoning=_probe_value)
        dumped = instance.model_dump()
        if dumped.get("reasoning_content") != _probe_value:
            raise PatchVerificationError(
                f"[{_LAUNCHER_TAG}] standalone-leaf egress verification "
                f"failed for {leaf_name}: model_dump()['reasoning_content']"
                f"={dumped.get('reasoning_content')!r}, expected probe. "
                f"Full dump: {dumped!r}"
            )
        if "reasoning" in dumped:
            raise PatchVerificationError(
                f"[{_LAUNCHER_TAG}] standalone-leaf egress verification "
                f"failed for {leaf_name}: model_dump() still contains "
                f"the bare 'reasoning' key; the alias and the unaliased "
                f"name are being emitted together. Full dump: {dumped!r}"
            )

    # --- Layer 3: nested wire-shape dump. This is the load-bearing
    # addition — the production wire path is not the leaf, it is the
    # wrapper, and v1's verifier never tested the wrapper.
    nested_response = ChatCompletionResponse(
        id="probe-id",
        model="probe-model",
        usage=UsageInfo(
            prompt_tokens=0,
            total_tokens=0,
            completion_tokens=0,
        ),
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    reasoning=_probe_value,
                ),
            ),
        ],
    )
    nested_wire = nested_response.model_dump_json()
    expected_nested_substr = f'"reasoning_content":"{_probe_value}"'
    if expected_nested_substr not in nested_wire:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] nested-egress verification failed for "
            f"ChatCompletionResponse: wire JSON does not contain "
            f"{expected_nested_substr!r}. This is the production "
            f"non-streaming wire path (api_router.py:70) — clients "
            f"would silently lose reasoning. Full wire: {nested_wire!r}"
        )
    if '"reasoning":' in nested_wire:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] nested-egress verification failed for "
            f"ChatCompletionResponse: wire JSON still contains the "
            f'bare "reasoning": key. Full wire: {nested_wire!r}'
        )

    nested_stream = ChatCompletionStreamResponse(
        id="probe-id",
        model="probe-model",
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(reasoning=_probe_value),
            ),
        ],
    )
    # serving.py:685/721/1208/1233 all use exclude_unset=True.
    nested_stream_wire = nested_stream.model_dump_json(exclude_unset=True)
    if expected_nested_substr not in nested_stream_wire:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] nested-egress verification failed for "
            f"ChatCompletionStreamResponse: wire JSON does not "
            f"contain {expected_nested_substr!r}. This is the "
            f"production streaming wire path "
            f"(serving.py:685/721/1208/1233) — streaming clients "
            f"would silently lose reasoning per chunk. Full wire: "
            f"{nested_stream_wire!r}"
        )
    if '"reasoning":' in nested_stream_wire:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] nested-egress verification failed for "
            f"ChatCompletionStreamResponse: wire JSON still contains "
            f'the bare "reasoning": key. Full wire: '
            f"{nested_stream_wire!r}"
        )


def _verify_reasoning_field_ingest(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_reasoning_field_ingest`` wrapped
    ``vllm.entrypoints.chat_utils._parse_chat_message_content``.
    """
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.entrypoints import chat_utils as _chat_utils_mod
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import vllm.entrypoints.chat_utils "
            f"for verification: {exc!r}"
        ) from exc
    _verify_target_carries_tag(
        _chat_utils_mod,
        "_parse_chat_message_content",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="vllm.entrypoints.chat_utils",
    )


def _verify_tool_call_in_think_rescue(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_tool_call_in_think_rescue`` wrapped BOTH
    the non-streaming ``extract_reasoning`` and the streaming
    ``extract_reasoning_streaming`` methods on ``Qwen3ReasoningParser``.
    Both must bear the tag; a half-applied rescue (only one wrapped)
    would leave one code path silently vulnerable to the §6.1 failure
    mode this patch exists to eliminate.
    """
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import "
            f"vllm.reasoning.qwen3_reasoning_parser.Qwen3ReasoningParser "
            f"for verification: {exc!r}"
        ) from exc
    _verify_target_carries_tag(
        Qwen3ReasoningParser,
        "extract_reasoning",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="Qwen3ReasoningParser",
    )
    _verify_target_carries_tag(
        Qwen3ReasoningParser,
        "extract_reasoning_streaming",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="Qwen3ReasoningParser",
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
    # Order notes:
    #
    # * qwen3_coder first because it owns the tool-parser surface; the
    #   metrics wrappers compose on top of it.
    # * hybrid_kv_allocator touches a disjoint surface
    #   (vllm.v1.core.kv_cache_utils) so its position is functionally
    #   arbitrary, grouped here with the other tool-parser-adjacent patches.
    # * extract_tool_calls_metrics then extract_tool_calls_streaming_metrics:
    #   both wrap methods on Qwen3CoderToolParser (disjoint methods);
    #   either order works. The non-streaming one is listed first only
    #   so that when it registers the shared Prometheus counter, the
    #   streaming one discovers and reuses it on the second pass.
    # * reasoning_field_egress and reasoning_field_ingest are
    #   independent of the tool-parser patches (different surfaces:
    #   Pydantic model + chat_utils function). Order between them is
    #   arbitrary.
    # * tool_call_in_think_rescue wraps the reasoning parser. It must
    #   come AFTER the reasoning_field_egress patch because the egress
    #   patch calls `model_rebuild(force=True)` on DeltaMessage, and
    #   the rescue patch constructs DeltaMessage instances — the
    #   Pydantic schema must be stable by that point.
    "monkey_patch_qwen3_coder",
    "monkey_patch_hybrid_kv_allocator",
    "monkey_patch_extract_tool_calls_metrics",
    "monkey_patch_extract_tool_calls_streaming_metrics",
    "monkey_patch_reasoning_field_egress",
    "monkey_patch_reasoning_field_ingest",
    "monkey_patch_tool_call_in_think_rescue",
    # Append future entries here; do not re-order existing entries
    # without re-auditing the dependency graph between patches.
)

_PATCH_VERIFICATION: dict[str, _PatchVerifier] = {
    "monkey_patch_qwen3_coder": _verify_qwen3_coder,
    "monkey_patch_hybrid_kv_allocator": _verify_hybrid_kv_allocator,
    "monkey_patch_extract_tool_calls_metrics": _verify_extract_tool_calls_metrics,
    "monkey_patch_extract_tool_calls_streaming_metrics": _verify_extract_tool_calls_streaming_metrics,
    "monkey_patch_reasoning_field_egress": _verify_reasoning_field_egress,
    "monkey_patch_reasoning_field_ingest": _verify_reasoning_field_ingest,
    "monkey_patch_tool_call_in_think_rescue": _verify_tool_call_in_think_rescue,
}


# --------------------------------------------------------------------
# Pre-flight checks (run BEFORE the per-patch import loop).
# --------------------------------------------------------------------
#
# These three checks defend against the spawn-vs-fork class of failure
# (sitecustomize missing, registry drift between launcher and
# sitecustomize, or a subprocess-only deviation that hides patch 2's
# install from EngineCore). Each refuses with a typed
# :class:`LauncherError` on failure; none of them log anything in the
# happy path.


_SITECUSTOMIZE_MISSING_MSG: str = (
    "sitecustomize.py is not the qwen36-agent-setup version (or is not "
    "importable at all). The most likely cause is the docker run "
    "command is missing the bind-mount "
    "`-v \"$PWD/sitecustomize.py:/opt/patches/sitecustomize.py:ro\"`. "
    "Without our sitecustomize, vLLM v1's spawned EngineCore subprocess "
    "will not re-install the patches and patch 2 "
    "(monkey_patch_hybrid_kv_allocator) will be silently dead in "
    "EngineCore — the boot-log GPU KV cache line will show the unpatched "
    "(~4x under-counted) value. See sitecustomize.py's docstring and "
    "README §7.S for the full rationale."
)


def _preflight_sitecustomize_present() -> ModuleType:
    """Refuse to boot if our ``sitecustomize`` is not loaded.

    The launcher's PID-1 patch loop runs in the API-server process,
    but vLLM v1 spawns its EngineCore as a fresh Python interpreter
    (CUDA forbids ``fork`` after init). The spawned interpreter does
    NOT inherit the parent's ``sys.modules`` — without OUR
    ``sitecustomize`` on its ``sys.path`` to re-import the patches,
    patch 2 (``monkey_patch_hybrid_kv_allocator``) is silently dead in
    EngineCore even though the launcher's PID-1 verifier reports
    success. A missing ``sitecustomize.py`` bind-mount is the single
    most common cause of this regression and the failure mode is
    silent (boot-log emits the unpatched ``GPU KV cache size`` line).

    Note: Debian's CPython ships a stub ``/usr/lib/python3.12/sitecustomize.py``
    with no module-level globals, so a bare ``import sitecustomize``
    will succeed even when our bind-mount is absent. We discriminate
    by checking the module exposes the expected ``_PATCH_MODULES``
    tuple — only OUR sitecustomize defines it.
    """
    try:
        import sitecustomize  # noqa: F401  -- imported for side-effects
    except ImportError as exc:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] sitecustomize is not importable from the "
            f"current Python environment ({sys.executable!r}, "
            f"sys.path={sys.path!r}). {_SITECUSTOMIZE_MISSING_MSG}"
        ) from exc

    if not hasattr(sitecustomize, "_PATCH_MODULES"):
        # The Debian-stub sitecustomize was loaded instead of ours.
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] sitecustomize was importable as "
            f"{sitecustomize.__file__!r} but does NOT define "
            f"`_PATCH_MODULES` — this is a system-default "
            f"sitecustomize, not the qwen36-agent-setup one. "
            f"{_SITECUSTOMIZE_MISSING_MSG}"
        )

    if not getattr(__import__("builtins"), "_qwen36_sitecustomize_loaded", False):
        # Our sitecustomize was somehow imported but its module-level
        # install code never fired. This would only happen if
        # `_qwen36_sitecustomize_loaded` was forged on builtins, or if
        # the file was edited to skip the install. Either way the
        # patches did NOT get auto-installed in PID 1, so the launcher
        # cannot rely on the PID-1 install side-effect.
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] sitecustomize was importable as "
            f"{sitecustomize.__file__!r} and exposes _PATCH_MODULES, "
            f"but the install-completion sentinel "
            f"`builtins._qwen36_sitecustomize_loaded` was not set. The "
            f"sitecustomize file may have been edited to skip the "
            f"install loop, or its module-level code raised silently "
            f"before reaching the install. Audit sitecustomize.py."
        )

    return sitecustomize


def _preflight_registry_drift_check(sitecustomize_module: ModuleType) -> None:
    """Refuse to boot if ``sitecustomize._PATCH_MODULES`` and
    :data:`_PATCH_MODULES` (this module) disagree.

    Drift between the two registries is a well-known footgun: a
    maintainer adds a new patch to one and forgets the other. The
    sitecustomize tuple is what gets imported in EngineCore; the
    launcher tuple is what gets verified in PID 1. A drift means
    EngineCore would either skip an installed patch (sitecustomize
    short, launcher long) or install a patch the launcher doesn't
    know how to verify (sitecustomize long, launcher short). Both
    are silent failures.
    """
    sitecustomize_modules = getattr(
        sitecustomize_module, "_PATCH_MODULES", None
    )
    if not isinstance(sitecustomize_modules, tuple) or not all(
        isinstance(m, str) for m in sitecustomize_modules
    ):
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] sitecustomize._PATCH_MODULES is not a "
            f"tuple of strings; got {type(sitecustomize_modules).__name__}: "
            f"{sitecustomize_modules!r}. The launcher cannot cross-check "
            f"the registry against a malformed value. Edit "
            f"sitecustomize.py to restore the documented contract."
        )
    if sitecustomize_modules != _PATCH_MODULES:
        # Build a precise diff so the operator sees exactly which entry
        # is in one tuple and not the other.
        only_in_sitecustomize = [
            m for m in sitecustomize_modules if m not in _PATCH_MODULES
        ]
        only_in_launcher = [
            m for m in _PATCH_MODULES if m not in sitecustomize_modules
        ]
        order_drift = (
            sitecustomize_modules != _PATCH_MODULES
            and not only_in_sitecustomize
            and not only_in_launcher
        )
        diff_lines = []
        if only_in_sitecustomize:
            diff_lines.append(
                f"  only in sitecustomize._PATCH_MODULES: "
                f"{only_in_sitecustomize!r}"
            )
        if only_in_launcher:
            diff_lines.append(
                f"  only in launch_with_patches._PATCH_MODULES: "
                f"{only_in_launcher!r}"
            )
        if order_drift:
            diff_lines.append(
                f"  same set, different order: "
                f"sitecustomize={sitecustomize_modules!r}, "
                f"launcher={_PATCH_MODULES!r}"
            )
        diff_text = "\n".join(diff_lines)
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] registry drift detected between "
            f"sitecustomize._PATCH_MODULES and "
            f"launch_with_patches._PATCH_MODULES:\n{diff_text}\n"
            f"Both tuples must be byte-identical (same entries in the "
            f"same order). The launcher uses its own tuple for the "
            f"per-patch verifier loop; sitecustomize uses its tuple for "
            f"the EngineCore re-install. A drift means EngineCore and "
            f"the API server are running with different patch sets, "
            f"which is silent in the boot log but visible at request "
            f"time as cross-process behaviour disagreement."
        )


_SUBPROCESS_INSTALL_PROBE: str = """\
import importlib
import sys

_TARGETS = [
    ("monkey_patch_qwen3_coder",
     "vllm.tool_parsers.qwen3coder_tool_parser",
     "Qwen3CoderToolParser",
     "_parse_xml_function_call"),
    ("monkey_patch_hybrid_kv_allocator",
     "vllm.v1.core.kv_cache_utils",
     None,
     "get_max_concurrency_for_kv_cache_config"),
    ("monkey_patch_hybrid_kv_allocator",
     "vllm.v1.core.kv_cache_utils",
     None,
     "_report_kv_cache_config"),
    ("monkey_patch_extract_tool_calls_metrics",
     "vllm.tool_parsers.qwen3coder_tool_parser",
     "Qwen3CoderToolParser",
     "extract_tool_calls"),
    ("monkey_patch_extract_tool_calls_streaming_metrics",
     "vllm.tool_parsers.qwen3coder_tool_parser",
     "Qwen3CoderToolParser",
     "extract_tool_calls_streaming"),
    ("monkey_patch_reasoning_field_ingest",
     "vllm.entrypoints.chat_utils",
     None,
     "_parse_chat_message_content"),
    ("monkey_patch_tool_call_in_think_rescue",
     "vllm.reasoning.qwen3_reasoning_parser",
     "Qwen3ReasoningParser",
     "extract_reasoning"),
    ("monkey_patch_tool_call_in_think_rescue",
     "vllm.reasoning.qwen3_reasoning_parser",
     "Qwen3ReasoningParser",
     "extract_reasoning_streaming"),
]

# sitecustomize already ran (CPython's site.py auto-loaded it before
# this -c probe started executing); the patches are in sys.modules
# of THIS interpreter. Re-import each target FROM SCRATCH (not from
# the patch module's cached references) so what we read is what
# vLLM's request handler will see.
lines = []
for patch_name, vllm_mod, cls_name, attr in _TARGETS:
    try:
        mod = importlib.import_module(vllm_mod)
    except ImportError as exc:
        lines.append(f"FAIL {patch_name} {vllm_mod} import: {exc!r}")
        continue
    target = mod
    if cls_name is not None:
        target = getattr(mod, cls_name, None)
        if target is None:
            lines.append(
                f"FAIL {patch_name} {vllm_mod}.{cls_name} not found"
            )
            continue
    fn = getattr(target, attr, None)
    if fn is None:
        lines.append(
            f"FAIL {patch_name} {vllm_mod}.{cls_name or '<module>'}.{attr} "
            f"not found"
        )
        continue
    tag = getattr(fn, "__qwen36_patch__", None)
    lines.append(
        f"OK {patch_name} {cls_name or '<module>'}.{attr} {tag!r}"
    )

# Egress patch: leaf-class-tag style probe (different attribute from
# function patches). Only check the tag is non-None and matches the
# patch's _PATCH_TAG.
try:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatMessage as _ChatMessage,
    )
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaMessage as _DeltaMessage,
    )
    chat_tag = getattr(_ChatMessage, "__qwen36_egress_patch__", None)
    delta_tag = getattr(_DeltaMessage, "__qwen36_egress_patch__", None)
    lines.append(
        f"OK monkey_patch_reasoning_field_egress ChatMessage {chat_tag!r}"
    )
    lines.append(
        f"OK monkey_patch_reasoning_field_egress DeltaMessage {delta_tag!r}"
    )
except Exception as exc:  # noqa: BLE001 -- probe must report all failures
    lines.append(
        f"FAIL monkey_patch_reasoning_field_egress probe: {exc!r}"
    )

# Print interpreter identity so the launcher can confirm the spawn
# subprocess is the same Python it would itself run.
print("EXEC", sys.executable)
for line in lines:
    print(line)
"""


def _preflight_subprocess_install_check() -> None:
    """Spawn a fresh interpreter, run the install probe, and confirm
    every patch tag is present and matches.

    Defends against three failure modes the in-process check cannot:

    1. ``sitecustomize`` is importable from PID 1's ``sys.path`` but
       NOT from a child interpreter's (e.g. PYTHONPATH not exported,
       env var unset, sys.path differs).
    2. The child interpreter is not the same Python as PID 1
       (different venv, different ``sys.executable``).
    3. The patches' subprocess install has a regression that doesn't
       reproduce in the in-process check (e.g. import side-effect
       races against EngineCore's own imports).

    Probes via ``subprocess.run([sys.executable, "-c", <probe>])``.
    The probe prints one line per target; the launcher parses the
    output and refuses if any line starts with ``FAIL`` or carries a
    ``None`` tag.
    """
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_INSTALL_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess install check exited "
            f"non-zero ({proc.returncode}). The spawn child cannot "
            f"install the patches — most likely sitecustomize.py is "
            f"not on the child's sys.path (PYTHONPATH unset or wrong) "
            f"or one of the patches refused on its own landmark check.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    output_lines = proc.stdout.strip().splitlines()
    exec_line = next(
        (line for line in output_lines if line.startswith("EXEC ")), None
    )
    target_lines = [
        line
        for line in output_lines
        if line.startswith("OK ") or line.startswith("FAIL ")
    ]

    if exec_line is None:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess install check produced no "
            f"EXEC line. The probe did not run to completion.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    child_executable = exec_line[len("EXEC "):].strip()
    if child_executable != sys.executable:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess install check ran under a "
            f"different interpreter ({child_executable!r}) than the "
            f"launcher itself ({sys.executable!r}). The container's "
            f"spawn children must be the same Python the launcher is. "
            f"Check `--entrypoint python3` in the docker run command."
        )

    failures = [line for line in target_lines if line.startswith("FAIL ")]
    if failures:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess install check found "
            f"{len(failures)} target(s) the spawn child could not "
            f"verify. Each failure is one patch that would silently "
            f"NOT install in EngineCore:\n  "
            + "\n  ".join(failures)
            + f"\nstderr:\n{proc.stderr}"
        )

    untagged = [
        line
        for line in target_lines
        if line.startswith("OK ") and line.endswith("None")
    ]
    if untagged:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess install check found "
            f"{len(untagged)} target(s) without the expected "
            f"__qwen36_patch__ / __qwen36_egress_patch__ tag in the "
            f"spawn child. The patch imported but did not stamp its "
            f"target — the install ran in PID 1 but NOT in the child:\n"
            f"  " + "\n  ".join(untagged)
        )


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

    # Pre-flight: refuse to boot if sitecustomize is missing or the
    # registry has drifted. These checks run BEFORE the per-patch
    # import loop because (a) sitecustomize already installed the
    # patches if it was importable, so the loop below is a no-op
    # validation rather than an install path; and (b) catching a
    # missing bind-mount HERE produces a launcher-specific error
    # message naming the missing flag, instead of a confusing
    # boot-log line emitted by the spawned EngineCore subprocess
    # several seconds later.
    sitecustomize_module = _preflight_sitecustomize_present()
    _preflight_registry_drift_check(sitecustomize_module)
    _preflight_subprocess_install_check()

    # Strict ordered install. Each patch's own typed exceptions and
    # the launcher's PatchVerificationError propagate unchanged.
    # Because sitecustomize already imported every patch in PID 1's
    # sys.modules, these importlib.import_module calls hit cache and
    # the loop is now a per-patch verifier pass (defense in depth).
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
