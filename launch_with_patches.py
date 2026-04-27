#!/usr/bin/env python3
"""launch_with_patches.py — strict, fail-loud entrypoint that installs
server-side runtime patches BEFORE handing off to vLLM's CLI.

Why this file exists
--------------------

CPython only honours ``PYTHONSTARTUP`` in *interactive* mode (see
``Lib/site.py`` / ``Modules/main.c``: ``Py_InspectFlag`` / the ``-i``
switch); the container's ``ENTRYPOINT ["vllm", "serve"]`` is
non-interactive, so ``PYTHONSTARTUP`` is never read. The launcher
replaces the container entrypoint with ``python /opt/patches/launch.py
serve <args>`` so each patch in :data:`_PATCH_MODULES` imports BEFORE
vLLM's CLI dispatches.

What it does
------------

1. Imports each module in :data:`_PATCH_MODULES` in declared order.
   Each patch is strict: it validates landmarks and raises a typed
   exception on mismatch. The launcher does **not** catch those — they
   propagate and the container exits non-zero with the patch's specific
   refusal message intact.
2. After each import, runs a per-module verifier from
   :data:`_PATCH_VERIFICATION`. This is defense in depth above the
   patch's own post-install check.
3. Hands off to ``vllm.entrypoints.cli.main`` via
   :func:`runpy.run_module` with ``alter_sys=True``, after munging
   :data:`sys.argv` so argv[0] is ``"vllm"``.

Pre-flight checks (run BEFORE the import loop) defend against the
spawn-vs-fork failure class: sitecustomize-present, registry drift
between launcher and sitecustomize, and a subprocess install probe
that confirms the spawned EngineCore interpreter sees patched targets.

Docker run shape
----------------

The canonical ``docker run`` invocation is the one in **README §8.2**;
it is the single source of truth for the deployment. Do not paraphrase
it here — drift between this docstring and §8.2 has bitten this repo
before. Load-bearing properties of that command (in case anyone is
auditing this file in isolation): ``sitecustomize.py`` bind-mount
(re-installs patches in spawned EngineCore — without it,
``monkey_patch_hybrid_kv_allocator`` is silently dead in the EngineCore
subprocess); ``PYTHONPATH=/opt/patches`` (CPython's ``site.py`` finds
sitecustomize on it); ``--entrypoint python3`` (overrides the image's
``[vllm, serve]``); ``--network host`` plus ``--host 127.0.0.1``
(loopback-only binding so /v1/* and /metrics are not world-accessible);
``--restart unless-stopped`` (auto-recover from engine crash);
``--health-cmd 'curl /health'`` (shallow liveness — deep wedge probe
lives at ``health_probe.sh`` per README §8.4).

Adding a new patch
------------------

1. Bind-mount its file at ``/opt/patches/<name>.py``.
2. Append its name to :data:`_PATCH_MODULES` (and to
   ``sitecustomize._PATCH_MODULES``, in the same order).
3. Append a verifier to :data:`_PATCH_VERIFICATION`.

The launcher will not catch any patch's typed install exception, will
not continue to handoff if a verifier reports failure, and will not
catch ``runpy``'s exceptions on handoff.
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
# Each verifier takes the imported patch module and raises
# PatchVerificationError if the patch's target does not bear the
# expected __qwen36_patch__ tag. Verifiers re-import vLLM internals
# from scratch (not from the patch module's cached references) so what
# gets checked is what ``vllm serve`` will see at request time.


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
    """Verify a class/module attribute bears the patch tag via both
    ``getattr`` and ``inspect.getattr_static`` — disagreement means a
    metaclass shim is shadowing the install. Also verifies
    ``__wrapped_original__`` is present (forging defense)."""
    installed_dynamic = getattr(target_obj, attr_name, None)
    if installed_dynamic is None:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {target_description}.{attr_name} is missing "
            f"after import of {patch_module_name!r}."
        )
    try:
        installed_static = inspect.getattr_static(target_obj, attr_name)
    except AttributeError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] inspect.getattr_static cannot resolve "
            f"{target_description}.{attr_name}: {exc!r}."
        ) from exc

    dynamic_tag = getattr(installed_dynamic, "__qwen36_patch__", None)
    static_tag = getattr(installed_static, "__qwen36_patch__", None)
    if dynamic_tag != expected_tag:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module_name!r}: "
            f"{target_description}.{attr_name} carries tag {dynamic_tag!r}, "
            f"expected {expected_tag!r} (clobbered by sibling patch or "
            f"vLLM import side-effect)."
        )
    if static_tag != expected_tag:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module_name!r}: "
            f"inspect.getattr_static sees {static_tag!r} on "
            f"{target_description}.{attr_name}, expected {expected_tag!r} "
            f"(metaclass shim shadowing install)."
        )

    if getattr(installed_dynamic, "__wrapped_original__", None) is None:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module_name!r}: "
            f"{target_description}.{attr_name} bears tag but lacks "
            f"__wrapped_original__ — forged tag or patch rewritten."
        )


def _verify_qwen3_coder(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_qwen3_coder`` replaced the parser.

    1. Tag presence on ``_parse_xml_function_call`` (cheapest signal).
    2. Behavioral: a truncated input ``"calculator>\\n<parameter=a"``
       must return ``None`` (not raise ``ValueError``). Defends against
       the tag-present-but-function-reverted regression.
    3. Negative control: a well-formed input must return a non-None
       ToolCall, so we know the None-on-truncation is not parser-broken.
    """
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import behavioral-check deps: {exc!r}"
        ) from exc
    _verify_target_carries_tag(
        Qwen3CoderToolParser,
        "_parse_xml_function_call",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="Qwen3CoderToolParser",
    )

    # Decoupled from the served model: ``Qwen3CoderToolParser.__init__``
    # only requires a truthy ``model_tokenizer`` whose ``get_vocab()``
    # returns non-None ids for ``<tool_call>``/``</tool_call>``.
    class _TokenizerMock:
        _vocab = {"<tool_call>": 1_000_001, "</tool_call>": 1_000_002}

        def get_vocab(self) -> dict[str, int]:
            return self._vocab

        def __bool__(self) -> bool:
            return True

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
    parser = Qwen3CoderToolParser(_TokenizerMock(), tools=[tool_calc])

    truncated = "calculator>\n<parameter=a"
    try:
        result = parser._parse_xml_function_call(truncated)
    except Exception as exc:  # noqa: BLE001 — any raise is a hard refusal
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module.__name__!r} FAILED: "
            f"_parse_xml_function_call({truncated!r}) raised "
            f"{type(exc).__name__}: {exc!r}; expected None."
        ) from exc
    if result is not None:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module.__name__!r} FAILED: "
            f"_parse_xml_function_call({truncated!r}) returned {result!r}; expected None."
        )

    well_formed = (
        "calculator>\n<parameter=a>\n2\n</parameter>\n"
        "<parameter=b>\n3\n</parameter>\n"
    )
    try:
        good = parser._parse_xml_function_call(well_formed)
    except Exception as exc:  # noqa: BLE001
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module.__name__!r} negative control "
            f"FAILED: well-formed input raised {type(exc).__name__}: {exc!r}."
        ) from exc
    if good is None:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module.__name__!r} negative control "
            f"FAILED: well-formed input returned None."
        )


def _verify_hybrid_kv_allocator(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_hybrid_kv_allocator`` replaced both targets
    AND the replacements use the filtered-groups divisor (1 attn group)
    instead of the unpatched ``len(kv_cache_groups)`` (4 groups).

    1. Tag presence on both targets.
    2. Behavioral: synthetic Qwen3.6-shape ``KVCacheConfig`` (1 attn +
       3 Mamba groups in mode ``"none"``); the patched concurrency value
       must match the divisor=1 calculation, not divisor=4. Discriminating
       term is the SUM of ``max_memory_usage_bytes``: patched sums only
       attn, unpatched sums attn + 3*mamba_O1.
    3. Cosmetic: ``_report_kv_cache_config`` must not raise on the same
       valid synthetic config.
    """
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.v1.core import kv_cache_utils as _kv_cache_utils_mod
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            MambaSpec,
        )
        from vllm.utils.math_utils import cdiv
        import torch
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import behavioral-check deps: {exc!r}"
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

    # 1 attn vs 4 total = 4× ratio, unambiguous discriminator.
    block_size = 16
    num_full_attn_layers = num_mamba_layers = 16
    huge = 256 * 1024 * 1024  # per-Mamba padded bytes — escape rounding

    full_attn_spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=4, head_size=256, dtype=torch.bfloat16
    )
    mamba_specs: list[MambaSpec] = [
        MambaSpec(
            block_size=block_size,
            shapes=shapes,
            dtypes=(torch.bfloat16,) * len(shapes),
            page_size_padded=huge,
            mamba_cache_mode="none",
        )
        for shapes in (((128, 128),), ((256,),), ((1024,),))
    ]
    synthetic_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"attn.{i}" for i in range(num_full_attn_layers)],
                kv_cache_spec=full_attn_spec,
            ),
            *(
                KVCacheGroupSpec(
                    layer_names=[f"mamba.{idx}.{i}" for i in range(num_mamba_layers)],
                    kv_cache_spec=spec,
                )
                for idx, spec in enumerate(mamba_specs)
            ),
        ],
    )

    # Stub VllmConfig — only the sub-attributes the patch reads.
    from types import SimpleNamespace

    stub_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none", num_gpu_blocks_override=None),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        model_config=SimpleNamespace(max_model_len=65536),
        scheduler_config=SimpleNamespace(max_num_encoder_input_tokens=0),
    )

    # Expected patched concurrency: divisor=1, only attn contributes.
    full_attn_bytes = full_attn_spec.max_memory_usage_bytes(stub_vllm_config)
    page = full_attn_spec.page_size_bytes
    expected = synthetic_config.num_blocks / cdiv(
        num_full_attn_layers * full_attn_bytes, page * num_full_attn_layers
    )

    actual = _kv_cache_utils_mod.get_max_concurrency_for_kv_cache_config(
        stub_vllm_config, synthetic_config
    )
    if not isinstance(actual, (int, float)):
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module.__name__!r}: get_max_concurrency "
            f"returned {actual!r} (type {type(actual).__name__})."
        )
    if abs(actual - expected) > 1e-6 * max(abs(expected), 1.0):
        # Compute unpatched expectation for the diagnostic.
        mamba_total = sum(
            s.max_memory_usage_bytes(stub_vllm_config) for s in mamba_specs
        )
        unpatched = synthetic_config.num_blocks / cdiv(
            num_full_attn_layers * (full_attn_bytes + mamba_total),
            page * num_full_attn_layers,
        )
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module.__name__!r} FAILED: "
            f"get_max_concurrency returned {actual!r}, expected patched "
            f"{expected!r} (unpatched ~{unpatched!r}). Function body reverted."
        )

    # Cosmetic: _report_kv_cache_config must not raise on valid input.
    try:
        _kv_cache_utils_mod._report_kv_cache_config(stub_vllm_config, synthetic_config)
    except Exception as exc:  # noqa: BLE001
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module.__name__!r}: _report_kv_cache_config "
            f"raised {type(exc).__name__}: {exc!r} on valid synthetic config."
        ) from exc


def _verify_reasoning_field_egress(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_reasoning_field_egress`` stamped its tag on
    each of the six target classes.

    Tag-only check: the patch's own Phase 3 self-verification constructs
    real ``ChatCompletionResponse`` and ``ChatCompletionStreamResponse``
    instances, dumps them via ``model_dump_json()`` and
    ``model_dump_json(exclude_unset=True)``, and asserts the wire bytes
    contain ``"reasoning_content":`` and not ``"reasoning":``. Re-doing
    that check here would duplicate the patch's own load-bearing
    verification — the launcher need only confirm the install
    propagated to every target via both attribute lookup and
    ``inspect.getattr_static``.
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
        from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import egress patch's six target "
            f"classes for verification: {exc!r}"
        ) from exc

    targets: tuple[tuple[type, str], ...] = (
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
    for target_cls, target_name in targets:
        dynamic = getattr(target_cls, "__qwen36_egress_patch__", None)
        if dynamic != expected_tag:
            raise PatchVerificationError(
                f"[{_LAUNCHER_TAG}] tag verification failed for "
                f"{patch_module.__name__!r}: {target_name} carries "
                f"__qwen36_egress_patch__={dynamic!r}, expected "
                f"{expected_tag!r}."
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
                f"{expected_tag!r} (metaclass shim shadowing install)."
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


def _verify_tool_call_in_think_detector(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_tool_call_in_think_detector`` wrapped the
    non-streaming ``extract_reasoning`` method on ``Qwen3ReasoningParser``.
    Streaming is intentionally unwrapped (the metric is a model-side
    property, not per-modality) — only ``extract_reasoning`` carries
    the tag.
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


def _verify_default_sampling_params(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_default_sampling_params`` wrapped
    ``ChatCompletionRequest.to_sampling_params``.

    Tag-only check: the patch's own Phase 7 self-verification constructs
    real ``ChatCompletionRequest`` instances, calls the wrapped method
    in five behavioural cases (no fields set, explicit temperature,
    small max_tokens cap, explicit presence_penalty=0.0, explicit
    max_completion_tokens) and asserts every Qwen3.6 default is applied
    iff the client did not explicitly send the field. Re-doing that
    here would duplicate the patch's own load-bearing verification —
    the launcher need only confirm the install propagated to the target
    via both attribute lookup and ``inspect.getattr_static``.
    """
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import "
            f"vllm.entrypoints.openai.chat_completion.protocol."
            f"ChatCompletionRequest for verification: {exc!r}"
        ) from exc
    _verify_target_carries_tag(
        ChatCompletionRequest,
        "to_sampling_params",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="ChatCompletionRequest",
    )


def _verify_qwen3_coder_grammar(patch_module: ModuleType) -> None:
    """Verify ``monkey_patch_qwen3_coder_grammar`` overrode
    ``Qwen3CoderToolParser.adjust_request`` and flipped
    ``supports_required_and_named=False``.

    Tag-only check: the patch's own Phase 9 self-verification constructs
    real ``ChatCompletionRequest`` instances, instantiates
    ``Qwen3CoderToolParser`` against a minimal model_tokenizer surface,
    calls the wrapped method in four behavioural cases (auto+tools,
    empty tools, explicit structured_outputs, tool_choice='none'),
    parses the structural_tag JSON, and round-trips it through
    ``xgrammar.Grammar.from_structural_tag``. Re-doing that here would
    duplicate the patch's own load-bearing verification — the launcher
    need only confirm the install propagated to the target via both
    attribute lookup and ``inspect.getattr_static``.
    """
    expected_tag = _expected_tag_from(patch_module)
    try:
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
    except ImportError as exc:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] cannot import "
            f"vllm.tool_parsers.qwen3coder_tool_parser.Qwen3CoderToolParser "
            f"for verification: {exc!r}"
        ) from exc
    _verify_target_carries_tag(
        Qwen3CoderToolParser,
        "adjust_request",
        expected_tag,
        patch_module_name=patch_module.__name__,
        target_description="Qwen3CoderToolParser",
    )
    if Qwen3CoderToolParser.supports_required_and_named is not False:
        raise PatchVerificationError(
            f"[{_LAUNCHER_TAG}] {patch_module.__name__!r}: "
            f"Qwen3CoderToolParser.supports_required_and_named is "
            f"{Qwen3CoderToolParser.supports_required_and_named!r}, "
            f"expected False (latent bug fix at engine/serving.py:646-665)."
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
    # Order notes: qwen3_coder owns the tool-parser surface; hybrid_kv
    # touches a disjoint kv_cache_utils surface; reasoning_field_egress
    # rebuilds DeltaMessage's Pydantic schema and so should come before
    # any patch that constructs DeltaMessage at request time;
    # reasoning_field_ingest wraps a chat_utils function (independent);
    # tool_call_in_think_detector wraps Qwen3ReasoningParser
    # extract_reasoning (independent of the egress rebuild);
    # default_sampling_params wraps ChatCompletionRequest.to_sampling_params
    # (independent of every other surface — request-time field rewrite
    # only); qwen3_coder_grammar overrides Qwen3CoderToolParser.adjust_request
    # (must come AFTER qwen3_coder — patch 1 only touches the unrelated
    # _parse_xml_function_call method on the same class — but ordering
    # is defensive: a future regression where one clobbers the other
    # would be visible to the launcher's tag verifier).
    "monkey_patch_qwen3_coder",
    "monkey_patch_hybrid_kv_allocator",
    "monkey_patch_reasoning_field_egress",
    "monkey_patch_reasoning_field_ingest",
    "monkey_patch_tool_call_in_think_detector",
    "monkey_patch_default_sampling_params",
    "monkey_patch_qwen3_coder_grammar",
)

_PATCH_VERIFICATION: dict[str, _PatchVerifier] = {
    "monkey_patch_qwen3_coder": _verify_qwen3_coder,
    "monkey_patch_hybrid_kv_allocator": _verify_hybrid_kv_allocator,
    "monkey_patch_reasoning_field_egress": _verify_reasoning_field_egress,
    "monkey_patch_reasoning_field_ingest": _verify_reasoning_field_ingest,
    "monkey_patch_tool_call_in_think_detector": _verify_tool_call_in_think_detector,
    "monkey_patch_default_sampling_params": _verify_default_sampling_params,
    "monkey_patch_qwen3_coder_grammar": _verify_qwen3_coder_grammar,
}


# --------------------------------------------------------------------
# Pre-flight checks (run BEFORE the per-patch import loop).
# --------------------------------------------------------------------
#
# These three checks defend against the spawn-vs-fork class of failure
# (sitecustomize missing, registry drift between launcher and
# sitecustomize, or a subprocess-only deviation that hides
# ``monkey_patch_hybrid_kv_allocator``'s install from EngineCore — that
# is the only patch in the suite whose targets run in the spawned
# subprocess, README §7.3). Each check refuses with a typed
# :class:`LauncherError` on failure; none of them log anything in the
# happy path.


_SITECUSTOMIZE_MISSING_MSG: str = (
    "sitecustomize.py is not the qwen36-agent-setup version (or is not "
    "importable). Most likely cause: the docker run command is missing "
    "`-v \"$PWD/sitecustomize.py:/opt/patches/sitecustomize.py:ro\"`. "
    "Without our sitecustomize, vLLM v1's spawned EngineCore subprocess "
    "won't re-install the patches and monkey_patch_hybrid_kv_allocator "
    "is silently dead in EngineCore. See sitecustomize.py docstring and "
    "README §7.S."
)


def _preflight_sitecustomize_present() -> ModuleType:
    """Refuse to boot if our ``sitecustomize`` is not loaded.

    Debian's CPython ships a stub ``/usr/lib/python3.12/sitecustomize.py``,
    so a bare ``import sitecustomize`` succeeds even when our bind-mount
    is absent. Discriminate by checking for ``_PATCH_MODULES`` (only OUR
    sitecustomize defines it) and the install-completion sentinel
    ``builtins._qwen36_sitecustomize_loaded``.
    """
    try:
        import sitecustomize  # noqa: F401  -- imported for side-effects
    except ImportError as exc:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] sitecustomize not importable "
            f"({sys.executable!r}, sys.path={sys.path!r}). "
            f"{_SITECUSTOMIZE_MISSING_MSG}"
        ) from exc

    if not hasattr(sitecustomize, "_PATCH_MODULES"):
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] sitecustomize at {sitecustomize.__file__!r} "
            f"does NOT define `_PATCH_MODULES` — system-default stub, not "
            f"ours. {_SITECUSTOMIZE_MISSING_MSG}"
        )

    if not getattr(__import__("builtins"), "_qwen36_sitecustomize_loaded", False):
        # Our sitecustomize was imported but its install never fired —
        # forged sentinel or edited install loop.
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] sitecustomize at {sitecustomize.__file__!r} "
            f"exposes _PATCH_MODULES but the install sentinel "
            f"`builtins._qwen36_sitecustomize_loaded` was not set. "
            f"Audit sitecustomize.py."
        )
    return sitecustomize


def _preflight_registry_drift_check(sitecustomize_module: ModuleType) -> None:
    """Refuse to boot if ``sitecustomize._PATCH_MODULES`` and
    :data:`_PATCH_MODULES` disagree. Drift means EngineCore (via
    sitecustomize) and the API server (via the launcher) are running
    different patch sets — silent in boot logs, visible only at request
    time as cross-process behaviour disagreement.
    """
    sm = getattr(sitecustomize_module, "_PATCH_MODULES", None)
    if not isinstance(sm, tuple) or not all(isinstance(m, str) for m in sm):
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] sitecustomize._PATCH_MODULES is not a tuple "
            f"of strings; got {type(sm).__name__}: {sm!r}."
        )
    if sm != _PATCH_MODULES:
        only_sc = [m for m in sm if m not in _PATCH_MODULES]
        only_launcher = [m for m in _PATCH_MODULES if m not in sm]
        order_drift = sm != _PATCH_MODULES and not only_sc and not only_launcher
        diff_lines: list[str] = []
        if only_sc:
            diff_lines.append(f"  only in sitecustomize: {only_sc!r}")
        if only_launcher:
            diff_lines.append(f"  only in launcher: {only_launcher!r}")
        if order_drift:
            diff_lines.append(f"  same set, different order: sc={sm!r}, launcher={_PATCH_MODULES!r}")
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] registry drift between sitecustomize._PATCH_MODULES "
            f"and launch_with_patches._PATCH_MODULES:\n"
            + "\n".join(diff_lines)
            + "\nBoth tuples must be byte-identical."
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
    ("monkey_patch_reasoning_field_ingest",
     "vllm.entrypoints.chat_utils",
     None,
     "_parse_chat_message_content"),
    ("monkey_patch_tool_call_in_think_detector",
     "vllm.reasoning.qwen3_reasoning_parser",
     "Qwen3ReasoningParser",
     "extract_reasoning"),
    ("monkey_patch_default_sampling_params",
     "vllm.entrypoints.openai.chat_completion.protocol",
     "ChatCompletionRequest",
     "to_sampling_params"),
    ("monkey_patch_qwen3_coder_grammar",
     "vllm.tool_parsers.qwen3coder_tool_parser",
     "Qwen3CoderToolParser",
     "adjust_request"),
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
    every patch tag is present in the child. Defends against the three
    failure modes the in-process check cannot detect: sitecustomize
    importable in PID 1 but not in the child (PYTHONPATH/sys.path
    drift), child interpreter differs from PID 1's (venv/sys.executable),
    or a subprocess-only install regression.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_INSTALL_PROBE],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess install check exited "
            f"non-zero ({proc.returncode}); spawn child cannot install "
            f"the patches.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    output_lines = proc.stdout.strip().splitlines()
    exec_line = next((l for l in output_lines if l.startswith("EXEC ")), None)
    target_lines = [
        l for l in output_lines if l.startswith("OK ") or l.startswith("FAIL ")
    ]
    if exec_line is None:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess probe produced no EXEC line.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    child_executable = exec_line[len("EXEC "):].strip()
    if child_executable != sys.executable:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess probe ran under "
            f"{child_executable!r} != launcher's {sys.executable!r}. "
            f"Check `--entrypoint python3`."
        )
    failures = [l for l in target_lines if l.startswith("FAIL ")]
    if failures:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess probe found {len(failures)} "
            f"target(s) the spawn child could not verify:\n  "
            + "\n  ".join(failures)
            + f"\nstderr:\n{proc.stderr}"
        )
    untagged = [
        l for l in target_lines if l.startswith("OK ") and l.endswith("None")
    ]
    if untagged:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] subprocess probe found {len(untagged)} "
            f"target(s) without the expected __qwen36_patch__ tag:\n  "
            + "\n  ".join(untagged)
        )


# --------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------


def _import_and_verify(module_name: str) -> ModuleType:
    """Import a patch module and run its verifier. The patch's own typed
    install exception and the launcher's PatchVerificationError both
    propagate unchanged so the container exits non-zero with the
    specific refusal message intact."""
    if module_name not in _PATCH_VERIFICATION:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] no verifier registered for {module_name!r}. "
            f"Add an entry to _PATCH_VERIFICATION."
        )
    patch_module: ModuleType = importlib.import_module(module_name)
    _PATCH_VERIFICATION[module_name](patch_module)
    return patch_module


def _handoff_to_vllm_cli(argv: list[str]) -> None:
    """Hand off to vLLM's CLI exactly as ``vllm <argv>`` would. Munges
    ``sys.argv`` so argv[0] is the canonical ``"vllm"`` string, then
    runs ``vllm.entrypoints.cli.main`` under ``__main__``. Does NOT
    catch exceptions from the runpy'd module — vLLM's tracebacks must
    surface as the user's diagnostic.
    """
    sys.argv = [_VLLM_ARGV0, *argv]
    runpy.run_module(_VLLM_CLI_MODULE, run_name="__main__", alter_sys=True)


def _usage_error_and_exit() -> None:
    """Refuse a bare invocation. A missing subcommand is a deployment
    misconfiguration; printing help and returning 0 would let a process
    supervisor read the container as "served and finished".
    """
    sys.stderr.write(
        f"[{_LAUNCHER_TAG}] usage: python launch_with_patches.py "
        f"<vllm-subcommand> [args...]\n"
        f"  e.g. python launch_with_patches.py serve --model ... "
        f"--host 127.0.0.1 --port 8000\n"
    )
    raise SystemExit(2)


def main(argv: Iterable[str] | None = None) -> None:
    """Install registered patches in order, verify each, then exec vLLM."""
    args: list[str] = list(argv if argv is not None else sys.argv[1:])
    if not args:
        _usage_error_and_exit()

    # Surface a missing vLLM with a launcher-specific message before
    # the first patch import dies with a confusing ImportError.
    try:
        importlib.import_module("vllm")
    except ImportError as exc:
        raise LauncherError(
            f"[{_LAUNCHER_TAG}] vllm is not importable from "
            f"{sys.executable!r}. Check the container image and PYTHONPATH."
        ) from exc

    # Pre-flight: refuse to boot if sitecustomize is missing or the
    # registry has drifted. These run BEFORE the per-patch import loop
    # — sitecustomize already installed the patches if importable, so
    # the loop below is a verifier pass.
    sitecustomize_module = _preflight_sitecustomize_present()
    _preflight_registry_drift_check(sitecustomize_module)
    _preflight_subprocess_install_check()

    # Strict ordered install + verify. Hold a reference to every
    # imported patch module for the lifetime of the process so the
    # install side-effects cannot be undone by GC.
    installed: list[ModuleType] = [
        _import_and_verify(name) for name in _PATCH_MODULES
    ]
    globals()["_INSTALLED_PATCHES"] = tuple(installed)

    _handoff_to_vllm_cli(args)


if __name__ == "__main__":
    main()
