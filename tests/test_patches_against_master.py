"""Static + structural-mirror verification that the 5 monkey-patches
will apply cleanly against the pinned master commit. Does NOT import
vLLM (host has no torch/CUDA/Triton). Walks each patch's landmark
constants against master source, AST-extracts function signatures, and
runs a real Pydantic v2 experiment on a structural mirror of vLLM's
wire wrappers (Section 3b) to prove the egress patch's mechanism. The
pinned commit must match HEAD of /tmp/qwen36_research/vllm. Exits 0
iff every check passes.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

PATCH_DIR = Path("/home/user/Desktop/qwen_36_agent_setup")
VLLM_DIR = Path("/tmp/qwen36_research/vllm")
EXPECTED_COMMIT = "8cd174fa358326d5cc4195446be2ebcd65c481ce"

# ANSI color codes for output. Only output to a tty would render these,
# but they are harmless in plain text.
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# --------------------------------------------------------------------
# Test reporting
# --------------------------------------------------------------------


class TestRun:
    """Accumulates test outcomes; non-zero exit code iff any fail."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def expect(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed.append(name)
            print(f"  {_GREEN}PASS{_RESET} {name}")
        else:
            self.failed.append((name, detail))
            print(f"  {_RED}FAIL{_RESET} {name}")
            if detail:
                # Indent failure detail.
                for line in detail.splitlines():
                    print(f"       {line}")

    def expect_eq(self, name: str, actual: Any, expected: Any) -> None:
        ok = actual == expected
        self.expect(
            name,
            ok,
            "" if ok else f"expected: {expected!r}\nactual:   {actual!r}",
        )

    def expect_in(self, name: str, needle: str, haystack: str) -> None:
        ok = needle in haystack
        self.expect(
            name,
            ok,
            ""
            if ok
            else f"needle: {needle!r}\nhaystack starts with: {haystack[:200]!r}",
        )

    def expect_not_in(self, name: str, needle: str, haystack: str) -> None:
        ok = needle not in haystack
        self.expect(
            name,
            ok,
            "" if ok else f"unexpectedly present: {needle!r}",
        )

    def section(self, title: str) -> None:
        print(f"\n{_BOLD}{title}{_RESET}")

    def report(self) -> int:
        print()
        if self.failed:
            print(
                f"{_BOLD}{_RED}FAILED{_RESET}: "
                f"{len(self.passed)} passed, {len(self.failed)} failed"
            )
            for name, detail in self.failed:
                print(f"  - {name}")
            return 1
        print(
            f"{_BOLD}{_GREEN}OK{_RESET}: "
            f"{len(self.passed)} passed, 0 failed"
        )
        return 0


run = TestRun()


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def read_text(path: Path) -> str:
    return path.read_text()


def patch_constant(patch_file: Path, name: str) -> Any:
    """Extract a top-level string/tuple constant from a patch file via AST.

    We can't import the patches (they import vLLM at module-import
    time), so AST-parse the source and pull the literal value of a
    named assignment.

    Supports str literals, tuples of str literals, and tuples of tuples
    of literals (for the multi-line landmarks like
    ``_CONCURRENCY_BUGGY_LANDMARK`` which is a single str spread across
    multiple lines via implicit concatenation).
    """
    tree = ast.parse(patch_file.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
            continue
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Name)
                and target.id == name
                and node.value is not None
            ):
                return ast.literal_eval(node.value)
    raise KeyError(f"constant {name!r} not found in {patch_file.name}")


def vllm_function_signature(
    relative_source_path: str,
    qualified_name: str,
) -> list[str]:
    """Extract a function/method's parameter name list via AST.

    ``qualified_name`` may be ``func`` (top-level) or
    ``ClassName.method`` (one-level nesting; arbitrary depth would
    require tweaking but we don't need it).
    """
    src = (VLLM_DIR / relative_source_path).read_text()
    tree = ast.parse(src)

    def _params(fn: ast.FunctionDef) -> list[str]:
        return [a.arg for a in fn.args.args]

    if "." not in qualified_name:
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == qualified_name
            ):
                return _params(node)
        raise KeyError(
            f"top-level function {qualified_name!r} not found in "
            f"{relative_source_path}"
        )

    cls_name, method_name = qualified_name.split(".", 1)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if (
                    isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and sub.name == method_name
                ):
                    return _params(sub)
    raise KeyError(
        f"method {qualified_name!r} not found in {relative_source_path}"
    )


# --------------------------------------------------------------------
# Section 0: Pinned commit consistency
# --------------------------------------------------------------------


def section_0_pin_consistency() -> None:
    run.section("0. Pinned commit consistency")

    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(VLLM_DIR),
        text=True,
    ).strip()
    run.expect_eq(
        "VLLM repo HEAD == EXPECTED_COMMIT", head_sha, EXPECTED_COMMIT
    )

    patch_files = sorted(PATCH_DIR.glob("monkey_patch_*.py"))
    for path in patch_files:
        commit = patch_constant(path, "_PINNED_VLLM_COMMIT")
        run.expect_eq(
            f"{path.name}._PINNED_VLLM_COMMIT == {EXPECTED_COMMIT[:12]}…",
            commit,
            EXPECTED_COMMIT,
        )


# --------------------------------------------------------------------
# Section 1: Patch 1 — qwen3_coder
# --------------------------------------------------------------------


def section_1_qwen3_coder() -> None:
    run.section("1. monkey_patch_qwen3_coder.py")
    patch_path = PATCH_DIR / "monkey_patch_qwen3_coder.py"
    qwen3_src = (VLLM_DIR / "vllm/tool_parsers/qwen3coder_tool_parser.py").read_text()
    abstract_src = (VLLM_DIR / "vllm/tool_parsers/abstract_tool_parser.py").read_text()
    engine_src = (VLLM_DIR / "vllm/entrypoints/openai/engine/protocol.py").read_text()

    # The buggy str.index(">") landmark; the qwen3-specific regex SHAPE in
    # the subclass; and the regex-attribute assignment in the subclass.
    run.expect_in(
        "buggy sentinel still present at master",
        patch_constant(patch_path, "_BUGGY_SENTINEL"),
        qwen3_src,
    )
    run.expect_in(
        "tool_call_parameter_regex shape unchanged in subclass __init__",
        patch_constant(patch_path, "_UPSTREAM_REGEX_LANDMARK"),
        qwen3_src,
    )
    expected_attr = patch_constant(patch_path, "_EXPECTED_REGEX_ATTR")
    run.expect(
        f"Qwen3CoderToolParser subclass __init__ assigns self.{expected_attr}",
        re.search(rf"\bself\.{re.escape(expected_attr)}\s*[:=]", qwen3_src) is not None,
    )
    # `self.tools` may be assigned by the subclass OR inherited from
    # `ToolParser.__init__`; the patch's MRO walk requires presence anywhere.
    run.expect(
        "self.tools assigned somewhere in the Qwen3CoderToolParser MRO",
        re.search(r"\bself\.tools\s*[:=]", qwen3_src + "\n" + abstract_src) is not None,
    )

    run.expect_eq(
        "_parse_xml_function_call signature",
        vllm_function_signature(
            "vllm/tool_parsers/qwen3coder_tool_parser.py",
            "Qwen3CoderToolParser._parse_xml_function_call",
        ),
        ["self", "function_call_str"],
    )
    run.expect_eq(
        "_convert_param_value signature",
        vllm_function_signature(
            "vllm/tool_parsers/qwen3coder_tool_parser.py",
            "Qwen3CoderToolParser._convert_param_value",
        ),
        ["self", "param_value", "param_name", "param_config", "func_name"],
    )
    run.expect_eq(
        "find_tool_properties signature",
        vllm_function_signature("vllm/tool_parsers/utils.py", "find_tool_properties"),
        ["tools", "tool_name"],
    )
    run.expect_in(
        "engine.protocol exports class FunctionCall", "class FunctionCall(", engine_src
    )
    run.expect_in("engine.protocol exports class ToolCall", "class ToolCall(", engine_src)


# --------------------------------------------------------------------
# Section 2: Patch 2 — hybrid_kv_allocator
# --------------------------------------------------------------------


def section_2_hybrid_kv() -> None:
    run.section("2. monkey_patch_hybrid_kv_allocator.py")
    patch_path = PATCH_DIR / "monkey_patch_hybrid_kv_allocator.py"
    target_src = (VLLM_DIR / "vllm/v1/core/kv_cache_utils.py").read_text()
    interface_src = (VLLM_DIR / "vllm/v1/kv_cache_interface.py").read_text()
    cache_cfg_src = (VLLM_DIR / "vllm/config/cache.py").read_text()

    # Multi-line landmarks.
    concurrency_landmark = patch_constant(patch_path, "_CONCURRENCY_BUGGY_LANDMARK")
    report_landmark = patch_constant(patch_path, "_REPORT_BUGGY_LANDMARK")
    run.expect_in(
        "concurrency buggy landmark present at master",
        concurrency_landmark,
        target_src,
    )
    run.expect_in(
        "report buggy landmark present at master",
        report_landmark,
        target_src,
    )

    # Signatures.
    gmc_sig = vllm_function_signature(
        "vllm/v1/core/kv_cache_utils.py",
        "get_max_concurrency_for_kv_cache_config",
    )
    run.expect_eq(
        "get_max_concurrency_for_kv_cache_config signature",
        gmc_sig,
        ["vllm_config", "kv_cache_config"],
    )
    rkc_sig = vllm_function_signature(
        "vllm/v1/core/kv_cache_utils.py",
        "_report_kv_cache_config",
    )
    run.expect_eq(
        "_report_kv_cache_config signature",
        rkc_sig,
        ["vllm_config", "kv_cache_config"],
    )

    # Coordinated patching: _report calls get_max_concurrency.
    run.expect_in(
        "_report_kv_cache_config still calls get_max_concurrency_…",
        "get_max_concurrency_for_kv_cache_config",
        target_src,
    )

    # MambaSpec.max_memory_usage_bytes signature.
    mamba_sig = vllm_function_signature(
        "vllm/v1/kv_cache_interface.py",
        "MambaSpec.max_memory_usage_bytes",
    )
    run.expect_eq(
        "MambaSpec.max_memory_usage_bytes signature",
        mamba_sig,
        ["self", "vllm_config"],
    )

    # FullAttentionSpec.max_memory_usage_bytes signature.
    full_sig = vllm_function_signature(
        "vllm/v1/kv_cache_interface.py",
        "FullAttentionSpec.max_memory_usage_bytes",
    )
    run.expect_eq(
        "FullAttentionSpec.max_memory_usage_bytes signature",
        full_sig,
        ["self", "vllm_config"],
    )

    # MambaCacheMode definition.
    run.expect_in(
        'MambaCacheMode = Literal["all", "align", "none"]',
        'MambaCacheMode = Literal["all", "align", "none"]',
        cache_cfg_src,
    )


# --------------------------------------------------------------------
# Section 3: Patch 3 — reasoning_field_egress (the redesigned one)
# --------------------------------------------------------------------


def section_3_egress_static() -> None:
    run.section("3a. monkey_patch_reasoning_field_egress.py — static")
    chat_src = (VLLM_DIR / "vllm/entrypoints/openai/chat_completion/protocol.py").read_text()
    engine_src = (VLLM_DIR / "vllm/entrypoints/openai/engine/protocol.py").read_text()

    for cls_name in (
        "ChatMessage",
        "ChatCompletionResponseChoice",
        "ChatCompletionResponse",
        "ChatCompletionResponseStreamChoice",
        "ChatCompletionStreamResponse",
    ):
        run.expect_in(
            f"chat_completion.protocol exports {cls_name}",
            f"class {cls_name}(",
            chat_src,
        )
    run.expect_in("engine.protocol exports DeltaMessage", "class DeltaMessage(", engine_src)

    leaf_re = re.compile(r"reasoning:\s*str\s*\|\s*None\s*=\s*None")
    run.expect("ChatMessage has reasoning: str | None = None", leaf_re.search(chat_src) is not None)
    run.expect(
        "DeltaMessage has reasoning: str | None = None",
        leaf_re.search(engine_src) is not None,
    )
    run.expect_not_in(
        "ChatMessage does not already declare reasoning_content", "reasoning_content:", chat_src
    )
    run.expect_not_in(
        "DeltaMessage does not already declare reasoning_content",
        "reasoning_content:",
        engine_src,
    )

    # Wrappers' nested-field shape.
    for label, needle in (
        ("ChatCompletionResponseChoice has 'message: ChatMessage'", "message: ChatMessage"),
        (
            "ChatCompletionResponseStreamChoice has 'delta: DeltaMessage'",
            "delta: DeltaMessage",
        ),
        (
            "ChatCompletionResponse has 'choices: list[ChatCompletionResponseChoice]'",
            "choices: list[ChatCompletionResponseChoice]",
        ),
        (
            "ChatCompletionStreamResponse has 'choices: list[ChatCompletionResponseStreamChoice]'",
            "choices: list[ChatCompletionResponseStreamChoice]",
        ),
    ):
        run.expect_in(label, needle, chat_src)


def section_3_egress_mechanism() -> None:
    """Real Pydantic v2 reproduction on a structural mirror of vLLM's
    wire wrappers. Verifies the patch's mechanism (alias on leaves +
    flag+rebuild on every level of the chain) produces the desired
    wire JSON for both standalone and nested dumps.
    """
    run.section("3b. monkey_patch_reasoning_field_egress.py — mechanism")

    from pydantic import BaseModel, ConfigDict, Field

    # Mirror the vLLM hierarchy:
    #
    #   OpenAIBaseModel              (extra="allow")
    #    ├── ChatMessage              {role, content, reasoning, tool_calls}
    #    ├── DeltaMessage             {role, content, reasoning, tool_calls}
    #    ├── ChatCompletionResponseChoice  {index, message: ChatMessage}
    #    ├── ChatCompletionResponseStreamChoice  {index, delta: DeltaMessage}
    #    ├── ChatCompletionResponse        {choices: list[…ResponseChoice]}
    #    └── ChatCompletionStreamResponse  {choices: list[…StreamChoice]}

    class OpenAIBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")

    class ChatMessageMirror(OpenAIBaseModel):
        role: str
        content: str | None = None
        reasoning: str | None = None
        tool_calls: list[Any] = Field(default_factory=list)

    class DeltaMessageMirror(OpenAIBaseModel):
        role: str | None = None
        content: str | None = None
        reasoning: str | None = None
        tool_calls: list[Any] = Field(default_factory=list)

    class ResponseChoiceMirror(OpenAIBaseModel):
        index: int
        message: ChatMessageMirror

    class StreamChoiceMirror(OpenAIBaseModel):
        index: int
        delta: DeltaMessageMirror

    class ResponseMirror(OpenAIBaseModel):
        id: str
        model: str
        choices: list[ResponseChoiceMirror]

    class StreamResponseMirror(OpenAIBaseModel):
        id: str
        model: str
        choices: list[StreamChoiceMirror]

    # Pre-patch sanity: nested dump emits "reasoning" not "reasoning_content".
    pre = ResponseMirror(
        id="x",
        model="m",
        choices=[
            ResponseChoiceMirror(
                index=0,
                message=ChatMessageMirror(role="assistant", reasoning="probe"),
            )
        ],
    )
    pre_wire = pre.model_dump_json()
    run.expect_in(
        "PRE-PATCH: nested wire emits 'reasoning' (sanity)",
        '"reasoning":"probe"',
        pre_wire,
    )

    # Apply the patch's mechanism in dependency order: leaves first,
    # then intermediates, then outermost.
    def install_leaf(cls: type) -> None:
        cls.model_fields["reasoning"].serialization_alias = "reasoning_content"
        cls.model_config["serialize_by_alias"] = True
        for attr in (
            "__pydantic_core_schema__",
            "__pydantic_validator__",
            "__pydantic_serializer__",
        ):
            if attr in cls.__dict__:
                delattr(cls, attr)
        cls.__pydantic_complete__ = False
        cls.model_rebuild(force=True)

    def install_wrapper(cls: type) -> None:
        cls.model_config["serialize_by_alias"] = True
        for attr in (
            "__pydantic_core_schema__",
            "__pydantic_validator__",
            "__pydantic_serializer__",
        ):
            if attr in cls.__dict__:
                delattr(cls, attr)
        cls.__pydantic_complete__ = False
        cls.model_rebuild(force=True)

    install_leaf(ChatMessageMirror)
    install_leaf(DeltaMessageMirror)
    install_wrapper(ResponseChoiceMirror)
    install_wrapper(StreamChoiceMirror)
    install_wrapper(ResponseMirror)
    install_wrapper(StreamResponseMirror)

    # Post-patch: non-streaming nested dump emits reasoning_content.
    post = ResponseMirror(
        id="x",
        model="m",
        choices=[
            ResponseChoiceMirror(
                index=0,
                message=ChatMessageMirror(role="assistant", reasoning="probe"),
            )
        ],
    )
    post_wire = post.model_dump_json()
    run.expect_in(
        "POST-PATCH: nested non-streaming wire emits reasoning_content",
        '"reasoning_content":"probe"',
        post_wire,
    )
    run.expect_not_in(
        "POST-PATCH: nested non-streaming wire has NO bare reasoning",
        '"reasoning":',
        post_wire,
    )

    # Post-patch: streaming nested dump (exclude_unset=True) emits
    # reasoning_content.
    post_stream = StreamResponseMirror(
        id="x",
        model="m",
        choices=[
            StreamChoiceMirror(
                index=0,
                delta=DeltaMessageMirror(reasoning="probe"),
            )
        ],
    )
    post_stream_wire = post_stream.model_dump_json(exclude_unset=True)
    run.expect_in(
        "POST-PATCH: nested streaming wire emits reasoning_content",
        '"reasoning_content":"probe"',
        post_stream_wire,
    )
    run.expect_not_in(
        "POST-PATCH: nested streaming wire has NO bare reasoning",
        '"reasoning":',
        post_stream_wire,
    )

    # Standalone leaf dumps still work too.
    leaf_chat = ChatMessageMirror(role="assistant", reasoning="probe")
    leaf_chat_wire = leaf_chat.model_dump_json()
    run.expect_in(
        "POST-PATCH: standalone ChatMessage emits reasoning_content",
        '"reasoning_content":"probe"',
        leaf_chat_wire,
    )
    run.expect_not_in(
        "POST-PATCH: standalone ChatMessage has NO bare reasoning",
        '"reasoning":',
        leaf_chat_wire,
    )

    leaf_delta = DeltaMessageMirror(reasoning="probe")
    leaf_delta_wire = leaf_delta.model_dump_json(exclude_unset=True)
    run.expect_in(
        "POST-PATCH: standalone DeltaMessage(exclude_unset) emits reasoning_content",
        '"reasoning_content":"probe"',
        leaf_delta_wire,
    )
    run.expect_not_in(
        "POST-PATCH: standalone DeltaMessage(exclude_unset) has NO bare reasoning",
        '"reasoning":',
        leaf_delta_wire,
    )

    # Negative control: patching ONLY leaves (the v1 behavior) FAILS
    # the nested test. Reset and re-run to prove this is the regression
    # the redesign actually fixes.
    class V1ChatMessage(OpenAIBaseModel):
        role: str
        content: str | None = None
        reasoning: str | None = None

    class V1Choice(OpenAIBaseModel):
        index: int
        message: V1ChatMessage

    class V1Response(OpenAIBaseModel):
        id: str
        choices: list[V1Choice]

    install_leaf(V1ChatMessage)  # only the leaf, like v1 of the patch
    v1_post = V1Response(
        id="x",
        choices=[
            V1Choice(
                index=0,
                message=V1ChatMessage(role="assistant", reasoning="probe"),
            )
        ],
    )
    v1_wire = v1_post.model_dump_json()
    # This wire SHOULD contain bare reasoning (v1 behavior) — we assert
    # that as the negative control to prove the v2 redesign was needed.
    run.expect_in(
        "NEGATIVE CONTROL: v1 (leaves-only) leaks 'reasoning' on nested",
        '"reasoning":"probe"',
        v1_wire,
    )
    run.expect_not_in(
        "NEGATIVE CONTROL: v1 (leaves-only) does NOT emit reasoning_content nested",
        '"reasoning_content"',
        v1_wire,
    )


# --------------------------------------------------------------------
# Section 4: Patch 4 — reasoning_field_ingest
# --------------------------------------------------------------------


def section_4_ingest() -> None:
    run.section("4. monkey_patch_reasoning_field_ingest.py")
    patch_path = PATCH_DIR / "monkey_patch_reasoning_field_ingest.py"
    chat_utils_src = (VLLM_DIR / "vllm/entrypoints/chat_utils.py").read_text()

    # Buggy line.
    bug = patch_constant(patch_path, "_BUG_LANDMARK")
    run.expect_in(
        "ingest bug line still present",
        bug,
        chat_utils_src,
    )
    role_gate = patch_constant(patch_path, "_ROLE_GATE_LANDMARK")
    run.expect_in(
        "role-gate landmark still present",
        role_gate,
        chat_utils_src,
    )

    # Signature.
    expected = patch_constant(patch_path, "_EXPECTED_PARAMS")
    sig = vllm_function_signature(
        "vllm/entrypoints/chat_utils.py",
        "_parse_chat_message_content",
    )
    run.expect_eq(
        "_parse_chat_message_content signature",
        sig,
        list(expected),
    )


# --------------------------------------------------------------------
# Section 5: Patch 5 — tool_call_in_think_rescue
# --------------------------------------------------------------------


def section_5_detector() -> None:
    """Patch 5 — tool_call_in_think_detector. Wraps BOTH
    ``extract_reasoning`` (non-streaming) AND ``extract_reasoning_streaming``
    so the WARNING fires regardless of delivery modality. Same byte-
    identical format string from both wrappers (one log forwarder regex
    covers both code paths).
    """
    run.section("5. monkey_patch_tool_call_in_think_detector.py")
    patch_path = PATCH_DIR / "monkey_patch_tool_call_in_think_detector.py"
    patch_src = patch_path.read_text()
    parser_src = (VLLM_DIR / "vllm/reasoning/qwen3_reasoning_parser.py").read_text()
    base_src = (VLLM_DIR / "vllm/reasoning/basic_parsers.py").read_text()

    # Class hierarchy still as expected.
    run.expect_in(
        "Qwen3ReasoningParser still subclasses BaseThinkingReasoningParser",
        "class Qwen3ReasoningParser(BaseThinkingReasoningParser):",
        parser_src,
    )
    run.expect_in(
        "BaseThinkingReasoningParser still defined in basic_parsers.py",
        "class BaseThinkingReasoningParser",
        base_src,
    )

    # Patch tag.
    expected_tag = patch_constant(patch_path, "_PATCH_TAG")
    run.expect_eq(
        "_PATCH_TAG is the v2 detector tag (wraps both surfaces)",
        expected_tag,
        "qwen36-agent-setup-tool-call-in-think-detector-v2",
    )

    # Non-streaming landmark substring still anchors extract_reasoning.
    ns_landmark = patch_constant(patch_path, "_NONSTREAMING_LANDMARK")
    run.expect_in(
        "non-streaming landmark present in master",
        ns_landmark,
        parser_src,
    )

    # Streaming landmark substring anchors extract_reasoning_streaming.
    s_landmark = patch_constant(patch_path, "_STREAMING_LANDMARK")
    run.expect_in(
        "streaming landmark present in master",
        s_landmark,
        parser_src,
    )

    # extract_reasoning signature unchanged at the pin.
    ns_sig = vllm_function_signature(
        "vllm/reasoning/qwen3_reasoning_parser.py",
        "Qwen3ReasoningParser.extract_reasoning",
    )
    run.expect_eq(
        "extract_reasoning signature",
        ns_sig,
        ["self", "model_output", "request"],
    )

    # extract_reasoning_streaming signature unchanged at the pin.
    s_sig = vllm_function_signature(
        "vllm/reasoning/qwen3_reasoning_parser.py",
        "Qwen3ReasoningParser.extract_reasoning_streaming",
    )
    run.expect_eq(
        "extract_reasoning_streaming signature",
        s_sig,
        [
            "self",
            "previous_text",
            "current_text",
            "delta_text",
            "previous_token_ids",
            "current_token_ids",
            "delta_token_ids",
        ],
    )

    # Both wrappers must use the same shared _WARNING_FORMAT constant —
    # this is what the host_ops/qwen36_warning_forwarder.py regex anchors
    # on. If the two wrappers ever drifted to different format strings,
    # the forwarder would silently miss one half of events.
    run.expect_in(
        "shared _WARNING_FORMAT constant defined",
        '_WARNING_FORMAT: str = (',
        patch_src,
    )
    # The forwarder anchors on this exact phrase; both wrappers funnel
    # through _WARNING_FORMAT and must contain it (verified by the
    # constant being a single source of truth).
    run.expect_in(
        "_WARNING_FORMAT contains the forwarder-required prefix",
        "model_emit_warning kind=tool_call_in_reasoning",
        patch_src,
    )

    # Both wrappers must reference _WARNING_FORMAT (proves the byte-
    # identical-format property is structural, not coincidental).
    nonstream_uses = patch_src.count("_WARNING_FORMAT,\n")
    run.expect_eq(
        "both wrappers funnel through the shared _WARNING_FORMAT constant",
        nonstream_uses,
        2,
    )


# --------------------------------------------------------------------
# Section 11: Patch 6 — default_sampling_params
# --------------------------------------------------------------------


def section_11_default_sampling_params() -> None:
    """Patch 6 — monkey_patch_default_sampling_params. Wraps
    ``ChatCompletionRequest.to_sampling_params`` to enforce Qwen3.6
    Best Practices defaults for fields the client did NOT explicitly
    send. The detection primitive is Pydantic v2's ``model_fields_set``.
    """
    run.section("11. monkey_patch_default_sampling_params.py")
    patch_path = PATCH_DIR / "monkey_patch_default_sampling_params.py"
    protocol_src = (
        VLLM_DIR / "vllm/entrypoints/openai/chat_completion/protocol.py"
    ).read_text()
    sampling_src = (VLLM_DIR / "vllm/sampling_params.py").read_text()

    # Patch tag.
    expected_tag = patch_constant(patch_path, "_PATCH_TAG")
    run.expect_eq(
        "_PATCH_TAG is the v1 default-sampling-params tag",
        expected_tag,
        "qwen36-agent-setup-default-sampling-params-v1",
    )

    # Target class still defined in the expected module.
    run.expect_in(
        "ChatCompletionRequest still defined in chat_completion.protocol",
        "class ChatCompletionRequest(",
        protocol_src,
    )

    # The body landmark we anchor on (proves to_sampling_params still
    # ends in a SamplingParams.from_optional(...) call — the wrapper's
    # delegate-then-overwrite contract relies on that shape).
    landmark = patch_constant(patch_path, "_FROM_OPTIONAL_LANDMARK")
    run.expect_in(
        "from_optional landmark present in to_sampling_params source",
        landmark,
        protocol_src,
    )

    # Method signature.
    expected_params = patch_constant(patch_path, "_EXPECTED_PARAMS")
    sig = vllm_function_signature(
        "vllm/entrypoints/openai/chat_completion/protocol.py",
        "ChatCompletionRequest.to_sampling_params",
    )
    run.expect_eq(
        "ChatCompletionRequest.to_sampling_params signature",
        sig,
        list(expected_params),
    )

    # Each field the patch claims to enforce a default for must still
    # appear as an attribute on SamplingParams (we do setattr on the
    # returned instance — a missing attribute would be silently ignored
    # without strict-attr typing).
    qwen_defaults = patch_constant(patch_path, "QWEN36_DEFAULTS")
    run.expect(
        "QWEN36_DEFAULTS is a non-empty dict",
        isinstance(qwen_defaults, dict) and bool(qwen_defaults),
        f"got {type(qwen_defaults).__name__}: {qwen_defaults!r}",
    )
    for name in qwen_defaults:
        # SamplingParams field-line regex: `    <name>: <type> = <default>`
        pattern = re.compile(
            rf"^\s{{4}}{re.escape(name)}\s*:", re.MULTILINE
        )
        run.expect(
            f"SamplingParams declares field {name!r}",
            pattern.search(sampling_src) is not None,
            f"no `    {name}:` field declaration in vllm/sampling_params.py",
        )

    # The request-side fields whose presence in `model_fields_set` we
    # rely on must each be declared in ChatCompletionRequest at master.
    required = patch_constant(patch_path, "_REQUEST_FIELDS_REQUIRED")
    for name in required:
        pattern = re.compile(
            rf"^\s{{4}}{re.escape(name)}\s*:", re.MULTILINE
        )
        run.expect(
            f"ChatCompletionRequest declares field {name!r}",
            pattern.search(protocol_src) is not None,
            f"no `    {name}:` field declaration in chat_completion/protocol.py",
        )


# --------------------------------------------------------------------
# Section 6: Launcher self-consistency (renumbered after section 5
# absorbed the rescue/detector slot)
# --------------------------------------------------------------------


def section_6_launcher() -> None:
    run.section("6. launch_with_patches.py — registry consistency")
    launcher_src = (PATCH_DIR / "launch_with_patches.py").read_text()

    # Every patch module is registered in _PATCH_MODULES (the 11 surviving).
    expected_modules = (
        "monkey_patch_qwen3_coder",
        "monkey_patch_hybrid_kv_allocator",
        "monkey_patch_reasoning_field_egress",
        "monkey_patch_reasoning_field_ingest",
        "monkey_patch_tool_call_in_think_detector",
        "monkey_patch_default_sampling_params",
        "monkey_patch_qwen3_coder_grammar",
        "monkey_patch_request_memory_snapshot",
        "monkey_patch_tool_role_media_preserve",
        "monkey_patch_mm_cache_validator_eviction",
        "monkey_patch_qwen3_coder_streaming_truncation",
    )
    for name in expected_modules:
        run.expect_in(
            f"_PATCH_MODULES contains {name}",
            f'"{name}"',
            launcher_src,
        )
    for name in expected_modules:
        run.expect_in(
            f"_PATCH_VERIFICATION includes {name}",
            f'"{name}":',
            launcher_src,
        )

    # Deleted patches must not be referenced by name in _PATCH_MODULES /
    # _PATCH_VERIFICATION (these strings may appear in historical
    # comments, but not as registry entries).
    for deleted in (
        "monkey_patch_extract_tool_calls_metrics",
        "monkey_patch_extract_tool_calls_streaming_metrics",
        "monkey_patch_tool_call_in_think_rescue",
    ):
        run.expect_not_in(
            f"_PATCH_MODULES does NOT contain {deleted}",
            f'"{deleted}",',
            launcher_src,
        )


# --------------------------------------------------------------------
# Section 6b: Logger-name consistency across all patches.
# --------------------------------------------------------------------
#
# Per Item 4 of the 2026-04-28 implementation: every patch module must
# init its logger via `init_logger(f"vllm.qwen36_patches.{__name__}")`
# (the prefix makes Python's logging hierarchy attach vLLM's stdout
# handler to the patch's logger so info/warning lines reach
# `docker logs qwen36`). A bare `init_logger(__name__)` produces a
# logger that is invisible by default — silent in containers.


def section_6b_logger_naming() -> None:
    run.section("6b. Patch logger-naming consistency")
    expected_literal = 'init_logger(f"vllm.qwen36_patches.{__name__}")'
    for path in sorted(PATCH_DIR.glob("monkey_patch_*.py")):
        src = path.read_text()
        run.expect_in(
            f"{path.name} uses init_logger(f'vllm.qwen36_patches.{{__name__}}')",
            expected_literal,
            src,
        )
        # Forbid the bare form.
        run.expect_not_in(
            f"{path.name} does NOT use bare init_logger(__name__)",
            "init_logger(__name__)",
            src,
        )


# --------------------------------------------------------------------
# Section 9: No silent failures — anti-pattern checker
# --------------------------------------------------------------------


def section_9_no_silent_failures() -> None:
    """Anti-pattern check across the 7 patches: refuse on ``except: pass``,
    refuse on ``except ImportError`` blocks that don't raise, refuse on
    ``_logger.debug()`` (DEBUG is invisible; surprises must surface at
    WARNING+), enforce ``except Exception`` count budget per file (target:
    zero), and verify each typed RefusedError class is actually raised.
    """
    run.section("9. No silent failures — anti-pattern check")

    patch_files = sorted(PATCH_DIR.glob("monkey_patch_*.py"))

    # (a) No ``except <X>: pass`` anywhere.
    forbidden_pass_re = re.compile(
        r"except\s+[A-Za-z_][A-Za-z_0-9.]*(\s+as\s+\w+)?:\s*\n\s*pass\b"
    )
    for path in patch_files:
        src = path.read_text()
        # Strip docstrings to avoid matching inside example code.
        # We're conservative — just check raw source.
        match = forbidden_pass_re.search(src)
        run.expect(
            f"{path.name}: no 'except: pass' anti-pattern",
            match is None,
            f"matched: {match.group(0)!r}" if match else "",
        )

    # (b) ``except ImportError`` blocks must end in ``raise`` (typed
    # refusal). The patches' top-level imports are all required.
    import_error_re = re.compile(
        r"except\s+ImportError(\s+as\s+\w+)?:\s*\n((?:\s+.*\n)+?)(?=\S)",
        re.MULTILINE,
    )
    for path in patch_files:
        src = path.read_text()
        for m in import_error_re.finditer(src):
            block = m.group(2)
            ok = "raise" in block
            run.expect(
                f"{path.name}: 'except ImportError' block raises",
                ok,
                f"block content: {block.strip()[:200]!r}" if not ok else "",
            )

    # (c) Forbid logger.debug() across the surviving patches — DEBUG is
    # invisible by default and unexpected runtime conditions must surface
    # at WARNING or higher. Catches a future regression that re-loosens it.
    for path in patch_files:
        src = path.read_text()
        run.expect_not_in(
            f"{path.name}: no _logger.debug()",
            "_logger.debug(",
            src,
        )

    # (d) Count REAL ``except Exception`` clauses via AST (not regex,
    # which over-counts string mentions inside docstrings). Each patch's
    # count must equal a hand-audited budget.
    #
    # All but one patch use typed exceptions only. ``mm_cache_validator_
    # eviction`` carries TWO ``except Exception`` clauses: one in the
    # production wrapper (eviction itself failing must NEVER mask the
    # original validator exception the caller is waiting on — best-
    # effort defense-in-depth) and one in Phase 7's test harness helper
    # that mirrors the production wrapper's structure. Both are
    # explicitly load-bearing — narrowing them would either crash the
    # response path on a downstream renderer bug or leave the test
    # harness less faithful to the wrapper it is verifying.
    expected_counts = {
        "monkey_patch_qwen3_coder.py": 0,
        "monkey_patch_hybrid_kv_allocator.py": 0,
        "monkey_patch_reasoning_field_egress.py": 0,
        "monkey_patch_reasoning_field_ingest.py": 0,
        "monkey_patch_tool_call_in_think_detector.py": 0,
        "monkey_patch_default_sampling_params.py": 0,
        "monkey_patch_qwen3_coder_grammar.py": 0,
        "monkey_patch_request_memory_snapshot.py": 0,
        "monkey_patch_tool_role_media_preserve.py": 0,
        "monkey_patch_mm_cache_validator_eviction.py": 2,
    }
    # Sanity: every monkey_patch_*.py file in the repo must have an
    # entry in this dict. Catches the silent-skip bug where adding a
    # new patch leaves it unchecked because the dict was never updated.
    discovered_files = {p.name for p in patch_files}
    expected_files = set(expected_counts)
    missing = discovered_files - expected_files
    extra = expected_files - discovered_files
    run.expect(
        "every monkey_patch_*.py is listed in the except-Exception budget",
        not missing and not extra,
        f"missing from dict: {sorted(missing)!r}; extra in dict: {sorted(extra)!r}",
    )
    for path in patch_files:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            run.expect(
                f"{path.name}: parses as Python",
                False,
                "SyntaxError",
            )
            continue
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            exc_type = node.type
            # ``except Exception`` or ``except Exception as e``
            if isinstance(exc_type, ast.Name) and exc_type.id == "Exception":
                count += 1
        expected = expected_counts.get(path.name)
        if expected is not None:
            run.expect_eq(
                f"{path.name}: real 'except Exception' clause count == {expected}",
                count,
                expected,
            )

    # (e) Every patch's typed RefusedError-style class is referenced
    # in at least one ``raise`` statement, proving the patch can
    # actually surface a refusal.
    for path in patch_files:
        src = path.read_text()
        # Find ``class XxxError(...)`` declarations whose name ends in
        # one of the typical refusal-error suffixes used in this repo.
        error_class_re = re.compile(
            r"^class\s+(\w*(?:Refused|Ambiguity)Error)\(",
            re.MULTILINE,
        )
        names = error_class_re.findall(src)
        for cls_name in names:
            raise_re = re.compile(rf"raise\s+{cls_name}\b")
            run.expect(
                f"{path.name}: raises {cls_name} from at least one site",
                raise_re.search(src) is not None,
                f"class declared but never raised — dead refusal path",
            )


# --------------------------------------------------------------------
# Section 10: sitecustomize + README consistency
# --------------------------------------------------------------------


def section_10_sitecustomize_and_readme() -> None:
    """Static cross-checks: sitecustomize._PATCH_MODULES == launcher's
    (drift would silently split EngineCore from PID 1); README §8.2
    docker run includes the load-bearing flags and bind-mounts every
    registered patch; pinned commit matches.
    """
    run.section("10. sitecustomize + README consistency")

    sitecustomize_path = PATCH_DIR / "sitecustomize.py"
    launcher_path = PATCH_DIR / "launch_with_patches.py"
    readme_path = PATCH_DIR / "README.md"

    run.expect(
        "sitecustomize.py exists at repo root",
        sitecustomize_path.is_file(),
        f"missing: {sitecustomize_path}",
    )
    if not sitecustomize_path.is_file():
        return  # skip remaining checks; their preconditions are unmet

    sitecustomize_modules = patch_constant(sitecustomize_path, "_PATCH_MODULES")
    is_str_tuple = isinstance(sitecustomize_modules, tuple) and all(
        isinstance(m, str) for m in sitecustomize_modules
    )
    run.expect(
        "sitecustomize._PATCH_MODULES is a tuple of strings",
        is_str_tuple,
        f"got {type(sitecustomize_modules).__name__}: "
        f"{sitecustomize_modules!r}",
    )

    launcher_modules = patch_constant(launcher_path, "_PATCH_MODULES")
    run.expect_eq(
        "sitecustomize._PATCH_MODULES == launch_with_patches._PATCH_MODULES",
        sitecustomize_modules,
        launcher_modules,
    )

    readme_src = readme_path.read_text()

    # The canonical `docker run` lives in install.sh (not README §8.2 prose).
    # All flag-presence and bind-mount checks read install.sh.
    install_path = PATCH_DIR / "install.sh"
    install_src = install_path.read_text()

    # install.sh contains exactly one `docker run` invocation (and only
    # one). A regression that adds a second invocation would mean two
    # containers stomping each other.
    docker_run_count = install_src.count("docker run ")
    run.expect_eq(
        "install.sh contains exactly one `docker run` invocation",
        docker_run_count,
        1,
    )

    # sitecustomize bind-mount — load-bearing for patches 3 and 8 reaching
    # the EngineCore subprocess.
    run.expect_in(
        "install.sh docker run includes sitecustomize bind-mount",
        "sitecustomize.py:/opt/patches/sitecustomize.py:ro",
        install_src,
    )
    # 2026-04-28: bridge networking with --network host explicitly avoided.
    # With --network host, the EngineCore subprocess's ZMQ IPC port (and
    # any other internal RPC vLLM opens) binds to the host's all-interfaces,
    # which on a publicly-routable machine = the public IP.
    run.expect_in(
        "install.sh docker run publishes only on host loopback",
        "-p 127.0.0.1:8001:8001",
        install_src,
    )
    run.expect_not_in(
        "install.sh docker run does NOT use --network host",
        "--network host",
        install_src,
    )
    # The vLLM CLI must bind on 0.0.0.0 *inside the container's* network
    # namespace so the publish DNAT can reach it. This is the container's
    # own private interface, not a host interface.
    run.expect_in(
        "install.sh docker run binds vLLM CLI to 0.0.0.0 INSIDE the container",
        "--host 0.0.0.0",
        install_src,
    )

    # PYTHONPATH=/opt/patches is load-bearing: without it, CPython's
    # site.py never finds /opt/patches/sitecustomize.py, and patch 3
    # (hybrid_kv_allocator) is silently dead in the spawned EngineCore.
    run.expect_in(
        "install.sh docker run includes -e PYTHONPATH=/opt/patches",
        "PYTHONPATH=/opt/patches",
        install_src,
    )

    # --entrypoint python3 is load-bearing: without it, the image's
    # default ENTRYPOINT [vllm, serve] runs and the launcher is
    # never executed at all (no patches install).
    run.expect_in(
        "install.sh docker run includes --entrypoint python3",
        "--entrypoint python3",
        install_src,
    )

    # The launcher itself must be bind-mounted into /opt/patches/launch.py
    # (the path the docker command then invokes). Source name in repo is
    # launch_with_patches.py; in-container path is /opt/patches/launch.py.
    run.expect_in(
        "install.sh docker run bind-mounts launch_with_patches.py at /opt/patches/launch.py",
        "launch_with_patches.py:/opt/patches/launch.py:ro",
        install_src,
    )

    # mnbt=4096 (post-Pillars-of-Creation OOM tuning, §11 B9/B12).
    run.expect_in(
        "install.sh docker run uses --max-num-batched-tokens 4096",
        "--max-num-batched-tokens 4096",
        install_src,
    )
    run.expect_not_in(
        "install.sh docker run does NOT still use --max-num-batched-tokens 8192",
        "--max-num-batched-tokens 8192",
        install_src,
    )

    # -cc pin for cudagraph capture sizes — drift-immune against future
    # image bumps.
    run.expect_in(
        "install.sh docker run includes -cc cudagraph_capture_sizes pin",
        "cudagraph_capture_sizes",
        install_src,
    )
    run.expect_in(
        "install.sh docker run pins cudagraph capture sizes to [1,2,4,8]",
        "[1,2,4,8]",
        install_src,
    )

    # Every entry in _PATCH_MODULES must have a corresponding bind-mount
    # line in install.sh's docker run. Catches the regression where a
    # patch is added to the launcher tuple but the maintainer forgets
    # the bind-mount.
    if isinstance(sitecustomize_modules, tuple):
        for module_name in sitecustomize_modules:
            expected_mount = (
                f"{module_name}.py:/opt/patches/{module_name}.py:ro"
            )
            run.expect_in(
                f"install.sh docker run bind-mounts {module_name}",
                expected_mount,
                install_src,
            )

    # 2026-04-28: §8.3 now contains the 2-concurrent warmup shape (loose
    # assertions on the load-bearing markers — exact JSON varies).
    run.expect(
        "README §8.3 warmup uses enable_thinking:false",
        re.search(r"enable_thinking.*false", readme_src) is not None,
        "expected a chat_template_kwargs entry with enable_thinking:false in §8.3",
    )
    run.expect(
        "README §8.3 warmup uses --max-time 60",
        "--max-time 60" in readme_src,
        "expected --max-time 60 on the warmup curl invocations",
    )

    # 2026-04-28: §8.4 wedge-recovery probe has its own subsection.
    run.expect_in(
        "README §8.4 documents the deep-probe install path",
        "qwen36_deep_probe.sh",
        readme_src,
    )

    # README's pinned commit string must match EXPECTED_COMMIT.
    run.expect_in(
        f"README references EXPECTED_COMMIT={EXPECTED_COMMIT[:12]}…",
        EXPECTED_COMMIT,
        readme_src,
    )

    # Pinned image digest is referenced in three places — README §8.2,
    # install.sh, and uninstall.sh. Drift between any two is a bug:
    # install.sh would launch a container uninstall.sh's validator
    # would then refuse to remove. Catch divergence here.
    install_src = (PATCH_DIR / "install.sh").read_text()
    uninstall_src = (PATCH_DIR / "uninstall.sh").read_text()
    digest_in_install = re.search(
        r'IMAGE_DIGEST="(sha256:[0-9a-f]{64})"', install_src
    )
    digest_in_uninstall = re.search(
        r'EXPECTED_DIGEST="(sha256:[0-9a-f]{64})"', uninstall_src
    )
    run.expect(
        "install.sh declares IMAGE_DIGEST as a sha256:... constant",
        digest_in_install is not None,
    )
    run.expect(
        "uninstall.sh declares EXPECTED_DIGEST as a sha256:... constant",
        digest_in_uninstall is not None,
    )
    if digest_in_install and digest_in_uninstall:
        run.expect_eq(
            "install.sh IMAGE_DIGEST == uninstall.sh EXPECTED_DIGEST",
            digest_in_install.group(1),
            digest_in_uninstall.group(1),
        )
        run.expect_in(
            "README §8.2 references the same pinned digest",
            digest_in_install.group(1),
            readme_src,
        )

    # install.sh and uninstall.sh must use the same container name —
    # if they drift, uninstall.sh's validator looks for the wrong
    # container. install.sh's docker run uses `--name "${CONTAINER_NAME}"`
    # which is the variable's expanded form.
    run.expect_in(
        "install.sh declares CONTAINER_NAME=qwen36",
        'CONTAINER_NAME="qwen36"',
        install_src,
    )
    run.expect_in(
        "uninstall.sh declares CONTAINER_NAME=qwen36",
        'CONTAINER_NAME="qwen36"',
        uninstall_src,
    )
    run.expect_in(
        'install.sh docker run uses --name "${CONTAINER_NAME}"',
        '--name "${CONTAINER_NAME}"',
        install_src,
    )


# --------------------------------------------------------------------
# Section 12: Patch 7 — qwen3_coder_grammar AST verification.
# --------------------------------------------------------------------


def section_12_qwen3_coder_grammar() -> None:
    """Patch 7 — monkey_patch_qwen3_coder_grammar. Overrides
    ``Qwen3CoderToolParser.adjust_request`` (currently inherited from
    ``ToolParser``) and flips ``supports_required_and_named=False``.
    Verify the expected master-side shape: parser class exists and is a
    subclass of ToolParser; either the subclass declares its own
    adjust_request OR the inherited base method exists at
    abstract_tool_parser.py; the base class still defines
    ``supports_required_and_named``.
    """
    run.section("12. monkey_patch_qwen3_coder_grammar.py")
    patch_path = PATCH_DIR / "monkey_patch_qwen3_coder_grammar.py"
    parser_src_path = VLLM_DIR / "vllm/tool_parsers/qwen3coder_tool_parser.py"
    abstract_src_path = VLLM_DIR / "vllm/tool_parsers/abstract_tool_parser.py"

    parser_src = parser_src_path.read_text()
    abstract_src = abstract_src_path.read_text()

    # Patch tag.
    expected_tag = patch_constant(patch_path, "_PATCH_TAG")
    run.expect_eq(
        "_PATCH_TAG is the v1 grammar tag",
        expected_tag,
        "qwen36-agent-setup-qwen3-coder-grammar-v1",
    )

    # Parse the parser source via AST to find the class definition.
    parser_tree = ast.parse(parser_src)
    parser_cls_node: ast.ClassDef | None = None
    for node in ast.walk(parser_tree):
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3CoderToolParser":
            parser_cls_node = node
            break
    run.expect(
        "Qwen3CoderToolParser class exists in master at the pinned commit",
        parser_cls_node is not None,
        "qwen3coder_tool_parser.py does not declare class Qwen3CoderToolParser",
    )

    # Subclass must NOT already declare adjust_request (otherwise patch's
    # remit is closed upstream).
    if parser_cls_node is not None:
        subclass_methods = {
            sub.name
            for sub in parser_cls_node.body
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        run.expect(
            "Qwen3CoderToolParser does NOT yet declare adjust_request "
            "(remit of patch 7)",
            "adjust_request" not in subclass_methods,
            "if upstream ships an override, delete patch 7 — see removal "
            "trigger in the docstring",
        )
        run.expect(
            "Qwen3CoderToolParser still subclasses ToolParser",
            any(
                isinstance(b, ast.Name) and b.id == "ToolParser"
                for b in parser_cls_node.bases
            ),
            "base list: " + ", ".join(
                b.id if isinstance(b, ast.Name) else "<expr>"
                for b in parser_cls_node.bases
            ),
        )

    # The inherited base method must still exist with the (self, request)
    # signature shape we override against.
    base_sig = vllm_function_signature(
        "vllm/tool_parsers/abstract_tool_parser.py",
        "ToolParser.adjust_request",
    )
    run.expect_eq(
        "ToolParser.adjust_request signature",
        base_sig,
        ["self", "request"],
    )

    # supports_required_and_named is still declared on the BASE class —
    # patch 7 sets the SUBCLASS attribute to False; the subclass dict
    # initially does not contain it (inherited).
    run.expect_in(
        "ToolParser still declares supports_required_and_named",
        "supports_required_and_named",
        abstract_src,
    )
    if parser_cls_node is not None:
        subclass_assignments = {
            t.id
            for sub in parser_cls_node.body
            if isinstance(sub, ast.Assign)
            for t in sub.targets
            if isinstance(t, ast.Name)
        }
        # Also class-level annotated assignments (the form used in the
        # base class itself).
        for sub in parser_cls_node.body:
            if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                subclass_assignments.add(sub.target.id)
        run.expect(
            "Qwen3CoderToolParser does NOT yet override "
            "supports_required_and_named (remit of patch 7)",
            "supports_required_and_named" not in subclass_assignments,
            "if upstream ships False here, patch 7's flag flip is "
            "redundant — re-audit",
        )

    # Source landmarks the patch reads at import time must still be
    # present in the master source.
    base_landmark = patch_constant(patch_path, "_BASE_LANDMARK")
    run.expect_in(
        "BASE_LANDMARK still present in ToolParser.adjust_request",
        base_landmark,
        abstract_src,
    )
    base_get_json_landmark = patch_constant(
        patch_path, "_BASE_GET_JSON_LANDMARK"
    )
    run.expect_in(
        "BASE_GET_JSON_LANDMARK still present in ToolParser.adjust_request",
        base_get_json_landmark,
        abstract_src,
    )


# --------------------------------------------------------------------
# Section 13: Patch 10 — mm_cache_validator_eviction AST verification.
# --------------------------------------------------------------------


def section_13_mm_cache_validator_eviction() -> None:
    """Patch 10 — monkey_patch_mm_cache_validator_eviction. Wraps
    ``OpenAIServingChat.create_chat_completion`` and on
    ``ValueError`` / ``VLLMValidationError`` calls
    ``self.renderer.clear_mm_cache_async()`` to restore the sender↔
    receiver mm-cache mirror invariant. Verify the expected master-side
    shape: target class exists; create_chat_completion is async with the
    expected signature; both load-bearing source landmarks
    (render_chat_request call + get_max_tokens call) are present in
    the body; get_max_tokens still raises ValueError; BaseRenderer
    exposes a no-arg async clear_mm_cache_async; OpenAIServing.__init__
    still assigns self.renderer = engine_client.renderer.
    """
    run.section("13. monkey_patch_mm_cache_validator_eviction.py")
    patch_path = PATCH_DIR / "monkey_patch_mm_cache_validator_eviction.py"
    serving_src = (
        VLLM_DIR / "vllm/entrypoints/openai/chat_completion/serving.py"
    ).read_text()
    base_serving_src = (
        VLLM_DIR / "vllm/entrypoints/openai/engine/serving.py"
    ).read_text()
    utils_src = (VLLM_DIR / "vllm/entrypoints/utils.py").read_text()
    renderer_src = (VLLM_DIR / "vllm/renderers/base.py").read_text()
    exceptions_src = (VLLM_DIR / "vllm/exceptions.py").read_text()

    # Patch tag.
    expected_tag = patch_constant(patch_path, "_PATCH_TAG")
    run.expect_eq(
        "_PATCH_TAG is the v1 mm-cache-validator-eviction tag",
        expected_tag,
        "qwen36-agent-setup-mm-cache-validator-eviction-v1",
    )

    # Target class still defined in the expected module.
    run.expect_in(
        "OpenAIServingChat still defined in chat_completion.serving",
        "class OpenAIServingChat(",
        serving_src,
    )
    run.expect_in(
        "OpenAIServing still defined in engine.serving",
        "class OpenAIServing:",
        base_serving_src,
    )

    # The wrapped method's signature.
    expected_params = patch_constant(patch_path, "_EXPECTED_PARAMS")
    sig = vllm_function_signature(
        "vllm/entrypoints/openai/chat_completion/serving.py",
        "OpenAIServingChat.create_chat_completion",
    )
    run.expect_eq(
        "OpenAIServingChat.create_chat_completion signature",
        sig,
        list(expected_params),
    )

    # Verify the method is declared async at master.
    serving_tree = ast.parse(serving_src)
    chat_cls_node: ast.ClassDef | None = None
    for node in ast.walk(serving_tree):
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingChat":
            chat_cls_node = node
            break
    run.expect(
        "OpenAIServingChat class exists in master at the pinned commit",
        chat_cls_node is not None,
        "chat_completion/serving.py does not declare class OpenAIServingChat",
    )
    if chat_cls_node is not None:
        method_node: ast.AsyncFunctionDef | ast.FunctionDef | None = None
        for sub in chat_cls_node.body:
            if (
                isinstance(sub, (ast.AsyncFunctionDef, ast.FunctionDef))
                and sub.name == "create_chat_completion"
            ):
                method_node = sub
                break
        run.expect(
            "OpenAIServingChat declares create_chat_completion",
            method_node is not None,
            "method missing in subclass body",
        )
        if method_node is not None:
            run.expect(
                "create_chat_completion is async (AsyncFunctionDef)",
                isinstance(method_node, ast.AsyncFunctionDef),
                f"got {type(method_node).__name__} — wrapper's await-and-"
                f"catch contract requires async",
            )

    # Both source landmarks must still be present in the method body.
    render_landmark = patch_constant(patch_path, "_RENDER_LANDMARK")
    run.expect_in(
        "render_chat_request landmark present in create_chat_completion",
        render_landmark,
        serving_src,
    )
    get_max_tokens_landmark = patch_constant(
        patch_path, "_GET_MAX_TOKENS_LANDMARK"
    )
    run.expect_in(
        "get_max_tokens landmark present in create_chat_completion",
        get_max_tokens_landmark,
        serving_src,
    )

    # get_max_tokens still raises ValueError (the typed exception we catch).
    raise_landmark = patch_constant(
        patch_path, "_GET_MAX_TOKENS_RAISE_LANDMARK"
    )
    run.expect_in(
        "get_max_tokens still raises ValueError at master",
        raise_landmark,
        utils_src,
    )

    # OpenAIServing.__init__ still assigns self.renderer = engine_client.renderer.
    # This is what makes the wrapper's `self.renderer.clear_mm_cache_async()`
    # call work. If upstream factors out the attribute, the wrapper's
    # defensive shape-check fires and the eviction is a no-op (with a
    # warning) — but at install time we want to refuse instead.
    run.expect_in(
        "OpenAIServing.__init__ still assigns self.renderer = engine_client.renderer",
        "self.renderer = engine_client.renderer",
        base_serving_src,
    )

    # BaseRenderer exposes async clear_mm_cache_async with signature (self).
    run.expect_in(
        "BaseRenderer still defines clear_mm_cache_async",
        "async def clear_mm_cache_async(self)",
        renderer_src,
    )

    # VLLMValidationError still extends ValueError.
    run.expect_in(
        "VLLMValidationError still extends ValueError",
        "class VLLMValidationError(ValueError)",
        exceptions_src,
    )

    # AST-level check: the production wrapper's exception-catch contract.
    # Catches the drift scenario where someone relaxes the except clause
    # to `Exception` (which would mask actual bugs and bypass the wrapper's
    # documented "only validator throws trigger eviction" guarantee). The
    # Phase 7 behavioural cases test the patch's STRUCTURAL TWIN built by
    # _build_test_wrapper, not the production wrapper directly — that's
    # acceptable because the wrappers are documented as identical, but
    # we verify that identity here at the AST level.
    patch_tree = ast.parse(patch_path.read_text())
    wrapper_node: ast.AsyncFunctionDef | None = None
    for node in ast.walk(patch_tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_create_chat_completion_with_mm_cache_eviction"
        ):
            wrapper_node = node
            break
    run.expect(
        "patch defines async _create_chat_completion_with_mm_cache_eviction",
        wrapper_node is not None,
        "wrapper async function not found in patch source",
    )
    if wrapper_node is not None:
        # Find the outer try/except that owns the eviction logic.
        try_nodes = [n for n in wrapper_node.body if isinstance(n, ast.Try)]
        run.expect_eq(
            "wrapper has exactly one top-level try/except",
            len(try_nodes),
            1,
        )
        if try_nodes:
            handlers = try_nodes[0].handlers
            run.expect_eq(
                "wrapper's try has exactly one except handler",
                len(handlers),
                1,
            )
            if handlers:
                exc_type = handlers[0].type
                # The catch must be a tuple of exactly (ValueError,
                # VLLMValidationError). Reject `except Exception`,
                # `except ValueError` alone, etc.
                run.expect(
                    "wrapper catches a tuple of exception types",
                    isinstance(exc_type, ast.Tuple),
                    f"got {type(exc_type).__name__ if exc_type else 'None'} — "
                    f"the catch must be `except (ValueError, VLLMValidationError)`",
                )
                if isinstance(exc_type, ast.Tuple):
                    caught_names = [
                        elt.id for elt in exc_type.elts if isinstance(elt, ast.Name)
                    ]
                    run.expect_eq(
                        "wrapper catches exactly (ValueError, VLLMValidationError)",
                        sorted(caught_names),
                        ["VLLMValidationError", "ValueError"],
                    )


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


def main() -> int:
    print(f"{_BOLD}Verifying patches against vLLM @ "
          f"{EXPECTED_COMMIT}{_RESET}")
    print(f"  patches: {PATCH_DIR}")
    print(f"  vllm:    {VLLM_DIR}")

    sections: list[Callable[[], None]] = [
        section_0_pin_consistency,
        section_1_qwen3_coder,
        section_2_hybrid_kv,
        section_3_egress_static,
        section_3_egress_mechanism,
        section_4_ingest,
        section_5_detector,
        section_11_default_sampling_params,
        section_12_qwen3_coder_grammar,
        section_13_mm_cache_validator_eviction,
        section_6_launcher,
        section_6b_logger_naming,
        section_9_no_silent_failures,
        section_10_sitecustomize_and_readme,
    ]
    for fn in sections:
        try:
            fn()
        except Exception as e:
            run.expect(
                f"{fn.__name__} raised {type(e).__name__}",
                False,
                f"{type(e).__name__}: {e}",
            )

    return run.report()


if __name__ == "__main__":
    sys.exit(main())
