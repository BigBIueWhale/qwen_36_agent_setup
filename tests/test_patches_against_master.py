"""Static + structural-mirror verification that all 7 monkey-patches
will apply cleanly against the pinned master commit.

This script does NOT import vLLM (the host has no torch / CUDA / Triton).
Instead it:

1. Walks each patch's source-level landmark constants and asserts each
   one is present in the corresponding master source file.
2. AST-parses each master source file to extract function signatures
   and verifies they match what the patches expect.
3. For the egress patch (the only one with a non-trivial mechanism
   that doesn't reduce to landmark matching), runs a real Pydantic v2
   experiment on a structural mirror of vLLM's wire wrappers and
   verifies the patch's mechanism produces the desired wire JSON.

The pinned commit under test is read from each patch's
``_PINNED_VLLM_COMMIT`` constant — they must all agree, and they must
match the HEAD of /tmp/qwen36_research/vllm.

Exits with status 0 iff every check passes; non-zero on any failure.
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
    abstract_src = (
        VLLM_DIR / "vllm/tool_parsers/abstract_tool_parser.py"
    ).read_text()
    engine_src = (VLLM_DIR / "vllm/entrypoints/openai/engine/protocol.py").read_text()

    # The buggy landmark string — must be present.
    sentinel = patch_constant(patch_path, "_BUGGY_SENTINEL")
    run.expect_in(
        "buggy sentinel str.index(\">\") still present at master",
        sentinel,
        qwen3_src,
    )

    # Regex shape landmark — must appear specifically in the SUBCLASS
    # source (this is the qwen3-specific compiled regex, not inherited).
    regex_landmark = patch_constant(patch_path, "_UPSTREAM_REGEX_LANDMARK")
    run.expect_in(
        "tool_call_parameter_regex shape unchanged in subclass __init__",
        regex_landmark,
        qwen3_src,
    )

    # ``self.tool_call_parameter_regex`` MUST be assigned in the subclass
    # source (it is a qwen3-specific compiled pattern). Use a strict
    # word-boundary regex so we don't match unrelated identifiers.
    expected_attr = patch_constant(patch_path, "_EXPECTED_REGEX_ATTR")
    regex_attr_re = re.compile(
        rf"\bself\.{re.escape(expected_attr)}\s*[:=]"
    )
    run.expect(
        "Qwen3CoderToolParser subclass __init__ assigns "
        f"self.{expected_attr}",
        regex_attr_re.search(qwen3_src) is not None,
        f"regex {regex_attr_re.pattern!r} did not match",
    )

    # ``self.tools`` may be assigned by the subclass OR inherited from
    # ``ToolParser.__init__`` (the actual situation at this pin).
    # The patch's MRO-walking landmark check requires the assignment to
    # appear ANYWHERE in the inheritance chain — and so do we, so a
    # refactor that moves the assignment between subclass/base is fine
    # but its disappearance from BOTH is caught.
    tools_assignment_re = re.compile(r"\bself\.tools\s*[:=]")
    run.expect(
        "self.tools assigned somewhere in the Qwen3CoderToolParser MRO "
        "(subclass __init__ OR base)",
        tools_assignment_re.search(qwen3_src + "\n" + abstract_src) is not None,
        "neither qwen3coder_tool_parser.py nor abstract_tool_parser.py "
        "assigns self.tools — the patched _parse_xml_function_call body "
        "would AttributeError at request time.",
    )

    # Signatures.
    sig = vllm_function_signature(
        "vllm/tool_parsers/qwen3coder_tool_parser.py",
        "Qwen3CoderToolParser._parse_xml_function_call",
    )
    run.expect_eq(
        "_parse_xml_function_call signature",
        sig,
        ["self", "function_call_str"],
    )

    sig_cpv = vllm_function_signature(
        "vllm/tool_parsers/qwen3coder_tool_parser.py",
        "Qwen3CoderToolParser._convert_param_value",
    )
    run.expect_eq(
        "_convert_param_value signature",
        sig_cpv,
        ["self", "param_value", "param_name", "param_config", "func_name"],
    )

    sig_ftp = vllm_function_signature(
        "vllm/tool_parsers/utils.py",
        "find_tool_properties",
    )
    run.expect_eq(
        "find_tool_properties signature",
        sig_ftp,
        ["tools", "tool_name"],
    )

    # ToolCall / FunctionCall importable from engine/protocol.py.
    run.expect_in(
        "engine.protocol exports class FunctionCall",
        "class FunctionCall(",
        engine_src,
    )
    run.expect_in(
        "engine.protocol exports class ToolCall",
        "class ToolCall(",
        engine_src,
    )


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
    chat_src = (
        VLLM_DIR / "vllm/entrypoints/openai/chat_completion/protocol.py"
    ).read_text()
    engine_src = (VLLM_DIR / "vllm/entrypoints/openai/engine/protocol.py").read_text()

    # All six target classes must exist.
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
    run.expect_in(
        "engine.protocol exports DeltaMessage",
        "class DeltaMessage(",
        engine_src,
    )

    # Each leaf carries `reasoning: str | None = None`.
    leaf_re = re.compile(r"reasoning:\s*str\s*\|\s*None\s*=\s*None")
    run.expect(
        "ChatMessage has reasoning: str | None = None",
        leaf_re.search(chat_src) is not None,
    )
    run.expect(
        "DeltaMessage has reasoning: str | None = None",
        leaf_re.search(engine_src) is not None,
    )

    # No pre-existing reasoning_content field.
    run.expect_not_in(
        "ChatMessage does not already declare reasoning_content",
        "reasoning_content:",
        chat_src,
    )
    run.expect_not_in(
        "DeltaMessage does not already declare reasoning_content",
        "reasoning_content:",
        engine_src,
    )

    # Wrappers' nested-field shape.
    # ChatCompletionResponseChoice: message: ChatMessage
    run.expect_in(
        "ChatCompletionResponseChoice has 'message: ChatMessage'",
        "message: ChatMessage",
        chat_src,
    )
    # ChatCompletionResponseStreamChoice: delta: DeltaMessage
    run.expect_in(
        "ChatCompletionResponseStreamChoice has 'delta: DeltaMessage'",
        "delta: DeltaMessage",
        chat_src,
    )
    # ChatCompletionResponse: choices: list[ChatCompletionResponseChoice]
    run.expect_in(
        "ChatCompletionResponse has 'choices: list[ChatCompletionResponseChoice]'",
        "choices: list[ChatCompletionResponseChoice]",
        chat_src,
    )
    # ChatCompletionStreamResponse: choices: list[ChatCompletionResponseStreamChoice]
    run.expect_in(
        "ChatCompletionStreamResponse has 'choices: list[ChatCompletionResponseStreamChoice]'",
        "choices: list[ChatCompletionResponseStreamChoice]",
        chat_src,
    )


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


def section_5_rescue() -> None:
    run.section("5. monkey_patch_tool_call_in_think_rescue.py")
    patch_path = PATCH_DIR / "monkey_patch_tool_call_in_think_rescue.py"
    parser_src = (
        VLLM_DIR / "vllm/reasoning/qwen3_reasoning_parser.py"
    ).read_text()
    base_src = (VLLM_DIR / "vllm/reasoning/basic_parsers.py").read_text()
    engine_src = (VLLM_DIR / "vllm/entrypoints/openai/engine/protocol.py").read_text()

    # Class hierarchy.
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

    # Source landmarks.
    nonstreaming = patch_constant(patch_path, "_NONSTREAMING_LANDMARK")
    streaming = patch_constant(patch_path, "_STREAMING_LANDMARK")
    run.expect_in(
        "non-streaming landmark present",
        nonstreaming,
        parser_src,
    )
    run.expect_in(
        "streaming landmark present",
        streaming,
        parser_src,
    )

    # Signatures (use AST extraction).
    sig_extract = vllm_function_signature(
        "vllm/reasoning/qwen3_reasoning_parser.py",
        "Qwen3ReasoningParser.extract_reasoning",
    )
    run.expect_eq(
        "extract_reasoning signature",
        sig_extract,
        ["self", "model_output", "request"],
    )
    sig_stream = vllm_function_signature(
        "vllm/reasoning/qwen3_reasoning_parser.py",
        "Qwen3ReasoningParser.extract_reasoning_streaming",
    )
    run.expect_eq(
        "extract_reasoning_streaming signature",
        sig_stream,
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

    # Token literals — ``<think>``/``</think>`` in property bodies.
    expected_start = patch_constant(patch_path, "_EXPECTED_START_TOKEN")
    expected_end = patch_constant(patch_path, "_EXPECTED_END_TOKEN")
    run.expect_in(
        "start_token literal '<think>' present",
        f'"{expected_start}"',
        parser_src,
    )
    run.expect_in(
        "end_token literal '</think>' present",
        f'"{expected_end}"',
        parser_src,
    )

    # DeltaMessage carries reasoning + content.
    run.expect_in(
        "DeltaMessage has reasoning field",
        "reasoning: str | None = None",
        engine_src,
    )
    run.expect_in(
        "DeltaMessage has content field",
        "content: str | None = None",
        engine_src,
    )

    # PR #35687 awareness: master's extract_reasoning has the
    # implicit-end-via-tool_call branch (lines 142-156). Our patch's
    # premise is: that branch only fires when </think> is ABSENT, so
    # when both </think> and a mid-think <tool_call> are emitted, the
    # partition runs first and our rescue is still load-bearing.
    # Verify the partition still happens BEFORE the implicit-end check.
    extract_block_match = re.search(
        r"def extract_reasoning\(.*?(?=\n    def |\nclass )",
        parser_src,
        re.DOTALL,
    )
    extract_body = extract_block_match.group(0) if extract_block_match else ""
    partition_pos = extract_body.find("model_output.partition(self.end_token)")
    tool_call_pos = extract_body.find(
        "model_output.find(self._tool_call_tag)"
    )
    run.expect(
        "extract_reasoning still partitions on </think> BEFORE implicit-end-via-tool_call check",
        partition_pos != -1
        and tool_call_pos != -1
        and partition_pos < tool_call_pos,
        f"partition_pos={partition_pos}, tool_call_pos={tool_call_pos}",
    )


# --------------------------------------------------------------------
# Section 6: Patch 6 — extract_tool_calls_metrics
# --------------------------------------------------------------------


def section_6_metrics() -> None:
    run.section("6. monkey_patch_extract_tool_calls_metrics.py")
    patch_path = PATCH_DIR / "monkey_patch_extract_tool_calls_metrics.py"
    patch_src = patch_path.read_text()
    target_src = (
        VLLM_DIR / "vllm/tool_parsers/qwen3coder_tool_parser.py"
    ).read_text()
    engine_src = (VLLM_DIR / "vllm/entrypoints/openai/engine/protocol.py").read_text()
    prom_src = (
        VLLM_DIR / "vllm/v1/metrics/prometheus.py"
    ).read_text()

    # Counter name MUST NOT contain "vllm" — vLLM's
    # unregister_vllm_metrics() in vllm/v1/metrics/prometheus.py would
    # deregister it at PrometheusStatLogger init.
    counter_name = patch_constant(patch_path, "_COUNTER_NAME")
    run.expect(
        f"counter name does NOT contain 'vllm' (substring) "
        f"(name={counter_name!r})",
        "vllm" not in counter_name,
        "would be deregistered by unregister_vllm_metrics() — see the "
        "in-patch note. Choose a name without 'vllm' as a substring.",
    )
    # Confirm the upstream filter still has the "vllm" substring check
    # — if it changes, our defensive rename may be unnecessary.
    run.expect(
        "upstream unregister_vllm_metrics still filters on 'vllm' substring",
        '"vllm" in collector._name' in prom_src,
        "vLLM's filter has changed; re-audit whether the rename is "
        "still needed.",
    )

    # Signature.
    sig = vllm_function_signature(
        "vllm/tool_parsers/qwen3coder_tool_parser.py",
        "Qwen3CoderToolParser.extract_tool_calls",
    )
    run.expect_eq(
        "extract_tool_calls signature",
        sig,
        ["self", "model_output", "request"],
    )

    # Silent-failure landmarks.
    landmarks = patch_constant(patch_path, "_SILENT_FAILURE_LANDMARKS")
    for lm in landmarks:
        run.expect_in(
            f"silent-failure landmark {lm!r} present",
            lm,
            target_src,
        )

    # ExtractedToolCallInformation carries the three attrs.
    run.expect_in(
        "ExtractedToolCallInformation declared in engine.protocol",
        "class ExtractedToolCallInformation",
        engine_src,
    )
    for attr in ("tools_called", "tool_calls", "content"):
        run.expect_in(
            f"ExtractedToolCallInformation declares {attr}",
            f"{attr}:",
            engine_src,
        )


# --------------------------------------------------------------------
# Section 7: Patch 7 — extract_tool_calls_streaming_metrics
# --------------------------------------------------------------------


def section_7_streaming_metrics() -> None:
    run.section("7. monkey_patch_extract_tool_calls_streaming_metrics.py")
    patch_path = PATCH_DIR / "monkey_patch_extract_tool_calls_streaming_metrics.py"
    target_src = (
        VLLM_DIR / "vllm/tool_parsers/qwen3coder_tool_parser.py"
    ).read_text()

    # Counter-name guard (mirror of section 6's check; both patches
    # must agree on the same non-vllm-prefixed name).
    counter_name = patch_constant(patch_path, "_COUNTER_NAME")
    run.expect(
        f"streaming patch counter name agrees with non-streaming "
        f"(name={counter_name!r})",
        counter_name == "qwen3_coder_silent_tool_call_failures_total",
        "the two metrics patches must register the same counter name "
        "for the cooperative-discovery-on-collision path to work.",
    )
    run.expect(
        "streaming patch counter name does NOT contain 'vllm'",
        "vllm" not in counter_name,
        "would be deregistered by unregister_vllm_metrics().",
    )

    sig = vllm_function_signature(
        "vllm/tool_parsers/qwen3coder_tool_parser.py",
        "Qwen3CoderToolParser.extract_tool_calls_streaming",
    )
    run.expect_eq(
        "extract_tool_calls_streaming signature",
        sig,
        [
            "self",
            "previous_text",
            "current_text",
            "delta_text",
            "previous_token_ids",
            "current_token_ids",
            "delta_token_ids",
            "request",
        ],
    )

    # Streaming emit landmark.
    landmark = patch_constant(patch_path, "_STREAMING_EMIT_LANDMARK")
    run.expect_in(
        "streaming emit landmark present",
        landmark,
        target_src,
    )


# --------------------------------------------------------------------
# Section 8: Launcher self-consistency
# --------------------------------------------------------------------


def section_8_launcher() -> None:
    run.section("8. launch_with_patches.py — registry consistency")
    launcher_src = (PATCH_DIR / "launch_with_patches.py").read_text()

    # Every patch module is registered in _PATCH_MODULES.
    expected_modules = (
        "monkey_patch_qwen3_coder",
        "monkey_patch_hybrid_kv_allocator",
        "monkey_patch_extract_tool_calls_metrics",
        "monkey_patch_extract_tool_calls_streaming_metrics",
        "monkey_patch_reasoning_field_egress",
        "monkey_patch_reasoning_field_ingest",
        "monkey_patch_tool_call_in_think_rescue",
    )
    for name in expected_modules:
        run.expect_in(
            f"_PATCH_MODULES contains {name}",
            f'"{name}"',
            launcher_src,
        )
    # Every patch module has a verifier.
    for name in expected_modules:
        run.expect_in(
            f"_PATCH_VERIFICATION includes {name}",
            f'"{name}":',
            launcher_src,
        )

    # The new nested-egress check (load-bearing) is wired into
    # _verify_reasoning_field_egress.
    run.expect_in(
        "launcher's egress verifier constructs ChatCompletionResponse",
        "ChatCompletionResponse(",
        launcher_src,
    )
    run.expect_in(
        "launcher's egress verifier constructs ChatCompletionStreamResponse",
        "ChatCompletionStreamResponse(",
        launcher_src,
    )
    run.expect_in(
        "launcher's egress verifier exercises model_dump_json() (non-streaming)",
        "model_dump_json()",
        launcher_src,
    )
    run.expect_in(
        "launcher's egress verifier exercises model_dump_json(exclude_unset=True) (streaming)",
        "model_dump_json(exclude_unset=True)",
        launcher_src,
    )


# --------------------------------------------------------------------
# Section 9: No silent failures — anti-pattern checker
# --------------------------------------------------------------------


def section_9_no_silent_failures() -> None:
    """Static check: every patch must complain loudly on unexpected
    state. This section scans each patch's source for known
    silent-failure anti-patterns and refuses if any are found.

    The intent is to catch a future regression where a maintainer
    "loosens" a patch by silently catching an exception that should
    have refused, or by adding a logs-only fallback for a missing
    dependency.

    Each anti-pattern we forbid here corresponds to a real lesson
    from the v1 → v2 rewrite:

    * ``except Exception: pass`` — swallows real bugs, including the
      ``EgressPatchRefusedError`` family this whole repo's discipline
      relies on.
    * ``except ImportError`` followed by anything other than ``raise``
      — the patches' import-time deps (vllm, pydantic, prometheus_client)
      are all required; degrading silently to a feature-disabled mode
      defeats the whole reason an operator deployed the patch.
    * ``logger.debug(`` for anything that signals an UNEXPECTED runtime
      condition — DEBUG is invisible by default; surprises must surface
      at WARNING or higher.

    The two metrics patches (6, 7) are allowed exactly ONE
    ``except Exception`` each: the per-call instrumentation guard,
    which the docstring documents as "observability MUST NOT take
    down a request". That guard now logs at WARNING (not DEBUG); the
    section verifies that, too.
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

    # (c) Forbid logger.debug() in the metrics patches' instrumentation
    # path — they were the only files that previously logged at DEBUG
    # for an unexpected runtime exception, and we tightened both to
    # WARNING. Catch a future regression that re-loosens it.
    metrics_files = (
        "monkey_patch_extract_tool_calls_metrics.py",
        "monkey_patch_extract_tool_calls_streaming_metrics.py",
    )
    for fname in metrics_files:
        path = PATCH_DIR / fname
        src = path.read_text()
        run.expect_not_in(
            f"{fname}: no logger.debug() (use WARNING for unexpected runtime conditions)",
            "_logger.debug(",
            src,
        )

    # (d) Count REAL ``except Exception`` clauses via AST (not regex,
    # which over-counts string mentions inside docstrings). Each
    # patch's count must equal a hand-audited budget; any deviation
    # demands re-audit.
    expected_counts = {
        "monkey_patch_qwen3_coder.py": 0,
        "monkey_patch_hybrid_kv_allocator.py": 0,
        "monkey_patch_reasoning_field_egress.py": 0,
        "monkey_patch_reasoning_field_ingest.py": 0,
        "monkey_patch_tool_call_in_think_rescue.py": 1,  # property fallback
        # Metrics patches: outer per-call guard + inner logger-failure
        # guard (so observability never crashes a request).
        "monkey_patch_extract_tool_calls_metrics.py": 2,
        "monkey_patch_extract_tool_calls_streaming_metrics.py": 2,
    }
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
    """Static checks that don't require booting Docker:

    * ``sitecustomize.py`` exists at the repo root and parses as Python.
    * ``sitecustomize._PATCH_MODULES`` is a tuple of strings.
    * ``sitecustomize._PATCH_MODULES`` is byte-identical to
      ``launch_with_patches._PATCH_MODULES`` (the same drift check the
      launcher runs at boot, but also caught at static-test time).
    * The README's §8.2 docker run command includes the sitecustomize
      bind-mount and ``--host 127.0.0.1``.
    * The README's docker run includes ``-e PYTHONPATH=/opt/patches``
      (load-bearing for ``site.py`` to find sitecustomize at all) and
      ``--entrypoint python3`` (load-bearing for the launcher to run
      instead of the image's default ``vllm`` entrypoint).
    * Every entry in ``_PATCH_MODULES`` has a matching bind-mount line
      in the docker run command (catches the regression where a
      maintainer adds a patch to the launcher tuple but forgets to
      mount the source file).
    * The launcher itself is bind-mounted to ``/opt/patches/launch.py``.
    * The README's pinned commit string matches :data:`EXPECTED_COMMIT`.

    These checks correspond to the punch-list item C — without them, a
    maintainer can drift one of the invariants and only discover at
    deployment time (or worse, silently never).
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

    # README §8.2 must contain the sitecustomize bind-mount and the
    # --host 127.0.0.1 flag in the docker run block. Find the docker
    # run command (between ```bash ... ```), then check it contains
    # both strings.
    docker_run_re = re.compile(
        r"```bash\s*\n(docker run.*?)```", re.DOTALL
    )
    docker_run_match = docker_run_re.search(readme_src)
    run.expect(
        "README contains a `docker run` bash code block",
        docker_run_match is not None,
        "the §8.2 docker run command is missing or its fence is wrong",
    )
    if docker_run_match is not None:
        docker_run = docker_run_match.group(1)
        run.expect_in(
            "README §8.2 docker run includes sitecustomize bind-mount",
            "sitecustomize.py:/opt/patches/sitecustomize.py:ro",
            docker_run,
        )
        run.expect_in(
            "README §8.2 docker run includes --host 127.0.0.1",
            "--host 127.0.0.1",
            docker_run,
        )
        # The §8.2 command must NOT bind to 0.0.0.0 anywhere.
        run.expect_not_in(
            "README §8.2 docker run does NOT bind to 0.0.0.0",
            "--host 0.0.0.0",
            docker_run,
        )

        # PYTHONPATH=/opt/patches is load-bearing: without it, CPython's
        # site.py never finds /opt/patches/sitecustomize.py, and patch 2
        # (hybrid_kv_allocator) is silently dead in the spawned EngineCore.
        # Catch the regression of a maintainer dropping the env var.
        run.expect_in(
            "README §8.2 docker run includes -e PYTHONPATH=/opt/patches",
            "PYTHONPATH=/opt/patches",
            docker_run,
        )

        # --entrypoint python3 is load-bearing: without it, the image's
        # default ENTRYPOINT [vllm, serve] runs and the launcher is
        # never executed at all (no patches install).
        run.expect_in(
            "README §8.2 docker run includes --entrypoint python3",
            "--entrypoint python3",
            docker_run,
        )

        # The launcher itself must be bind-mounted into /opt/patches/launch.py
        # (the path the docker command then invokes). The launcher source
        # file's name in the repo is launch_with_patches.py; the in-container
        # path is /opt/patches/launch.py.
        run.expect_in(
            "README §8.2 docker run bind-mounts launch_with_patches.py "
            "at /opt/patches/launch.py",
            "launch_with_patches.py:/opt/patches/launch.py:ro",
            docker_run,
        )

        # Every entry in _PATCH_MODULES must have a corresponding bind-mount
        # line in the docker run command. Catches the regression where a
        # patch is added to the launcher tuple but the operator forgets
        # to add the bind-mount; without the mount the patch import
        # raises ImportError at PID 1 startup, but without the static check
        # the maintainer would only learn at deployment time.
        if isinstance(sitecustomize_modules, tuple):
            for module_name in sitecustomize_modules:
                expected_mount = (
                    f"{module_name}.py:/opt/patches/{module_name}.py:ro"
                )
                run.expect_in(
                    f"README §8.2 docker run bind-mounts {module_name}",
                    expected_mount,
                    docker_run,
                )

    # README's pinned commit string must match EXPECTED_COMMIT.
    run.expect_in(
        f"README references EXPECTED_COMMIT={EXPECTED_COMMIT[:12]}…",
        EXPECTED_COMMIT,
        readme_src,
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
        section_5_rescue,
        section_6_metrics,
        section_7_streaming_metrics,
        section_8_launcher,
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
