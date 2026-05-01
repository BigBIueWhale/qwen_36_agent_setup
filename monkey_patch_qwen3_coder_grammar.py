"""Strict, fail-loud server-side grammar constraint for Qwen3 tool emission.

Why this patch must exist
-------------------------

With ``--tool-call-parser qwen3_coder`` and the default ``tool_choice="auto"``,
``Qwen3CoderToolParser`` inherits :meth:`ToolParser.adjust_request` from
``vllm/tool_parsers/abstract_tool_parser.py:85-122``. The base method only
sets ``request.structured_outputs`` for ``tool_choice="required"`` or named
function tool_choice (via ``get_json_schema_from_tools``). For agentic
``tool_choice="auto"`` workflows — which is what Qwen Code CLI / Qwen-Agent
actually emit — the hook is a no-op. The model emits XML tool calls
**completely unconstrained** and a post-hoc parser extracts what it can.

**What the constraint enforces:**

* **Function-name pinning.** xgrammar's literal ``<function=NAME>``
  framing makes hallucinated names structurally unreachable — the
  function name on the wire is always one of the registered tools.
* **Tool-call framing integrity.** The ``<tool_call>`` / ``</tool_call>``
  and ``<function=…>`` / ``</function>`` markers are part of the
  structural_tag, so they cannot be elided or duplicated.
* The ``supports_required_and_named`` latent bug is fixed in the same
  override (see below).

**What this patch deliberately does NOT do: set
``request._grammar_from_tool_parser = True``.** Despite the field's
docstring at ``protocol.py:401-402`` reading parser-agnostic, its actual
contract is Mistral-format-specific — established in PR #39217 by a
Mistral engineer in the same commit that introduced the flag and its
readers. Setting it from a non-Mistral parser routes the request into
Mistral-only dispatch sites at
``vllm/entrypoints/openai/chat_completion/serving.py:823`` (streaming;
asserts ``isinstance(tool_parser, MistralToolParser)``) and ``:1407``
(non-streaming; calls ``MistralToolParser.build_non_streaming_tool_calls``
without an ``isinstance`` guard) — neither method exists on
``Qwen3CoderToolParser``. xgrammar enforcement is decoupled from the
flag: ``vllm/v1/structured_output/backend_xgrammar.py:92-106`` consumes
``request.structured_outputs.structural_tag`` directly. So our
function-name pinning and framing integrity properties are delivered
without setting the flag.

**What the constraint does NOT enforce: parameter-body shape.**
Qwen3.6 emits XML (``<parameter=KEY>VALUE</parameter>``) *inside* the
``<function=NAME>...</function>`` framing, and ``structural_tag.schema``
is JSON-oriented at xgrammar ``≥ 0.1.32, < 1.0.0`` — it does not
validate JSON Schemas against XML body text. We include the per-tool
``schema`` in the structural_tag payload regardless: it round-trips
cleanly through ``xgr.Grammar.from_structural_tag`` (Phase 6 verifier),
and a future xgrammar release that enforces XML bodies against JSON
Schema activates it without changing this patch. Until that lands, the
§7.5 detector and the post-hoc qwen3_coder parser remain responsible
for argument-shape correctness, and clients with strict-shape
requirements validate ``tool_calls[i].function.arguments`` themselves.

xgrammar's bitmask FSM stays dormant during
``<think>...</think>`` (``vllm/config/structured_outputs.py:41``
``enable_in_reasoning: bool = False``) and engages only after ``</think>``,
so thinking-mode reasoning remains unconstrained.

This patch ALSO sets ``Qwen3CoderToolParser.supports_required_and_named =
False``. Latent bug: with the inherited ``True``, ``tool_choice="required"``
runs the standard JSON-list path at ``engine/serving.py:646-665``
(``TypeAdapter(list[FunctionDefinition]).validate_json(content)``) on
the model's XML output; the validation fails, ``contextlib.suppress(
ValidationError)`` swallows the error, and the response is silently empty.
GLM4 has the same XML-shape problem and ships
``supports_required_and_named = False``; Qwen3 does not.

This patch does NOT fix the ``<tool_call>``-in-``<think>`` model OOD case
(README §6.5 / §7.5 detector). That emission happens BEFORE ``</think>``
where the FSM is dormant by design; the §7.5 detector remains correct
and complementary.

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce``.
xgrammar pin: ``>= 0.1.32, < 1.0.0`` (``vllm/requirements/common.txt``);
both ``StructuralTagItem`` and ``Grammar.from_structural_tag`` are
load-bearing-verified at import time on a single-tool synthetic request.

Removal triggers
----------------

* vLLM merges ``Qwen3CoderToolParser.adjust_request`` upstream (the parser
  ships its own structural_tag), OR
* vLLM ships a ``qwen3_xml`` parser with grammar enforcement (the
  community recommendation per PR #25028).

Either makes this patch redundant; delete the file and its mount.

Independent latent upstream bug — NOT this patch's job to fix
-------------------------------------------------------------

``vllm/entrypoints/openai/chat_completion/serving.py:1407`` reads
``use_mistral_tool_parser = request._grammar_from_tool_parser`` and
dispatches into ``MistralToolParser.build_non_streaming_tool_calls``
without an ``isinstance`` guard. The streaming-side dispatch at line 823
HAS the guard (the ``assert``); the non-streaming path does not. This is
upstream's own internal inconsistency — an analogous correct check
already lives at ``vllm/entrypoints/openai/engine/serving.py:619``
(``is_mistral_tool_parser(tool_parser_cls) AND
request._grammar_from_tool_parser``). With our patch's "do not set the
flag" stance, line 1407 is never entered for our requests, so the bug
is unweaponized in this deployment. Worth filing upstream as a separate
PR; not blocking this patch.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import vllm  # noqa: F401  — availability landmark; must not be guarded
import xgrammar as xgr

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers import qwen3coder_tool_parser as _qwen3coder_mod
from vllm.tool_parsers.abstract_tool_parser import ToolParser


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-qwen3-coder-grammar-v1"

# Source landmark — substring required in the BASE class's adjust_request
# body. Catches an upstream refactor of the extension point's contract
# before this patch silently overrides a different shape.
_BASE_LANDMARK: str = "if not request.tools:"

# The base class still imposes the no-op for tool_choice="auto" (only
# fires when get_json_schema_from_tools returns non-None). If the base
# starts setting structured_outputs unconditionally we must re-audit.
_BASE_GET_JSON_LANDMARK: str = "get_json_schema_from_tools("

_EXPECTED_PARAMS: list[str] = ["self", "request"]


_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class Qwen3CoderGrammarPatchRefusedError(RuntimeError):
    """A precondition for the qwen3_coder grammar wrapper was violated.

    Raised at import time only. The patch either applies cleanly or the
    process does not come up — there is no half-installed path. A wrong
    structural_tag shape silently produces a parser that cannot compile
    requests at runtime; refusing to boot is strictly safer.
    """


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise Qwen3CoderGrammarPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {msg}"
        )


# --------------------------------------------------------------------
# Phase 1: Locate target class and verify class hierarchy.
# --------------------------------------------------------------------

_ParserCls = getattr(_qwen3coder_mod, "Qwen3CoderToolParser", None)
_require(
    _ParserCls is not None and inspect.isclass(_ParserCls),
    "Qwen3CoderToolParser missing or not a class.",
)
_require(
    issubclass(_ParserCls, ToolParser),
    "Qwen3CoderToolParser is no longer a subclass of ToolParser.",
)


# --------------------------------------------------------------------
# Phase 2: Verify the inherited adjust_request shape we override.
# --------------------------------------------------------------------
# The class does NOT define its own adjust_request; it inherits the base.
# If a future Qwen3CoderToolParser ships its own override, we refuse —
# the upstream override is the trigger for deleting this patch.

_subclass_dict_adjust = _ParserCls.__dict__.get("adjust_request", None)
_require(
    _subclass_dict_adjust is None,
    "Qwen3CoderToolParser already declares its own adjust_request "
    "(via the SUBCLASS dict, not the inherited base). Upstream has "
    "shipped the override this patch was designed to provide; delete "
    "this patch file and its mount.",
)

_base_adjust = getattr(ToolParser, "adjust_request", None)
_require(
    _base_adjust is not None and callable(_base_adjust),
    "ToolParser.adjust_request missing or not callable; the official "
    "extension point this patch overrides has been removed.",
)
try:
    _base_sig = inspect.signature(_base_adjust)  # type: ignore[arg-type]
    _base_src = inspect.getsource(_base_adjust)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise Qwen3CoderGrammarPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect ToolParser.adjust_request: {_exc!r}"
    ) from _exc

_require(
    list(_base_sig.parameters) == _EXPECTED_PARAMS,
    f"ToolParser.adjust_request signature drifted; expected "
    f"{_EXPECTED_PARAMS!r}, got {list(_base_sig.parameters)!r}.",
)
_require(
    _BASE_LANDMARK in _base_src,
    f"landmark {_BASE_LANDMARK!r} missing from ToolParser.adjust_request "
    f"source — upstream restructured the extension point; re-audit "
    f"before bumping the pinned commit.",
)
_require(
    _BASE_GET_JSON_LANDMARK in _base_src,
    f"landmark {_BASE_GET_JSON_LANDMARK!r} missing from "
    f"ToolParser.adjust_request source — base no-op contract for "
    f"tool_choice='auto' may no longer hold; re-audit.",
)


# --------------------------------------------------------------------
# Phase 3: Verify the supports_required_and_named class flag we flip.
# --------------------------------------------------------------------

_inherited_flag = getattr(_ParserCls, "supports_required_and_named", None)
_require(
    _inherited_flag is True,
    f"Qwen3CoderToolParser.supports_required_and_named is "
    f"{_inherited_flag!r}, expected True (inherited from ToolParser). "
    f"A subclass override has appeared; the latent JSON-list-path bug "
    f"may have been fixed upstream.",
)
_require(
    "supports_required_and_named" not in _ParserCls.__dict__,
    "Qwen3CoderToolParser already declares its own "
    "supports_required_and_named (via the SUBCLASS dict). Upstream has "
    "addressed the JSON-list-path latent bug; delete this patch file "
    "and its mount.",
)


# --------------------------------------------------------------------
# Phase 4: Verify the StructuredOutputsParams shape we construct.
# --------------------------------------------------------------------
# A drift in field names ("structural_tag" rename or removal) would
# silently no-op our install at request time.

_so_field_names: set[str] = set()
if hasattr(StructuredOutputsParams, "__dataclass_fields__"):
    _so_field_names = set(StructuredOutputsParams.__dataclass_fields__)
elif hasattr(StructuredOutputsParams, "model_fields"):
    _so_field_names = set(StructuredOutputsParams.model_fields)
_require(
    "structural_tag" in _so_field_names,
    "StructuredOutputsParams.structural_tag field is missing; "
    "xgrammar pipeline shape changed.",
)


# --------------------------------------------------------------------
# Phase 5: Verify the request schema fields we read.
# --------------------------------------------------------------------

_req_field_names = set(getattr(ChatCompletionRequest, "model_fields", {}))
for _required in (
    "tools",
    "tool_choice",
    "structured_outputs",
    "response_format",
):
    _require(
        _required in _req_field_names,
        f"ChatCompletionRequest no longer declares {_required!r}; "
        f"the patch's per-request gate is no longer expressible.",
    )

_require(
    "_grammar_from_tool_parser" in getattr(ChatCompletionRequest, "__private_attributes__", {}),
    "ChatCompletionRequest._grammar_from_tool_parser PrivateAttr is "
    "missing; the Mistral-grammar dispatch plumbing has been redesigned "
    "upstream and our 'do not set it from a non-Mistral parser' invariant "
    "may no longer be the right choice. Re-audit before bumping the "
    "pinned commit.",
)


# --------------------------------------------------------------------
# Phase 6: Verify xgrammar exposes the load-bearing entry points.
# --------------------------------------------------------------------
# Two surfaces matter: vLLM's own backend at
# vllm/v1/structured_output/backend_xgrammar.py uses
# xgr.StructuralTagItem(...) + xgr.Grammar.from_structural_tag(tags,
# triggers) for validation, and compile_structural_tag(tags, triggers)
# for compile. If either surface vanishes our request-time payload
# becomes uncompile-able.

for _name in ("StructuralTagItem", "Grammar"):
    _require(
        hasattr(xgr, _name),
        f"xgrammar does not expose {_name!r}; xgrammar version drift.",
    )
_require(
    callable(getattr(xgr.Grammar, "from_structural_tag", None)),
    "xgrammar.Grammar.from_structural_tag is missing or not callable; "
    "xgrammar version drift.",
)


# --------------------------------------------------------------------
# Phase 7: Build the structural_tag payload and the wrapper.
# --------------------------------------------------------------------


def _build_structural_tag(tools: list[Any]) -> str:
    """Render the tools list to xgrammar's deprecated-but-still-supported
    ``structures``+``triggers`` JSON shape.

    The legacy shape is what
    ``vllm/v1/structured_output/backend_xgrammar.py:92-106`` accepts and
    routes into ``xgr.StructuralTagItem`` (with ``schema=json.dumps(...)``
    pre-applied at backend time). Per-tool ``begin``/``end`` strings
    bracket the JSON-schema body so xgrammar enforces the literal
    ``<tool_call>\\n<function=NAME>\\n...\\n</function>\\n</tool_call>``
    framing Qwen3.6 emits.

    A tool whose ``parameters`` is None contributes an empty-object
    schema; xgrammar accepts ``{}`` as "any JSON object".
    """
    structures: list[dict[str, Any]] = []
    for t in tools:
        fn = t.function
        params = fn.parameters if fn.parameters is not None else {}
        structures.append(
            {
                "begin": f"<tool_call>\n<function={fn.name}>\n",
                "schema": params,
                "end": "\n</function>\n</tool_call>",
            }
        )
    return json.dumps({"structures": structures, "triggers": ["<tool_call>"]})


def _adjust_request_qwen3_grammar(
    self: Any,
    request: ChatCompletionRequest | ResponsesRequest,
) -> ChatCompletionRequest | ResponsesRequest:
    """Override of ``Qwen3CoderToolParser.adjust_request`` that installs an
    xgrammar ``structural_tag`` constraint when the model would otherwise
    emit XML tool calls under ``tool_choice="auto"``.

    Skipped if the client already supplied a structured-outputs
    constraint or a response_format; never trampling explicit client
    intent. Skipped on ResponsesRequest (the surface uses ``request.text``
    rather than ``request.structured_outputs``).

    **We do NOT set ``request._grammar_from_tool_parser = True``.** Despite
    the field's docstring at ``protocol.py:401-402`` reading parser-agnostic
    ("CAUTION: Should only be set by ``ToolParser.adjust_request``"), its
    actual contract — established in PR #39217 by a Mistral engineer in the
    same commit that introduced the flag and its readers — is
    Mistral-format-specific. Setting it routes the request through
    Mistral-only dispatch sites at
    ``vllm/entrypoints/openai/chat_completion/serving.py:823, 1407`` that
    call ``MistralToolParser.extract_maybe_reasoning_and_tool_streaming``
    and ``MistralToolParser.build_non_streaming_tool_calls`` — neither
    method exists on ``Qwen3CoderToolParser``. The streaming dispatch's
    ``assert isinstance(tool_parser, MistralToolParser)`` is the authored
    contract, not a defensive scaffold; the non-streaming dispatch lacks
    even that assertion (a separate latent upstream bug, see README §6.10
    and §12 item 16).

    The ``structural_tag`` enforcement is **independent of the flag**:
    ``vllm/v1/structured_output/backend_xgrammar.py:92-106`` consumes
    ``request.structured_outputs.structural_tag`` directly, with no
    consultation of ``_grammar_from_tool_parser``. So omitting the flag
    write loses none of the function-name pinning or framing-integrity
    enforcement this patch promises in §7.7.

    The ``_grammar_from_tool_parser`` PrivateAttr's existence is still
    validated at Phase 5 above as a landmark — its removal upstream
    would mean the Mistral-grammar plumbing changed, and our "do not
    set it" invariant must be re-audited.
    """
    if not request.tools:
        return request
    if isinstance(request, ResponsesRequest):
        # Different field shape (request.text); leave to base behaviour.
        return request
    if request.tool_choice == "none":
        return request
    if request.structured_outputs is not None:
        return request
    if request.response_format is not None:
        return request

    request.structured_outputs = StructuredOutputsParams(
        structural_tag=_build_structural_tag(request.tools)
    )
    return request


_adjust_request_qwen3_grammar.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_adjust_request_qwen3_grammar.__wrapped_original__ = _base_adjust  # type: ignore[attr-defined]
_adjust_request_qwen3_grammar.__name__ = "adjust_request"
_adjust_request_qwen3_grammar.__qualname__ = (
    f"{_ParserCls.__qualname__}.adjust_request"
)
_adjust_request_qwen3_grammar.__module__ = _ParserCls.__module__


# --------------------------------------------------------------------
# Phase 8: Install both the method override and the class-flag flip,
# then verify both via dynamic and static lookup.
# --------------------------------------------------------------------

_ParserCls.adjust_request = _adjust_request_qwen3_grammar
_ParserCls.supports_required_and_named = False

_installed_dynamic = getattr(_ParserCls, "adjust_request")
_require(
    getattr(_installed_dynamic, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install: tag absent via attribute access on "
    "Qwen3CoderToolParser.adjust_request.",
)
_installed_static = inspect.getattr_static(_ParserCls, "adjust_request")
_require(
    getattr(_installed_static, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install: inspect.getattr_static disagrees with attribute "
    "access on Qwen3CoderToolParser.adjust_request; a metaclass shim "
    "or descriptor is shadowing the install.",
)

_require(
    _ParserCls.supports_required_and_named is False,
    f"post-install: Qwen3CoderToolParser.supports_required_and_named "
    f"is {_ParserCls.supports_required_and_named!r}, expected False.",
)
_require(
    inspect.getattr_static(_ParserCls, "supports_required_and_named") is False,
    "post-install: inspect.getattr_static disagrees with attribute "
    "access on Qwen3CoderToolParser.supports_required_and_named.",
)


# --------------------------------------------------------------------
# Phase 9: Behavioural verification on real ChatCompletionRequest +
# Qwen3CoderToolParser instances. Load-bearing — a tag-only check passes
# even when the wrapper produces an xgrammar-uncompilable payload.
# --------------------------------------------------------------------
# The parser's __init__ requires only a model_tokenizer whose get_vocab()
# returns non-None ids for <tool_call>/</tool_call>. The launcher's
# patch-2 verifier already uses this exact shape; we re-use it here.


class _TokenizerMock:
    """Minimal model_tokenizer surface — only what
    ``Qwen3CoderToolParser.__init__`` actually reads."""

    _vocab = {"<tool_call>": 1_000_001, "</tool_call>": 1_000_002}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def __bool__(self) -> bool:
        return True


_tool_calc = ChatCompletionToolsParam(
    type="function",
    function=FunctionDefinition(
        name="calculator",
        description="Evaluate a math expression.",
        parameters={
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    ),
)
_parser_probe = _ParserCls(_TokenizerMock(), tools=[_tool_calc])


# Case 1: client sent tools, no structured_outputs, no response_format,
# tool_choice defaults to "auto" — patch must install a structural_tag.
_req_auto = ChatCompletionRequest(
    model="probe",
    messages=[{"role": "user", "content": "what is 2+2?"}],
    tools=[_tool_calc],
)
_returned = _parser_probe.adjust_request(_req_auto)
_require(
    _returned is _req_auto,
    "Phase 9 case 1: adjust_request returned a different object "
    "(expected in-place mutation + return self).",
)
_require(
    _req_auto.structured_outputs is not None,
    "Phase 9 case 1: structured_outputs is None after adjust_request; "
    "the precondition gate did not fire.",
)
_tag_str = _req_auto.structured_outputs.structural_tag
_require(
    isinstance(_tag_str, str) and _tag_str,
    f"Phase 9 case 1: structural_tag is {_tag_str!r}; expected non-empty str.",
)
_tag_obj = json.loads(_tag_str)
_require(
    isinstance(_tag_obj, dict)
    and "structures" in _tag_obj
    and "triggers" in _tag_obj,
    f"Phase 9 case 1: structural_tag JSON shape is {_tag_obj!r}; "
    f"expected {{'structures': [...], 'triggers': [...]}}.",
)
_require(
    isinstance(_tag_obj["structures"], list) and len(_tag_obj["structures"]) == 1,
    f"Phase 9 case 1: structures list length is "
    f"{len(_tag_obj['structures'])!r}; expected 1.",
)
_first = _tag_obj["structures"][0]
_require(
    _first["begin"] == "<tool_call>\n<function=calculator>\n"
    and _first["end"] == "\n</function>\n</tool_call>",
    f"Phase 9 case 1: structure begin/end tags drifted: {_first!r}.",
)
_require(
    _tag_obj["triggers"] == ["<tool_call>"],
    f"Phase 9 case 1: triggers list is {_tag_obj['triggers']!r}; "
    f"expected ['<tool_call>'].",
)
# Confirm the inverse — we deliberately do NOT set the Mistral-only
# ``_grammar_from_tool_parser`` flag. See the wrapper docstring for why.
_require(
    getattr(_req_auto, "_grammar_from_tool_parser", None) is False,
    "Phase 9 case 1: _grammar_from_tool_parser flag was set unexpectedly. "
    "This is a Mistral-only flag (PR #39217); setting it from a "
    "non-Mistral parser routes the request into Mistral-specific dispatch "
    "sites at chat_completion/serving.py:823 and :1407 that call methods "
    "Qwen3CoderToolParser does not implement. The patch was changed to "
    "stop setting this flag; if you re-introduced the write, undo it.",
)

# Round-trip through xgrammar — load-bearing. Either compile_structural_tag
# entry point (vLLM's own backend uses both) must accept what we produced.
_xgr_tags = [
    xgr.StructuralTagItem(
        begin=s["begin"], schema=json.dumps(s["schema"]), end=s["end"]
    )
    for s in _tag_obj["structures"]
]
xgr.Grammar.from_structural_tag(_xgr_tags, _tag_obj["triggers"])

# Case 2: empty tools list — precondition gate must fire BEFORE we touch
# structured_outputs (negative control: proves the patch is not a
# blanket override).
_req_empty = ChatCompletionRequest(
    model="probe",
    messages=[{"role": "user", "content": "hi"}],
)
_req_empty.tools = []  # explicit; bypasses the auto-default to None
_returned_empty = _parser_probe.adjust_request(_req_empty)
_require(
    _req_empty.structured_outputs is None,
    f"Phase 9 case 2: structured_outputs is "
    f"{_req_empty.structured_outputs!r}; expected None on empty tools.",
)

# Case 3: client supplied structured_outputs already — patch must NOT
# trample it (negative control: proves we honour explicit client intent).
_req_explicit = ChatCompletionRequest(
    model="probe",
    messages=[{"role": "user", "content": "what is 2+2?"}],
    tools=[_tool_calc],
    structured_outputs=StructuredOutputsParams(json='{"type":"object"}'),
)
_pre_so = _req_explicit.structured_outputs
_parser_probe.adjust_request(_req_explicit)
_require(
    _req_explicit.structured_outputs is _pre_so,
    "Phase 9 case 3: explicit client structured_outputs was overwritten; "
    "the patch is rewriting client intent.",
)

# Case 4: tool_choice="none" — patch must skip even with non-empty tools.
_req_none = ChatCompletionRequest(
    model="probe",
    messages=[{"role": "user", "content": "what is 2+2?"}],
    tools=[_tool_calc],
    tool_choice="none",
)
_parser_probe.adjust_request(_req_none)
_require(
    _req_none.structured_outputs is None,
    f"Phase 9 case 4: structured_outputs is "
    f"{_req_none.structured_outputs!r}; expected None on tool_choice='none'.",
)


_logger.info(
    "[%s] applied: overrode %s.adjust_request and flipped "
    "supports_required_and_named=False for vLLM commit %s "
    "(per-request xgrammar structural_tag installs on tool_choice='auto'; "
    "thinking-mode FSM dormant by enable_in_reasoning=False default).",
    _PATCH_TAG,
    _ParserCls.__qualname__,
    _PINNED_VLLM_COMMIT,
)
