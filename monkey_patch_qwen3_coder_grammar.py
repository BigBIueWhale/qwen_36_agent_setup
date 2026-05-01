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
* **Parameter-body shape (Qwen3 hybrid XML+JSON).** Each tool's body is
  enforced to the trained-shape ``<parameter=KEY>VALUE</parameter>`` XML
  framing, with VALUE encoded per-type per Qwen3-Coder's chat template:
  raw text (no quoting) for ``type:"string"`` properties, JSON literals
  (numbers, ``true``/``false``, arrays, objects) for everything else.
  This hybrid is what the chat template at ``chat_template.jinja:122``
  renders for historical assistant tool calls
  (``args_value | string if args_value is string else args_value | tojson``)
  and what the post-hoc ``Qwen3CoderToolParser._parse_xml_function_call``
  reads back. xgrammar's ``JSONSchemaFormat(style="qwen_xml")``
  (``xgrammar/structural_tag.py:35-67``; shipped in xgrammar 0.1.32, the
  lower bound of vLLM's ``>=0.1.32,<1.0.0`` pin in
  ``vllm/requirements/common.txt``) implements both halves: literal
  ``<parameter=`` / ``</parameter>`` framing is mandatory, and the body
  content type-checks against the schema (``"42"`` is rejected for
  ``type:integer``, ``42`` is accepted; for ``type:string`` both raw and
  JSON-quoted text are accepted because both are valid string content).
  The FSM rejects an empty body, rejects JSON-object-shaped bodies
  (``{}`` after the ``<function=NAME>`` opener), rejects missing
  ``required`` parameters, and rejects type-mismatched non-string values.
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
function-name pinning, framing integrity, and qwen_xml body
enforcement are delivered without setting the flag.

**v2 history (May 2026): the body-style fix.** Earlier (v1) versions of
this patch built the structural_tag using xgrammar's deprecated
``structures``+``triggers`` JSON shape. That shape silently routed
through ``xgrammar/structural_tag.py:506-535``'s
``StructuralTag.from_legacy_structural_tag`` converter, which hardcodes
``JSONSchemaFormat(style="json")`` — the legacy API has no way to
express XML body enforcement. xgrammar consequently forced the model
into standard JSON output between ``<function=NAME>`` and
``</function>``, contradicting Qwen3-Coder's training-shape XML and
the post-hoc ``Qwen3CoderToolParser._parse_xml_function_call`` regex
which scans for ``<parameter=KEY>VALUE</parameter>``. The parser
matched zero ``<parameter=`` markers in JSON output, set
``param_dict={}``, and emitted ``arguments="{}"`` for every single tool
call. Self-aware re-prompting could not escape — the FSM masked every
non-``{`` token. v2 emits the modern
``StructuralTag``+``triggered_tags`` shape with explicit
``style="qwen_xml"``, recovered the body enforcement, and added a
Phase 9 byte-walk regression guard that asserts the matcher rejects
``{`` at body-open and accepts ``<parameter=…>…</parameter>``.

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
_PATCH_TAG: str = "qwen36-agent-setup-qwen3-coder-grammar-v2"

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
# We compile the modern StructuralTag JSON-string shape via
# xgr.Grammar.from_structural_tag(spec_str) and exercise it through
# xgr.GrammarCompiler + xgr.GrammarMatcher in Phase 9. If any of these
# surfaces vanish (xgrammar major version bump, name rename) our
# request-time payload becomes uncompile-able and we refuse to install.
#
# `xgr.StructuralTagItem` is the legacy-shape factory we no longer use,
# but its presence is the xgrammar-version landmark for "structural-tag
# entry points are stable" — its disappearance correlates strongly with
# a renamed/redesigned modern surface and is still worth refusal-checking.

for _name in ("StructuralTagItem", "Grammar", "GrammarCompiler", "GrammarMatcher", "TokenizerInfo"):
    _require(
        hasattr(xgr, _name),
        f"xgrammar does not expose {_name!r}; xgrammar version drift.",
    )
_require(
    callable(getattr(xgr.Grammar, "from_structural_tag", None)),
    "xgrammar.Grammar.from_structural_tag is missing or not callable; "
    "xgrammar version drift.",
)
_require(
    callable(getattr(xgr.GrammarCompiler, "compile_structural_tag", None)),
    "xgrammar.GrammarCompiler.compile_structural_tag is missing or not "
    "callable; xgrammar version drift. The Phase 9 byte-walk regression "
    "guard depends on this method.",
)


# --------------------------------------------------------------------
# Phase 7: Build the structural_tag payload and the wrapper.
# --------------------------------------------------------------------


def _build_structural_tag(tools: list[Any]) -> str:
    """Render the tools list to xgrammar's modern ``StructuralTag`` +
    ``triggered_tags`` shape with ``JSONSchemaFormat(style="qwen_xml")``.

    Per-tool ``begin``/``end`` strings bracket the body so xgrammar
    enforces the literal
    ``<tool_call>\\n<function=NAME>\\n…\\n</function>\\n</tool_call>``
    framing. Inside that framing, ``style="qwen_xml"`` compiles the
    JSON Schema into a body grammar that requires
    ``<parameter=KEY>VALUE</parameter>`` for each property, with type
    and ``required`` enforcement. This is exactly the shape Qwen3-Coder
    is trained to emit and the shape ``Qwen3CoderToolParser._parse_xml_function_call``
    (``vllm/tool_parsers/qwen3coder_tool_parser.py:225-253``) reads.

    Why the modern shape, not the legacy ``structures``+``triggers``
    one: the legacy converter at
    ``xgrammar/structural_tag.py:506-535`` (``from_legacy_structural_tag``)
    hardcodes ``JSONSchemaFormat(style="json")``, with no way to override.
    A legacy payload silently forces the model into JSON body output —
    valid JSON, but the qwen3_coder XML parser then matches zero
    ``<parameter=`` markers in it and emits ``arguments="{}"`` for every
    tool call. v1 of this patch hit that pothole; v2 sidesteps it by
    using the modern shape, which routes through
    ``vllm/v1/structured_output/backend_xgrammar.py:106``'s
    ``compiler.compile_structural_tag(grammar_spec)`` directly.

    A tool whose ``parameters`` is ``None`` contributes an empty-object
    schema; xgrammar accepts an empty body (no ``<parameter=…>``
    markers) for it.
    """
    tags: list[dict[str, Any]] = []
    for t in tools:
        fn = t.function
        params = fn.parameters if fn.parameters is not None else {}
        tags.append(
            {
                "type": "tag",
                "begin": f"<tool_call>\n<function={fn.name}>\n",
                "content": {
                    "type": "json_schema",
                    "json_schema": params,
                    "style": "qwen_xml",
                },
                "end": "\n</function>\n</tool_call>",
            }
        )
    return json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": tags,
                "at_least_one": False,
                "stop_after_first": False,
            },
        }
    )


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
    write loses none of the function-name pinning, framing integrity,
    or qwen_xml body enforcement this patch promises in §7.7.

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

# Refuse the legacy shape outright. The legacy 'structures'+'triggers'
# payload silently routes through xgrammar's from_legacy_structural_tag
# (xgrammar/structural_tag.py:506-535) which hardcodes
# JSONSchemaFormat(style='json') — forcing the model into JSON body
# output that vLLM's qwen3_coder XML parser cannot read, producing
# arguments='{}' for every tool call. v1 of this patch hit that pothole.
_require(
    "structures" not in _tag_obj,
    "Phase 9 case 1: structural_tag carries the legacy 'structures' key. "
    "The legacy shape forces JSON body enforcement via "
    "from_legacy_structural_tag → JSONSchemaFormat(style='json'), and "
    "Qwen3-Coder emits XML — the mismatch produces arguments='{}' for "
    "every tool call. _build_structural_tag must emit the modern "
    "{'type':'structural_tag','format':{'type':'triggered_tags',...}} "
    "shape with style='qwen_xml' on each tag's content.",
)

# Top-level shape: modern StructuralTag.
_require(
    isinstance(_tag_obj, dict)
    and _tag_obj.get("type") == "structural_tag"
    and isinstance(_tag_obj.get("format"), dict),
    f"Phase 9 case 1: structural_tag JSON shape is {_tag_obj!r}; expected "
    f"{{'type':'structural_tag','format':{{'type':'triggered_tags',...}}}}.",
)
_fmt = _tag_obj["format"]
_require(
    _fmt.get("type") == "triggered_tags",
    f"Phase 9 case 1: format.type is {_fmt.get('type')!r}; "
    f"expected 'triggered_tags'.",
)
_require(
    _fmt.get("triggers") == ["<tool_call>"],
    f"Phase 9 case 1: triggers list is {_fmt.get('triggers')!r}; "
    f"expected ['<tool_call>'].",
)
_tags = _fmt.get("tags")
_require(
    isinstance(_tags, list) and len(_tags) == 1,
    f"Phase 9 case 1: format.tags length is "
    f"{len(_tags) if isinstance(_tags, list) else None!r}; expected 1.",
)
_first = _tags[0]
_require(
    _first.get("type") == "tag"
    and _first.get("begin") == "<tool_call>\n<function=calculator>\n"
    and _first.get("end") == "\n</function>\n</tool_call>",
    f"Phase 9 case 1: tag begin/end framing drifted: {_first!r}.",
)
_content = _first.get("content")
_require(
    isinstance(_content, dict)
    and _content.get("type") == "json_schema"
    and _content.get("style") == "qwen_xml",
    f"Phase 9 case 1: tag content is {_content!r}; expected "
    f"{{'type':'json_schema','json_schema':...,'style':'qwen_xml'}}. "
    f"style='json' (the JSONSchemaFormat default at "
    f"xgrammar/structural_tag.py:35) compiles to a JSON-body grammar "
    f"that Qwen3-Coder's XML parser cannot read; only 'qwen_xml' "
    f"compiles to the <parameter=KEY>VALUE</parameter> body grammar "
    f"the parser is built for.",
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

# Round-trip through xgrammar — load-bearing.
# vLLM's backend at backend_xgrammar.py:106 calls compiler.compile_structural_tag(grammar_spec)
# on the modern-shape JSON string directly. Grammar.from_structural_tag is
# the equivalent surface for build-time validation.
xgr.Grammar.from_structural_tag(_tag_str)

# Functional regression guard — the load-bearing functional check.
# v1 of this patch's payload also compiled cleanly through
# Grammar.from_structural_tag; only this byte-walk catches "compiles, but
# enforces the wrong body shape." We construct a synthetic TokenizerInfo
# with a 256-byte vocab so the matcher accepts any UTF-8 byte at the
# tokenizer level — leaving the grammar as the sole gate.
_synthetic_tokenizer_info = xgr.TokenizerInfo(
    encoded_vocab=[bytes([_i]) for _i in range(256)],
)
_synthetic_compiler = xgr.GrammarCompiler(_synthetic_tokenizer_info)
_compiled_ctx = _synthetic_compiler.compile_structural_tag(_tag_str)

# Positive: a canonical XML body for the calculator tool MUST be accepted
# byte-by-byte. If the matcher rejects any byte, qwen_xml style isn't
# actually enforced — most likely an xgrammar version drop or a payload
# malformation we missed in the shape checks above.
_canonical_xml = (
    "<tool_call>\n<function=calculator>\n"
    "<parameter=expr>\n2+2\n</parameter>\n"
    "</function>\n</tool_call>"
)
_matcher = xgr.GrammarMatcher(_compiled_ctx)
for _i, _ch in enumerate(_canonical_xml):
    _require(
        _matcher.accept_string(_ch),
        f"Phase 9 regression guard: FSM rejected canonical Qwen XML body "
        f"byte at index {_i} (char {_ch!r}) of "
        f"{_canonical_xml!r}. The compiled grammar does not accept the "
        f"training-shape body Qwen3-Coder emits — "
        f"JSONSchemaFormat(style='qwen_xml') is not in effect. Either "
        f"xgrammar dropped/renamed the qwen_xml style or "
        f"_build_structural_tag is producing a malformed payload.",
    )

# Negative: a body that opens with '{' (the JSON path the v1 bug forced)
# MUST be rejected. We walk up to the body-open boundary and confirm that
# '{' is masked.
_matcher = xgr.GrammarMatcher(_compiled_ctx)
_pre_open = "<tool_call>\n<function=calculator>\n"
for _ch in _pre_open:
    _require(
        _matcher.accept_string(_ch),
        f"Phase 9 regression guard: FSM unexpectedly rejected the "
        f"pre-body framing byte {_ch!r}. The grammar's tool-call envelope "
        f"is malformed; this should never happen if the canonical-XML "
        f"walk above passed.",
    )
_require(
    not _matcher.accept_string("{"),
    "Phase 9 regression guard: FSM accepted '{' immediately after "
    "<function=calculator>\\n — JSON body shape is allowed by the grammar, "
    "meaning JSONSchemaFormat(style='qwen_xml') is NOT in effect. The "
    "qwen3_coder parser reads only XML; with JSON body it produces "
    "arguments='{}' for every tool call. This is the exact bug v2 was "
    "written to fix; refusing to install.",
)

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
    "(per-request xgrammar structural_tag with JSONSchemaFormat(style='qwen_xml') "
    "installs on tool_choice='auto', enforcing function-name pinning, framing, "
    "and Qwen XML parameter-body shape; thinking-mode FSM dormant by "
    "enable_in_reasoning=False default; Phase 9 byte-walks the matcher to "
    "prove '{' body-open is rejected and canonical <parameter=...>...</parameter> "
    "is accepted).",
    _PATCH_TAG,
    _ParserCls.__qualname__,
    _PINNED_VLLM_COMMIT,
)
