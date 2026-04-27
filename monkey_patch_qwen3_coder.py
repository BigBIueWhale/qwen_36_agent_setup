"""Strict, fail-loud runtime patch for vLLM tool-parser bug #39771.

Why this patch must exist: ``Qwen3CoderToolParser._parse_xml_function_call``
at ``vllm/tool_parsers/qwen3coder_tool_parser.py:236`` uses
``str.index(">")`` instead of ``str.find(">")``. On a truncated
``<parameter=NAME`` tag (no closing ``>``), ``str.index`` raises
``ValueError``; the outer ``try/except Exception`` at
``extract_tool_calls`` lines 320-324 catches it and returns
``tools_called=False, tool_calls=[]`` — collapsing **every well-formed
sibling tool call in the same response**. Sibling code at line 227
already uses the safe ``.find()/-1`` idiom; line 236 is an internal
inconsistency upstream PR #39772 acknowledges.

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce``.
Removal trigger: PR #39772 merges (the buggy-sentinel landmark check
below refuses against the fixed code).
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable, TypeAlias

import vllm  # noqa: F401  — availability landmark; must not be guarded
from vllm.logger import init_logger
from vllm.tool_parsers import qwen3coder_tool_parser as _qwen3coder_mod
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import find_tool_properties
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall

_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_BUGGY_SENTINEL: str = 'match_text.index(">")'
_EXPECTED_REGEX_ATTR: str = "tool_call_parameter_regex"
_UPSTREAM_REGEX_LANDMARK: str = "<parameter=(.*?)(?:</parameter>"
_PATCH_TAG: str = "qwen36-agent-setup-pr39772-backport"

ParseXmlFunctionCall: TypeAlias = Callable[[Any, str], Any]
_logger = init_logger(__name__)


class MonkeyPatchRefusedError(RuntimeError):
    """Precondition for the qwen3_coder tool-parser patch was violated.

    Raised at import time only; the patch either applies cleanly or
    the process does not come up.
    """


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise MonkeyPatchRefusedError(f"[{_PATCH_TAG}] refusing to patch: {msg}")


# Phase 1: Locate target.
_ParserCls = getattr(_qwen3coder_mod, "Qwen3CoderToolParser", None)
_require(
    _ParserCls is not None and inspect.isclass(_ParserCls),
    "Qwen3CoderToolParser missing or not a class.",
)
_require(
    issubclass(_ParserCls, ToolParser),
    "Qwen3CoderToolParser is no longer a subclass of ToolParser.",
)

# Phase 2: Landmark the buggy method.
_original: ParseXmlFunctionCall | None = getattr(
    _ParserCls, "_parse_xml_function_call", None
)
_require(
    _original is not None and callable(_original),
    "_parse_xml_function_call missing or not callable.",
)
try:
    _sig = inspect.signature(_original)  # type: ignore[arg-type]
    _original_src = inspect.getsource(_original)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect _parse_xml_function_call: {_exc!r}"
    ) from _exc
_require(
    list(_sig.parameters) == ["self", "function_call_str"],
    f"signature changed; got {list(_sig.parameters)!r}.",
)
_require(
    _BUGGY_SENTINEL in _original_src,
    f"buggy sentinel {_BUGGY_SENTINEL!r} not present. PR #39772 has likely "
    f"landed — delete this patch file and its mount.",
)

# Phase 3: MRO walk — load-bearing for inherited attributes.
# ``self.tool_call_parameter_regex`` is set in the SUBCLASS __init__;
# ``self.tools`` is set in BASE ``ToolParser.__init__``. A non-walking
# check would refuse incorrectly on the inherited assignment.
_init_sources: list[tuple[str, str]] = []
for _ancestor in _ParserCls.__mro__:
    if _ancestor is object or "__init__" not in _ancestor.__dict__:
        continue
    try:
        _init_sources.append(
            (_ancestor.__qualname__, inspect.getsource(_ancestor.__init__))
        )
    except (OSError, TypeError) as _exc:
        raise MonkeyPatchRefusedError(
            f"[{_PATCH_TAG}] cannot read {_ancestor.__qualname__}.__init__: {_exc!r}"
        ) from _exc
_require(_init_sources, f"no inspectable __init__ in MRO of {_ParserCls.__qualname__}.")

_combined_init_src = "\n".join(src for _, src in _init_sources)
_require(
    re.search(rf"\bself\.{re.escape(_EXPECTED_REGEX_ATTR)}\s*[:=]", _combined_init_src),
    f"no self.{_EXPECTED_REGEX_ATTR} assignment in MRO.",
)
# Regex SHAPE landmark must appear in the SUBCLASS source specifically.
_require(
    _UPSTREAM_REGEX_LANDMARK in _init_sources[0][1],
    f"regex shape {_UPSTREAM_REGEX_LANDMARK!r} not in {_init_sources[0][0]}.__init__.",
)
_require(
    re.search(r"\bself\.tools\s*[:=]", _combined_init_src),
    "no self.tools assignment in MRO; replacement would AttributeError.",
)


# Phase 4: The replacement.
def _parse_xml_function_call_strict(self: Any, function_call_str: str) -> Any:
    """Malformed ``<parameter=NAME`` returns ``None`` for the whole tool
    call (vs. raising ``ValueError`` which collapses all siblings).
    """
    if not isinstance(function_call_str, str):
        raise TypeError(
            f"[{_PATCH_TAG}] expected str, got {type(function_call_str).__name__!r}"
        )
    end_index = function_call_str.find(">")
    if end_index == -1:
        return None
    function_name = function_call_str[:end_index]
    param_config = find_tool_properties(self.tools, function_name)
    parameters_blob = function_call_str[end_index + 1 :]

    param_dict: dict[str, Any] = {}
    for match_text in self.tool_call_parameter_regex.findall(parameters_blob):
        if not isinstance(match_text, str):
            raise TypeError(
                f"[{_PATCH_TAG}] regex.findall yielded "
                f"{type(match_text).__name__!r}; upstream regex shape changed."
            )
        idx = match_text.find(">")
        if idx == -1:
            return None
        param_name = match_text[:idx]
        param_value = match_text[idx + 1 :]
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]
        param_dict[param_name] = self._convert_param_value(
            param_value, param_name, param_config, function_name
        )
    return ToolCall(
        type="function",
        function=FunctionCall(
            name=function_name,
            arguments=json.dumps(param_dict, ensure_ascii=False),
        ),
    )


_parse_xml_function_call_strict.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_parse_xml_function_call_strict.__wrapped_original__ = _original  # type: ignore[attr-defined]
_parse_xml_function_call_strict.__name__ = "_parse_xml_function_call"
_parse_xml_function_call_strict.__qualname__ = (
    f"{_ParserCls.__qualname__}._parse_xml_function_call"
)

# Phase 5: Install and verify — both dynamic and static lookup must agree.
_ParserCls._parse_xml_function_call = _parse_xml_function_call_strict
_require(
    getattr(_ParserCls._parse_xml_function_call, "__qwen36_patch__", None)
    == _PATCH_TAG,
    "post-install: tag absent via attribute access.",
)
_require(
    getattr(
        inspect.getattr_static(_ParserCls, "_parse_xml_function_call"),
        "__qwen36_patch__",
        None,
    )
    == _PATCH_TAG,
    "post-install: inspect.getattr_static disagrees.",
)

_logger.info(
    "[%s] applied: replaced %s.%s for vLLM commit %s "
    "(malformed <parameter= drops one call via None; siblings preserved).",
    _PATCH_TAG,
    _ParserCls.__module__,
    _ParserCls.__qualname__,
    _PINNED_VLLM_COMMIT,
)
