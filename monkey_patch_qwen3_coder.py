"""
Client patch 4 (see README.md §7.4 and §8.2).

Pre-imports vLLM's qwen3_coder tool parser and replaces its
_parse_xml_function_call method with a version that mirrors upstream PR #39772:
changes str.index(">") to str.find(">") and skips malformed parameters with a
logged warning instead of raising ValueError and degrading the whole tool call
to a silent-plain-text response.

This runs before `vllm serve` imports the parser when loaded via PYTHONSTARTUP.
Remove the file and the env var once PR #39772 lands in the pinned nightly.

Validated against vllm/vllm-openai@sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c
on 2026-04-21.
"""

import json
import logging

try:
    from vllm.tool_parsers import qwen3coder_tool_parser as _q
    from vllm.tool_parsers.utils import find_tool_properties
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ToolCall,
        FunctionCall,
    )
except Exception:
    # vLLM not importable yet (e.g. PYTHONSTARTUP fires in a bare interpreter);
    # vLLM's own CLI import chain will surface any real error later.
    raise SystemExit(0)

_logger = logging.getLogger(__name__)


def _patched_parse_xml_function_call(self, function_call_str):
    # Extract function name
    end_index = function_call_str.find(">")
    if end_index == -1:
        return None
    function_name = function_call_str[:end_index]

    param_config = find_tool_properties(self.tools, function_name)
    parameters = function_call_str[end_index + 1 :]

    param_dict = {}
    for match_text in self.tool_call_parameter_regex.findall(parameters):
        # --- The PR #39772 fix: find() instead of index(), skip if missing.
        idx = match_text.find(">")
        if idx == -1:
            _logger.warning(
                "Skipping malformed parameter (no '>' separator) "
                "in tool call for function %r: %r",
                function_name,
                match_text,
            )
            continue
        # ---

        param_name = match_text[:idx]
        param_value = str(match_text[idx + 1 :])

        # Preserve upstream's leading/trailing newline stripping behavior.
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


_q.Qwen3CoderToolParser._parse_xml_function_call = _patched_parse_xml_function_call
