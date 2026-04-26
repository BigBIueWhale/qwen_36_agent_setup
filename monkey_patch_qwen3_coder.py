"""Strict, fail-loud runtime patch for vLLM tool-parser bug #39771.

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce`` (README §3.2).
Mirrors upstream PR #39772 semantically.
Remove this file the moment #39772 is present in the pinned nightly
(README §11 step 3).

What the bug is
---------------

The upstream method ``Qwen3CoderToolParser._parse_xml_function_call``
at ``vllm/tool_parsers/qwen3coder_tool_parser.py:236`` contains::

    idx = match_text.index(">")

``str.index`` raises ``ValueError`` when the parameter name tag was
truncated mid-``<parameter=NAME`` (no closing ``>``) — an everyday
occurrence under ``max_tokens`` cutoff or a client disconnect. The
``ValueError`` propagates to the outer ``try/except Exception`` in
``extract_tool_calls`` (lines 320-324), which catches indiscriminately
and returns ``tools_called=False, content=<raw_xml_markup>``. Result:
every well-formed tool call in the same response is collapsed because
one parameter was incomplete. That is the silent-failure class the
README §6.13 catalog exists to name.

Why "forgive the broken param and fabricate the rest" is wrong
--------------------------------------------------------------

A ToolCall assembled from the well-formed subset of a truncated
parameter list has a correct-looking ``name``, valid JSON
``arguments``, and no indicator that fields were silently dropped.
An agent loop cannot distinguish "the model deliberately omitted this
field" from "the model got cut off mid-emission", and routes partial
arguments to destructive side effects (file_write, shell, api_post).
A warning line on stderr is not a recovery mechanism. It is the
absence of one.

What this patch does
--------------------

Replaces ``_parse_xml_function_call`` with a version that:

* Uses ``str.find(">")`` instead of ``str.index(">")``.
* On a malformed parameter tag, returns ``None`` for the **whole**
  tool call — propagating the "one bad param poisons this call"
  decision explicitly rather than via a smuggled exception.
* Leaves every sibling tool call in the same response untouched —
  this is the real substance of PR #39772. ``extract_tool_calls``
  line 313 filters ``None``s out and forwards the valid siblings.
* Client patch 3 (``client/validate_response.py``) then observes the
  leaked ``<parameter=`` markup in ``content`` and raises
  :class:`~client.validate_response.MarkupLeakError`; the agent
  loop's recovery table (README §7) retries with a bumped
  ``max_tokens``. That is the architecturally correct split.

Patch-discipline contract
-------------------------

This file is a patch, not a library. At import it:

1. Imports vLLM. Failure is a hard ImportError; we do not hide it.
2. Looks up ``Qwen3CoderToolParser`` by name in the expected module.
3. Verifies the class is a subclass of ``ToolParser``.
4. Verifies the method to replace exists, is callable, and has
   exactly the signature ``(self, function_call_str)``.
5. Reads the method's source and verifies the buggy landmark
   ``match_text.index(">")`` is present. If it is NOT — upstream
   has already fixed this — the patch refuses to apply rather
   than silently overwriting a function whose semantics it no
   longer understands.
6. Verifies the compiled regex attribute name
   (``tool_call_parameter_regex``) the replacement body references
   matches upstream's source landmark. Without this check, an
   upstream rename would cause the patched method to
   ``AttributeError`` on the first real request.
7. Verifies the sibling method ``_convert_param_value`` still
   takes ``(self, param_value, param_name, param_config,
   func_name)``, because the replacement body calls it with that
   signature.
8. Installs the replacement and verifies it carries the patch
   tag after assignment. A concurrent monkey-patch somewhere else
   that clobbered us between assignment and verification raises.
9. Logs a single INFO line via ``vllm.logger.init_logger`` naming
   the class, the method, and the pinned commit.

Any of 1-8 failing raises :class:`MonkeyPatchRefusedError` and the
interpreter does not continue. There is no ``SystemExit(0)`` or
``try/except Exception: pass`` on any path. A half-applied patch
returns the server to the silent-failure mode this file exists to
eliminate; that outcome is strictly worse than a loud crash at boot.

Loading
-------

README §8.2 sets ``PYTHONSTARTUP=/opt/patches/monkey_patch.py``.
CPython honours ``PYTHONSTARTUP`` only in *interactive* mode, so a
plain ``vllm serve`` entrypoint may not execute this file. A safer
loader is to wrap the entry as::

    python -c "import monkey_patch_qwen3_coder; \\
               from runpy import run_module; \\
               run_module('vllm.entrypoints.cli.main', \\
                          run_name='__main__')" serve …

after mounting this file on ``PYTHONPATH``. The load mechanism
fix lives in the deployment command, not in this file; flagged
here so the issue is not lost across re-reads.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable, TypeAlias


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_BUGGY_SENTINEL: str = 'match_text.index(">")'
_EXPECTED_REGEX_ATTR: str = "tool_call_parameter_regex"
_UPSTREAM_REGEX_LANDMARK: str = "<parameter=(.*?)(?:</parameter>"
_PATCH_TAG: str = "qwen36-agent-setup-pr39772-backport"


# Type alias for the method we replace. Kept at module level so
# callers inspecting the installed attribute can tell by signature
# that it still conforms.
ParseXmlFunctionCall: TypeAlias = Callable[[Any, str], Any]


class MonkeyPatchRefusedError(RuntimeError):
    """A precondition for the qwen3_coder tool-parser patch was violated.

    Raised at import time only. The patch either applies cleanly and
    correctly, or the process does not come up. A half-installed or
    silently-skipped patch returns the server to the silent-failure
    mode this file was written to eliminate.
    """


def _require(condition: object, failure_message: str) -> None:
    if not condition:
        raise MonkeyPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {failure_message}"
        )


# --------------------------------------------------------------------
# Phase 1: Import vLLM and locate the target surface.
# --------------------------------------------------------------------

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.tool_parsers import qwen3coder_tool_parser as _qwen3coder_mod
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import find_tool_properties
from vllm.entrypoints.openai.engine.protocol import (
    FunctionCall,
    ToolCall,
)

_logger = init_logger(__name__)


_ParserCls = getattr(_qwen3coder_mod, "Qwen3CoderToolParser", None)
_require(
    _ParserCls is not None,
    "Qwen3CoderToolParser is no longer exported from "
    "vllm.tool_parsers.qwen3coder_tool_parser. Upstream has moved or "
    "renamed the class; re-audit before bumping the pinned commit.",
)
_require(
    inspect.isclass(_ParserCls),
    "vllm.tool_parsers.qwen3coder_tool_parser.Qwen3CoderToolParser "
    "is no longer a class.",
)
_require(
    issubclass(_ParserCls, ToolParser),
    "Qwen3CoderToolParser is no longer a subclass of ToolParser. "
    "Upstream has restructured the tool-parser hierarchy and the "
    "contract this patch relies on may no longer hold.",
)


# --------------------------------------------------------------------
# Phase 2: Landmark the method we intend to replace.
# --------------------------------------------------------------------

_original: ParseXmlFunctionCall | None = getattr(
    _ParserCls, "_parse_xml_function_call", None
)
_require(
    _original is not None and callable(_original),
    "Qwen3CoderToolParser._parse_xml_function_call is missing or "
    "not callable.",
)

try:
    _sig = inspect.signature(_original)  # type: ignore[arg-type]
except (TypeError, ValueError) as _exc:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect signature of "
        f"_parse_xml_function_call: {_exc!r}"
    ) from _exc

_param_names = list(_sig.parameters)
_require(
    _param_names == ["self", "function_call_str"],
    f"_parse_xml_function_call signature changed; expected "
    f"(self, function_call_str), got {_param_names!r}.",
)

try:
    _original_src = inspect.getsource(_original)  # type: ignore[arg-type]
except (OSError, TypeError) as _exc:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] cannot read source of "
        f"_parse_xml_function_call (is vLLM installed without "
        f"accompanying .py files?): {_exc!r}"
    ) from _exc

_require(
    _BUGGY_SENTINEL in _original_src,
    f"buggy sentinel {_BUGGY_SENTINEL!r} is not present in "
    "_parse_xml_function_call source. Upstream PR #39772 has most "
    "likely already landed — delete this patch file and its mount "
    "per README §11 step 3 rather than re-patching a fixed function.",
)


# --------------------------------------------------------------------
# Phase 3: Landmark the instance attributes the replacement body reads.
# --------------------------------------------------------------------

# The replacement body references `self.tool_call_parameter_regex` and
# `self.tools`. Neither is settable at class-definition time (they are
# created in __init__), so we verify them via __init__ source.
#
# IMPORTANT: walk the MRO. ``self.tool_call_parameter_regex`` is set in
# the *subclass*'s ``__init__``; ``self.tools`` is set in the *base
# class* ``ToolParser.__init__`` and inherited via ``super().__init__()``.
# Looking at only ``_ParserCls.__init__`` source would miss the inherited
# assignment and refuse incorrectly. Combine sources across the MRO so
# either lineage is acceptable, then assert each landmark is present
# *somewhere* in the chain.

# Strict regex match: word-boundary ``self.tools`` followed by either
# ``=`` (assignment) or ``:`` (annotated declaration). Avoids matching
# ``self.tools_something`` or ``self.tools.append(...)`` calls.
_TOOLS_ASSIGNMENT_RE = re.compile(r"\bself\.tools\s*[:=]")
_REGEX_ATTR_ASSIGNMENT_RE = re.compile(
    rf"\bself\.{re.escape(_EXPECTED_REGEX_ATTR)}\s*[:=]"
)


def _collect_init_sources(cls: type) -> list[tuple[str, str]]:
    """Walk the MRO and return ``(qualname, source)`` for every ancestor
    that defines its own ``__init__`` (vs. inheriting one).

    Refuses with :class:`MonkeyPatchRefusedError` if no ``__init__`` is
    inspectable — that is a structural surprise we should not paper
    over.
    """
    collected: list[tuple[str, str]] = []
    for ancestor in cls.__mro__:
        if ancestor is object:
            continue
        if "__init__" not in ancestor.__dict__:
            continue
        try:
            src = inspect.getsource(ancestor.__init__)
        except (OSError, TypeError) as exc:
            raise MonkeyPatchRefusedError(
                f"[{_PATCH_TAG}] cannot read source of "
                f"{ancestor.__qualname__}.__init__ during MRO walk for "
                f"attribute-landmark verification: {exc!r}"
            ) from exc
        collected.append((ancestor.__qualname__, src))
    if not collected:
        raise MonkeyPatchRefusedError(
            f"[{_PATCH_TAG}] no inspectable __init__ found anywhere in "
            f"the MRO of {cls.__qualname__}. Cannot verify the "
            f"replacement body's attribute assumptions."
        )
    return collected


_init_sources = _collect_init_sources(_ParserCls)
_init_qualnames = [qn for qn, _ in _init_sources]
_combined_init_src = "\n".join(src for _, src in _init_sources)


def _require_attribute_in_mro(
    matcher: re.Pattern[str],
    attr_label: str,
    consequence: str,
) -> None:
    """Refuse if ``matcher`` does not match anywhere in the MRO's
    combined ``__init__`` sources. The error message names every
    ancestor we searched so an operator can audit the claim.
    """
    if matcher.search(_combined_init_src):
        return
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] no {attr_label} assignment found anywhere in "
        f"the __init__ chain of Qwen3CoderToolParser. {consequence} "
        f"Searched (in MRO order): {_init_qualnames!r}."
    )


_require_attribute_in_mro(
    _REGEX_ATTR_ASSIGNMENT_RE,
    f"self.{_EXPECTED_REGEX_ATTR}",
    f"The replacement body reads self.{_EXPECTED_REGEX_ATTR}; without "
    f"the attribute the patched call would AttributeError on the "
    f"first request.",
)
# The upstream-regex SHAPE landmark must appear in the SUBCLASS source
# (it is the subclass that compiles and assigns the regex). Don't
# weaken this by allowing matches in unrelated ancestors.
_require(
    _UPSTREAM_REGEX_LANDMARK in _init_sources[0][1],
    f"expected parameter-regex landmark {_UPSTREAM_REGEX_LANDMARK!r} "
    f"not found in {_init_sources[0][0]}.__init__ source. The upstream "
    f"regex's structural shape has changed; the replacement body's "
    f"assumption that each match yields a single 'KEY>VALUE' body may "
    f"no longer hold.",
)
_require_attribute_in_mro(
    _TOOLS_ASSIGNMENT_RE,
    "self.tools",
    "The replacement body passes self.tools to find_tool_properties; "
    "without this attribute the patched call would AttributeError at "
    "request time.",
)


# --------------------------------------------------------------------
# Phase 4: Landmark the sibling method the replacement body calls.
# --------------------------------------------------------------------

_cpv: Callable[..., Any] | None = getattr(
    _ParserCls, "_convert_param_value", None
)
_require(
    _cpv is not None and callable(_cpv),
    "_convert_param_value is missing on Qwen3CoderToolParser; the "
    "replacement body calls it and would AttributeError at request "
    "time.",
)
try:
    _cpv_params = list(inspect.signature(_cpv).parameters)  # type: ignore[arg-type]
except (TypeError, ValueError) as _exc:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect _convert_param_value: {_exc!r}"
    ) from _exc
_require(
    _cpv_params
    == ["self", "param_value", "param_name", "param_config", "func_name"],
    f"_convert_param_value signature changed; expected "
    f"(self, param_value, param_name, param_config, func_name), "
    f"got {_cpv_params!r}.",
)


# --------------------------------------------------------------------
# Phase 5: Verify the helper we import is the one we think it is.
# --------------------------------------------------------------------

try:
    _fp_params = list(inspect.signature(find_tool_properties).parameters)
except (TypeError, ValueError) as _exc:
    raise MonkeyPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect find_tool_properties: {_exc!r}"
    ) from _exc
_require(
    _fp_params == ["tools", "tool_name"],
    f"find_tool_properties signature changed; expected "
    f"(tools, tool_name), got {_fp_params!r}. The replacement body "
    f"calls it positionally with that shape.",
)


# --------------------------------------------------------------------
# Phase 6: The replacement.
# --------------------------------------------------------------------


def _parse_xml_function_call_strict(
    self: Any, function_call_str: str
) -> Any:
    """Strict replacement for Qwen3CoderToolParser._parse_xml_function_call.

    Contract vs. upstream:

    * Identical signature and return type.
    * Identical handling of a missing function-name terminator
      (``'>'``): return None.
    * DIFFERENT handling of a malformed parameter tag (missing
      ``'>'``): return None for the **entire tool call**, instead
      of raising ValueError. This turns a one-param truncation
      from "drop all N tool calls in the response" (upstream bug)
      into "drop this one tool call, keep siblings".
    * DELIBERATELY does NOT keep the other, well-formed params of
      a tool call whose param list contains a malformed entry. A
      ToolCall with silently-omitted fields is a correctness
      hazard in an agentic loop; returning None surfaces the
      failure to the silent-parser-failure detector
      (``client/validate_response.py``) where it is recovered
      deterministically via retry.

    Per-call validation: ``function_call_str`` is expected to be
    a ``str`` per the upstream signature. A non-str would indicate
    the parser's caller (``_get_function_calls`` line 270-272) has
    drifted; we raise TypeError rather than silently degrading,
    because degradation here looks identical to a legitimate
    ``return None`` and would make the upstream regression
    invisible.
    """
    if not isinstance(function_call_str, str):
        raise TypeError(
            f"[{_PATCH_TAG}] _parse_xml_function_call expected str for "
            f"function_call_str, got {type(function_call_str).__name__!r}"
        )

    end_index: int = function_call_str.find(">")
    if end_index == -1:
        return None
    function_name: str = function_call_str[:end_index]
    param_config = find_tool_properties(self.tools, function_name)
    parameters_blob: str = function_call_str[end_index + 1 :]

    param_dict: dict[str, Any] = {}
    for match_text in self.tool_call_parameter_regex.findall(parameters_blob):
        if not isinstance(match_text, str):
            # The regex has exactly one capture group in upstream's
            # compiled form (__init__ line 66-69); .findall returns
            # a list[str] under that shape. A tuple here would mean
            # someone added a second capturing group upstream and
            # the body below would mis-index. Raise rather than
            # guess which element of the tuple to use.
            raise TypeError(
                f"[{_PATCH_TAG}] self.tool_call_parameter_regex.findall "
                f"yielded {type(match_text).__name__!r}, expected str. "
                f"Upstream regex capture-group shape has changed; the "
                f"replacement body's parsing assumption no longer holds."
            )
        idx = match_text.find(">")
        if idx == -1:
            return None
        param_name: str = match_text[:idx]
        param_value: str = match_text[idx + 1 :]
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


# --------------------------------------------------------------------
# Phase 7: Install and verify.
# --------------------------------------------------------------------

_ParserCls._parse_xml_function_call = _parse_xml_function_call_strict

_installed = _ParserCls._parse_xml_function_call
_require(
    getattr(_installed, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install verification failed: "
    "Qwen3CoderToolParser._parse_xml_function_call does not bear the "
    "expected patch tag. A concurrent monkey-patch has clobbered ours.",
)

# Second-order verification: the class itself must resolve the
# replacement via normal attribute lookup (not just the instance
# __dict__), guarding against metaclass-level __getattribute__
# overrides that could otherwise hide our assignment.
_resolved = inspect.getattr_static(_ParserCls, "_parse_xml_function_call")
_require(
    getattr(_resolved, "__qwen36_patch__", None) == _PATCH_TAG,
    "static-lookup verification failed: "
    "inspect.getattr_static sees a different "
    "_parse_xml_function_call than normal attribute access. Something "
    "in the MRO or metaclass is shadowing our assignment; refusing to "
    "proceed.",
)


_logger.info(
    "[%s] applied: replaced %s.%s for vLLM commit %s "
    "(malformed <parameter= drops the single tool call via None "
    "return; siblings preserved; no silent field skipping).",
    _PATCH_TAG,
    _ParserCls.__module__,
    _ParserCls.__qualname__,
    _PINNED_VLLM_COMMIT,
)
