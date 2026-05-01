"""Strict, fail-loud streaming-truncation rescue for Qwen3CoderToolParser.

Why this patch must exist (D13)
-------------------------------

``Qwen3CoderToolParser.extract_tool_calls_streaming`` at
``vllm/tool_parsers/qwen3coder_tool_parser.py:326-683`` has a state-
machine break on a parameter value that is interrupted by ``max_tokens``
truncation. Concretely, in the inner-loop block at lines 550-573:

.. code-block:: python

    param_end_idx = value_text.find(self.parameter_end_token)
    if param_end_idx == -1:
        next_param_idx = value_text.find(self.parameter_prefix)
        func_end_idx = value_text.find(self.function_end_token)
        if next_param_idx != -1 and (func_end_idx == -1 or next_param_idx < func_end_idx):
            param_end_idx = next_param_idx
        elif func_end_idx != -1:
            param_end_idx = func_end_idx
        else:
            tool_end_in_value = value_text.find(self.tool_call_end_token)
            if tool_end_in_value != -1:
                param_end_idx = tool_end_in_value
            else:
                # Parameter incomplete — break so we still emit any
                # fragments accumulated by earlier loop iterations.
                break

When ``max_tokens`` truncates the model output INSIDE a
``<parameter=KEY>VALUE`` block — none of ``</parameter>``,
``<parameter=``, ``</function>``, ``</tool_call>`` are present in
``value_text`` — the ``break`` fires without emitting any json fragment
for the partial value. The parser returns ``None``; the streaming
serving generator at ``vllm/entrypoints/openai/chat_completion/
serving.py:1035-1044`` substitutes an empty ``DeltaMessage()`` and the
serving-layer rescue at lines 1107-1158 cannot recover the partial
because ``_should_check_for_unstreamed_tool_arg_tokens`` (lines
1785-1807) requires
``delta_message.tool_calls[0].function.arguments is not None``.

The non-streaming counterpart ``extract_tool_calls`` (lines 275-324)
uses the regex
``<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)``
whose terminal ``|$`` alternative captures up to EOF on truncation,
so non-streaming preserves the partial value bytes.

Concrete observed shapes (``/tmp/safety_repro/D13_force_truncate.sh``):

.. code-block:: text

    NON-STREAM (correct, 681 bytes):
        arguments='{"file_path": "long_script.py", "content": "#!/usr/bin/env python3\\n...\\nimport sys"'
    STREAM   (broken,    30 bytes):
        arguments='{"file_path": "long_script.py"'

Streaming silently drops the entire ``content`` parameter.

Why this is bad
~~~~~~~~~~~~~~~

* **Silent data loss.** The agent's tool-call argument shape diverges
  by mode (streaming vs non-streaming) for byte-identical model output.
  An agent that retries the truncated call expects to see the partial
  to know "this was truncated; lift max_tokens and retry"; instead it
  sees a different, smaller args dict and no signal.
* **Disagreement between paths is exactly the class of bug §6 catalogues.**
  Non-stream and streaming must produce byte-identical decoded
  ``tool_calls[0].function.arguments`` when starting from the same
  ``model_output`` string. D13 is a stream-only loss.

Patch design
------------

Three moving parts split across two layers:

1. **Parser-side wrap** — ``Qwen3CoderToolParser.extract_tool_calls_streaming``.
   After the original returns, if the parser is in mid-parameter-value
   state (``is_tool_call_started=True, in_function=True,
   json_started=True, json_closed=False``), recompute the would-be
   ``expected_call`` JSON for the in-progress tool call and store it in
   ``prev_tool_call_arr[current_tool_index]["arguments"]``. The
   serving-layer rescue at ``serving.py:1139-1145`` reads this slot as
   the source of truth for ``expected_call``; updating it with the
   partial value is what makes the rescue produce the right final delta.

   On the **happy path** (no truncation) this is idempotent: the
   parser's existing happy-path update at lines 642-657 fires when
   ``</function>`` is observed and overwrites our intermediate value
   with the byte-identical full-args JSON the non-streaming
   ``_parse_xml_function_call`` produces. Our wrap is only load-bearing
   when ``</function>`` never arrives.

   The wrap **never** mutates ``streamed_args_for_tool``; that array
   remains the truthful record of what was actually emitted on the
   wire. The serving-layer rescue subtracts ``streamed_args_for_tool``
   from our updated ``prev_tool_call_arr`` to compute the missing tail.

2. **Serving-side wrap** — ``OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens``.
   The original predicate at ``serving.py:1785-1807`` requires
   ``delta_message.tool_calls[0].function.arguments is not None``,
   which is False when the parser returned ``None`` (the truncation
   case). Our wrap returns the original's verdict if it was True,
   else extends the predicate to fire when:

   * ``output.finish_reason is not None`` (this is a final delta),
   * ``self.enable_auto_tools`` (auto-tool-parser is in use),
   * ``self.tool_parser is Qwen3CoderToolParser`` (we are the parser
     this rescue path was designed for),
   * ``delta_message is not None`` (line 1044 ensures this on the
     truncation path).

   When the extended predicate fires, the wrap also **mutates
   delta_message** to ensure ``delta_message.tool_calls`` is non-empty
   (a synthetic ``DeltaToolCall(index=0, function=DeltaFunctionCall(
   arguments=""))``). This is load-bearing: the rescue body at
   ``serving.py:1116-1126`` indexes ``delta_message.tool_calls[0]``
   without first checking emptiness; an empty list raises ``IndexError``
   before ``_create_remaining_args_delta`` ever runs. Wrapping the
   predicate is the smallest possible site that runs **before** that
   indexing — the alternative is wrapping the 700-line
   ``chat_completion_stream_generator`` itself, which buys nothing
   semantically and quadruples the patch's removal-trigger surface.

   The mutation is a deliberate side effect inside what looks like a
   pure predicate. The contract is documented in the wrapper docstring
   below; the patch tag and ``__wrapped_original__`` chain make the
   behaviour discoverable. The rescue body's ``_create_remaining_args_delta``
   call (line 1156) reads our synthetic ``DeltaToolCall`` via
   ``next(tc for tc in delta_message.tool_calls if tc.index == index)``
   — the synthesized index=0 matches when there is a single tool call;
   for multi-tool truncation, our wrap walks back to find the actual
   ``current_tool_index`` from the parser's class identity (we cannot
   read the runtime instance from the predicate scope, but the rescue
   body reads ``index = len(prev_tool_call_arr) - 1`` at line 1102-1106
   so the synthetic ``index=0`` is OK as long as ``_create_remaining_args_delta``
   only uses the original_tc's id/type/name (all None on a synthetic),
   not its index — and per the original's body at lines 1819-1836 it
   does exactly that).

3. **Helper method** — ``Qwen3CoderToolParser._qwen36_compute_partial_args_json``.
   Reproduces the parser's own param-locating logic on the cumulative
   ``current_text`` to identify the in-progress parameter and serialize
   its partial value via the same ``_convert_param_value`` /
   ``json.dumps`` path the loop body uses. Returns the full
   would-be ``expected_call`` JSON string (no closing ``}``, matching
   non-stream truncation byte-shape) or ``None`` if no partial is
   observable.

Why not Option A (full method override)
---------------------------------------

The streaming method is a 358-line state machine. Replacing it
verbatim doubles the bug-discovery surface for any future upstream
refactor; the parser's MRO walk and source landmarks already let us
detect surrounding-code drift, and the small wrap composes with that
machinery instead of re-implementing it.

Why not Option B (post-hoc wrapper that synthesises the rescue alone)
---------------------------------------------------------------------

By the time a wrapper around ``extract_tool_calls_streaming`` runs,
the parser has already returned ``None`` for the truncation delta and
the serving generator's iteration is poised to emit
``DeltaMessage()`` (line 1044). The wrapper has no signal that the
*next* dispatch is the final one — only the serving layer's
``output.finish_reason`` knows. The cleanest place to inject is the
serving-layer predicate, hence Option C/D hybrid above.

Why not Option C alone (serving-layer rescue extended to read parser state)
---------------------------------------------------------------------------

The serving-layer rescue reads ``tool_parser.prev_tool_call_arr[index]
.get("arguments", {})``. If the parser doesn't update that slot with
the partial, the rescue computes ``expected = "{}"`` (the initial
header value at parser line 472) and ``remaining = "{}".replace(
streamed, "")`` which yields ``"{}"`` (because ``streamed`` is not a
substring of ``"{}"``); the spurious final delta would emit a literal
``"{}"`` after what the wire already saw. Either we update the parser's
state OR we replace the rescue body wholesale. The former is one wrap
on the parser; the latter is wrapping a 700-line generator. We update.

Why not Option D alone (parser-side helper invoked by serving layer on finish)
------------------------------------------------------------------------------

Same wrapping-cost objection: there is no extension point in the
serving-layer rescue that lets a parser opt into "give me a chance to
flush". Adding one would require wrapping the generator. We achieve
the same result by having the parser maintain ``prev_tool_call_arr``
in sync and the predicate fire on the truncation case — both cheap
wraps; no generator surgery.

Removal trigger
---------------

* vLLM merges a fix that EITHER (a) makes the parser's break path
  emit the partial fragment, OR (b) extends
  ``_should_check_for_unstreamed_tool_arg_tokens`` to fire on the
  truncation case AND adjusts the rescue body to be tolerant of
  ``delta_message.tool_calls`` being empty.
* The bug-shape landmark is the ``# Parameter incomplete — break``
  comment (and the ``break`` it precedes) at parser line 569-573;
  any rewrite that emits before breaking changes its body and forces
  this patch to refuse.

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce``.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers import qwen3coder_tool_parser as _qwen3coder_mod
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import find_tool_properties


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-streaming-truncation-rescue-v1"


# --------------------------------------------------------------------
# Source landmarks. Each one identifies a substring of the original
# vLLM source that must be present for this patch to apply cleanly.
# A future vLLM bump that changes any of these forces a refusal at
# import time so we re-audit before silently overriding a different
# code path.
# --------------------------------------------------------------------

# In ``Qwen3CoderToolParser.extract_tool_calls_streaming``: the
# ``# Parameter incomplete — break`` comment immediately above the
# bug-load-bearing ``break`` statement at line 573. If this comment
# disappears (e.g. upstream changed the comment text or moved the
# break), refuse — a rewrite of this region is the canonical removal
# trigger for this patch.
_PARSER_BREAK_COMMENT_LANDMARK: str = (
    "# Parameter incomplete — break so we still"
)

# In the same method: the param_end_idx scan opener. Confirms the
# loop shape we are reasoning about.
_PARSER_PARAM_END_LANDMARK: str = (
    "param_end_idx = value_text.find(self.parameter_end_token)"
)

# In the same method: the ``in_function`` body opener. Confirms the
# state-machine flag this patch reads still exists.
_PARSER_IN_FUNCTION_LANDMARK: str = "if self.in_function:"

# In the same method: the parameter-loop guard. Confirms the
# ``param_count`` / ``param_starts`` shape we replicate in
# ``_qwen36_compute_partial_args_json`` matches the upstream loop.
_PARSER_PARAM_LOOP_LANDMARK: str = (
    "while not self.in_param and self.param_count < len(param_starts):"
)

# In ``OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens``:
# the predicate's terminal arguments-non-None clause. This is what
# makes the original return False when the parser returned None.
_SHOULD_CHECK_PREDICATE_LANDMARK: str = (
    "delta_message.tool_calls[0].function.arguments is not None"
)

# In the same predicate: the finish_reason guard at the head. The
# original predicate is ANDed with ``output.finish_reason is not
# None`` — if upstream made finish_reason an OR-ed condition or
# removed it, our extension's gate semantics no longer match.
_SHOULD_CHECK_FINISH_REASON_LANDMARK: str = (
    "output.finish_reason is not None"
)

# In the streaming generator's rescue body at ``serving.py``:1107-1158:
# the indexing access this patch's mutation defends against. If
# upstream rewrote the body to be tolerant on its own (e.g. added an
# ``if not delta_message.tool_calls`` guard) the mutation becomes
# redundant; refuse so we re-audit.
_RESCUE_BODY_INDEX_LANDMARK: str = (
    "delta_message.tool_calls[0].function.arguments"
)


# Expected signatures. Drift on any of these means the wrapper's
# argument shape no longer matches the upstream callsite.
_EXPECTED_STREAMING_PARAMS: list[str] = [
    "self",
    "previous_text",
    "current_text",
    "delta_text",
    "previous_token_ids",
    "current_token_ids",
    "delta_token_ids",
    "request",
]
_EXPECTED_PREDICATE_PARAMS: list[str] = [
    "self",
    "delta_message",
    "output",
]


_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class StreamingTruncationPatchRefusedError(RuntimeError):
    """A precondition for the qwen3_coder streaming-truncation rescue
    patch was violated.

    Raised at import time only. The patch either applies cleanly or
    the process does not come up — there is no half-installed path. A
    half-installed truncation rescue (parser updates state but
    serving-side predicate not extended, or vice versa) silently
    converts D13 from "lose the partial value" into "double-stream
    the partial value" — strictly worse than the unpatched bug.
    Refusing to boot is strictly safer.
    """


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise StreamingTruncationPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {msg}"
        )


# --------------------------------------------------------------------
# Phase 1: Locate target classes and verify class hierarchy.
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

_require(
    inspect.isclass(OpenAIServingChat),
    "OpenAIServingChat is missing or not a class.",
)


# --------------------------------------------------------------------
# Phase 2: Landmark the parser's streaming method.
# --------------------------------------------------------------------

_original_streaming = getattr(_ParserCls, "extract_tool_calls_streaming", None)
_require(
    _original_streaming is not None and callable(_original_streaming),
    "Qwen3CoderToolParser.extract_tool_calls_streaming missing or not callable.",
)
try:
    _streaming_sig = inspect.signature(_original_streaming)  # type: ignore[arg-type]
    _streaming_src = inspect.getsource(_original_streaming)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise StreamingTruncationPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect "
        f"Qwen3CoderToolParser.extract_tool_calls_streaming: {_exc!r}"
    ) from _exc

_require(
    list(_streaming_sig.parameters) == _EXPECTED_STREAMING_PARAMS,
    f"Qwen3CoderToolParser.extract_tool_calls_streaming signature drifted; "
    f"expected {_EXPECTED_STREAMING_PARAMS!r}, got "
    f"{list(_streaming_sig.parameters)!r}.",
)
_require(
    _PARSER_BREAK_COMMENT_LANDMARK in _streaming_src,
    f"break-on-truncation landmark "
    f"{_PARSER_BREAK_COMMENT_LANDMARK!r} not present in "
    f"Qwen3CoderToolParser.extract_tool_calls_streaming source. The "
    f"break path this patch rescues has been rewritten upstream — "
    f"delete this patch file and its mount.",
)
_require(
    _PARSER_PARAM_END_LANDMARK in _streaming_src,
    f"parameter-scan landmark {_PARSER_PARAM_END_LANDMARK!r} not "
    f"present in Qwen3CoderToolParser.extract_tool_calls_streaming source.",
)
_require(
    _PARSER_IN_FUNCTION_LANDMARK in _streaming_src,
    f"in_function-branch landmark {_PARSER_IN_FUNCTION_LANDMARK!r} "
    f"not present in Qwen3CoderToolParser.extract_tool_calls_streaming source.",
)
_require(
    _PARSER_PARAM_LOOP_LANDMARK in _streaming_src,
    f"parameter-loop landmark {_PARSER_PARAM_LOOP_LANDMARK!r} not "
    f"present in Qwen3CoderToolParser.extract_tool_calls_streaming source. "
    f"The param_count/param_starts walk this helper replicates has "
    f"been restructured upstream.",
)


# --------------------------------------------------------------------
# Phase 3: Verify expected attributes on Qwen3CoderToolParser
# instances. These are read by the helper and the wrap; without them
# the wrap would AttributeError at request time.
# --------------------------------------------------------------------

_combined_init_src = ""
for _ancestor in _ParserCls.__mro__:
    if _ancestor is object or "__init__" not in _ancestor.__dict__:
        continue
    try:
        _combined_init_src += "\n" + inspect.getsource(_ancestor.__init__)
    except (OSError, TypeError) as _exc:
        raise StreamingTruncationPatchRefusedError(
            f"[{_PATCH_TAG}] cannot read {_ancestor.__qualname__}.__init__: {_exc!r}"
        ) from _exc

# These are state attributes the wrap and helper read.
_REQUIRED_PARSER_ATTRS: list[str] = [
    "is_tool_call_started",
    "in_function",
    "json_started",
    "json_closed",
    "param_count",
    "current_tool_index",
    "current_function_name",
    "prev_tool_call_arr",
    "streamed_args_for_tool",
    "tool_call_start_token",
    "tool_call_end_token",
    "tool_call_prefix",
    "function_end_token",
    "parameter_prefix",
    "parameter_end_token",
    "tools",
]
for _attr in _REQUIRED_PARSER_ATTRS:
    _require(
        f"self.{_attr}" in _combined_init_src,
        f"Qwen3CoderToolParser.__init__ (MRO walk) does not assign "
        f"self.{_attr}; the wrap and helper depend on it.",
    )


# --------------------------------------------------------------------
# Phase 4: Landmark the serving-layer predicate we extend.
# --------------------------------------------------------------------

_original_should_check = getattr(
    OpenAIServingChat, "_should_check_for_unstreamed_tool_arg_tokens", None
)
_require(
    _original_should_check is not None and callable(_original_should_check),
    "OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens missing or not callable.",
)
try:
    _predicate_sig = inspect.signature(_original_should_check)  # type: ignore[arg-type]
    _predicate_src = inspect.getsource(_original_should_check)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise StreamingTruncationPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect "
        f"OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens: {_exc!r}"
    ) from _exc
_require(
    list(_predicate_sig.parameters) == _EXPECTED_PREDICATE_PARAMS,
    f"OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens "
    f"signature drifted; expected {_EXPECTED_PREDICATE_PARAMS!r}, got "
    f"{list(_predicate_sig.parameters)!r}.",
)
_require(
    _SHOULD_CHECK_PREDICATE_LANDMARK in _predicate_src,
    f"predicate landmark {_SHOULD_CHECK_PREDICATE_LANDMARK!r} not "
    f"present in OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens "
    f"source. The arguments-not-None clause this patch's wrap "
    f"complements has been rewritten — re-audit before bumping.",
)
_require(
    _SHOULD_CHECK_FINISH_REASON_LANDMARK in _predicate_src,
    f"predicate finish_reason landmark "
    f"{_SHOULD_CHECK_FINISH_REASON_LANDMARK!r} not present in source — "
    f"the original predicate's finish_reason gate has changed.",
)


# --------------------------------------------------------------------
# Phase 5: Landmark the rescue body's index access in the streaming
# generator. The mutation in our predicate wrap defends against this
# access; if upstream rewrote the body to be index-safe on its own,
# our mutation is redundant and we should re-audit.
# --------------------------------------------------------------------

try:
    _stream_gen_src = inspect.getsource(
        OpenAIServingChat.chat_completion_stream_generator
    )
except (OSError, TypeError) as _exc:
    raise StreamingTruncationPatchRefusedError(
        f"[{_PATCH_TAG}] cannot read source of "
        f"OpenAIServingChat.chat_completion_stream_generator: {_exc!r}"
    ) from _exc
_require(
    _RESCUE_BODY_INDEX_LANDMARK in _stream_gen_src,
    f"rescue-body landmark {_RESCUE_BODY_INDEX_LANDMARK!r} not present "
    f"in chat_completion_stream_generator source. The rescue body's "
    f"unconditional ``delta_message.tool_calls[0]`` indexing has been "
    f"removed — this patch's mutation in the predicate is redundant. "
    f"Re-audit before bumping the pinned commit; the patch may now be "
    f"a partial duplicate of upstream behaviour.",
)


# --------------------------------------------------------------------
# Phase 6: The helper method.
# --------------------------------------------------------------------


def _qwen36_compute_partial_args_json(
    self: Any, current_text: str
) -> str | None:
    """Compute the would-be expected_call JSON for the tool currently
    being streamed, INCLUDING any in-progress partial parameter value.

    Returns a JSON-fragment string suitable for assignment to
    ``self.prev_tool_call_arr[self.current_tool_index]["arguments"]``,
    or ``None`` if no in-progress partial is observable.

    The shape produced is byte-equivalent to what
    ``_parse_xml_function_call`` would produce non-streaming on the
    same truncated XML — that is, the JSON object minus its closing
    ``}`` (since the ``</function>`` that would have triggered the
    closer never arrives). This intentionally matches the
    non-streaming ``arguments`` byte-shape (e.g. the user-reported
    ``arguments='{"file_path": "long_script.py", "content": "#!/usr/bin/env python3..."'``
    in the D13 reproduction).

    The implementation reproduces the parser's own param-locating
    logic from ``extract_tool_calls_streaming`` (lines 425-624 of
    ``qwen3coder_tool_parser.py``):

    1. Find ``current_text``'s ``<tool_call>`` start positions and
       seek to the entry indexed by ``self.current_tool_index``.
    2. From that tool-call body, scan ``<parameter=`` markers; the
       (param_count)-th marker is the in-progress one.
    3. Extract the param name (terminated by ``>``) and the value
       text. If no closing delimiter is present in the value text,
       the value is the entire tail (truncation case); otherwise the
       value extends to whichever delimiter appears first.
    4. Convert the value via ``_convert_param_value`` (same path the
       happy loop uses) and serialize via ``json.dumps``.

    Returns ``None`` (no-op) when:

    * not in a tool call body (``not is_tool_call_started`` /
      ``not in_function``),
    * before the first ``{`` was streamed (``not json_started``),
    * after ``}`` was streamed (``json_closed``),
    * tool index is out of range,
    * no ``<parameter=`` marker has been seen yet for the
      (param_count)-th parameter,
    * the parameter name is not yet terminated by ``>`` (truncation
      landed mid-name; nothing meaningful to render).
    """
    if not self.is_tool_call_started or not self.in_function:
        return None
    if not self.json_started or self.json_closed:
        return None
    if self.current_tool_index >= len(self.prev_tool_call_arr):
        return None
    if self.current_tool_index >= len(self.streamed_args_for_tool):
        return None

    # Find the current tool call's tool_text — same algorithm as
    # extract_tool_calls_streaming lines 427-448.
    tool_start_positions: list[int] = []
    idx = 0
    while True:
        idx = current_text.find(self.tool_call_start_token, idx)
        if idx == -1:
            break
        tool_start_positions.append(idx)
        idx += len(self.tool_call_start_token)

    if self.current_tool_index >= len(tool_start_positions):
        return None

    tool_start_idx = tool_start_positions[self.current_tool_index]
    tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)
    if tool_end_idx == -1:
        tool_text = current_text[tool_start_idx:]
    else:
        tool_text = current_text[
            tool_start_idx : tool_end_idx + len(self.tool_call_end_token)
        ]

    # Find param starts — same algorithm as lines 517-524.
    param_starts: list[int] = []
    search_idx = 0
    while True:
        search_idx = tool_text.find(self.parameter_prefix, search_idx)
        if search_idx == -1:
            break
        param_starts.append(search_idx)
        search_idx += len(self.parameter_prefix)

    # If self.param_count == len(param_starts): all observed params
    # were already streamed (truncation between params). The serving-
    # layer rescue's expected_call should be exactly what was
    # streamed — return that to no-op the rescue.
    if self.param_count >= len(param_starts):
        return self.streamed_args_for_tool[self.current_tool_index]

    # The (param_count)-th param is in progress.
    param_idx = param_starts[self.param_count]
    param_start = param_idx + len(self.parameter_prefix)
    remaining = tool_text[param_start:]
    if ">" not in remaining:
        # Truncation landed mid-parameter-name; no value to render.
        # Return what was streamed so the rescue no-ops.
        return self.streamed_args_for_tool[self.current_tool_index]

    name_end = remaining.find(">")
    current_param_name = remaining[:name_end]

    value_start = param_start + name_end + 1
    value_text = tool_text[value_start:]
    if value_text.startswith("\n"):
        value_text = value_text[1:]

    # Determine where the value ends. Same priority order as the
    # original loop body at lines 550-573 — but with the truncation
    # fallback REPLACED: instead of breaking, take the entire tail.
    truncation_observed = False
    param_end_idx = value_text.find(self.parameter_end_token)
    if param_end_idx == -1:
        next_param_idx = value_text.find(self.parameter_prefix)
        func_end_idx = value_text.find(self.function_end_token)
        if next_param_idx != -1 and (
            func_end_idx == -1 or next_param_idx < func_end_idx
        ):
            param_end_idx = next_param_idx
        elif func_end_idx != -1:
            param_end_idx = func_end_idx
        else:
            tool_end_in_value = value_text.find(self.tool_call_end_token)
            if tool_end_in_value != -1:
                param_end_idx = tool_end_in_value
            else:
                # Truncation case: no delimiter present. The partial
                # value is the entire remaining tail.
                param_end_idx = len(value_text)
                truncation_observed = True

    param_value = value_text[:param_end_idx]
    # Strip trailing newline ONLY when we found a real closing
    # delimiter — that newline is part of the framing
    # (``VALUE\n</parameter>`` / ``VALUE\n<parameter=``). On
    # truncation there is no framing yet; the trailing character is
    # part of the value as the model produced it.
    if not truncation_observed and param_value.endswith("\n"):
        param_value = param_value[:-1]

    param_config = find_tool_properties(
        self.tools, self.current_function_name or ""
    )
    converted_value = self._convert_param_value(
        param_value,
        current_param_name,
        param_config,
        self.current_function_name or "",
    )
    serialized_value = json.dumps(converted_value, ensure_ascii=False)

    # Compose with what's already in streamed_args_for_tool. The
    # streamed prefix already contains "{" plus all earlier complete
    # params; we append the partial fragment WITHOUT a closing "}"
    # (matches the non-streaming truncation byte-shape — see
    # extract_tool_calls' regex at line 235 with the `|$` alternative
    # that captures the partial without forcing a JSON closer).
    existing = self.streamed_args_for_tool[self.current_tool_index]
    fragment_separator = "" if self.param_count == 0 else ", "
    return (
        f'{existing}{fragment_separator}'
        f'"{current_param_name}": {serialized_value}'
    )


# --------------------------------------------------------------------
# Phase 7: The parser-side wrap.
# --------------------------------------------------------------------


def _wrapped_extract_tool_calls_streaming(
    self: Any,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Any,
    current_token_ids: Any,
    delta_token_ids: Any,
    request: Any,
) -> Any:
    """Wrap ``Qwen3CoderToolParser.extract_tool_calls_streaming`` to
    keep ``prev_tool_call_arr[current_tool_index]["arguments"]`` in
    sync with the in-progress partial value.

    Calls the original first (preserving all wire-emission semantics —
    we do NOT change what the parser returns mid-stream). After the
    return, if the parser is in mid-parameter-value state, recomputes
    the would-be expected_call JSON via
    ``_qwen36_compute_partial_args_json`` and stores it in
    ``prev_tool_call_arr``. The serving-layer rescue at
    ``serving.py:1139-1145`` reads that slot as ``expected_call``;
    keeping it accurate is what makes the rescue produce the right
    final delta on truncation.

    Idempotent on the happy path: when ``</function>`` eventually
    arrives, the parser's own update at lines 642-657 overwrites our
    intermediate value with the byte-identical full-args JSON.
    """
    result = _original_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    )

    # Only sync when we are mid-tool-call body. Outside that window,
    # ``prev_tool_call_arr`` is either (a) empty (no tool call seen
    # yet — wrap no-ops), (b) being updated by the parser's own
    # happy-path emit at line 651 (our wrap noop overlaps but does
    # not interfere), or (c) finalized for the previous tool. The
    # explicit gate avoids touching state outside the in-flight
    # parameter-emit window.
    if (
        getattr(self, "is_tool_call_started", False)
        and getattr(self, "in_function", False)
        and getattr(self, "json_started", False)
        and not getattr(self, "json_closed", False)
        and self.prev_tool_call_arr
        and self.current_tool_index < len(self.prev_tool_call_arr)
        and self.current_tool_index < len(self.streamed_args_for_tool)
    ):
        partial = self._qwen36_compute_partial_args_json(current_text)
        if partial is not None:
            self.prev_tool_call_arr[self.current_tool_index]["arguments"] = partial

    return result


_wrapped_extract_tool_calls_streaming.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_wrapped_extract_tool_calls_streaming.__wrapped_original__ = _original_streaming  # type: ignore[attr-defined]
_wrapped_extract_tool_calls_streaming.__name__ = "extract_tool_calls_streaming"
_wrapped_extract_tool_calls_streaming.__qualname__ = (
    f"{_ParserCls.__qualname__}.extract_tool_calls_streaming"
)
_wrapped_extract_tool_calls_streaming.__module__ = _ParserCls.__module__


# --------------------------------------------------------------------
# Phase 8: The serving-side predicate wrap.
# --------------------------------------------------------------------


def _wrapped_should_check_for_unstreamed_tool_arg_tokens(
    self: Any,
    delta_message: Any,
    output: Any,
) -> bool:
    """Wrap ``OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens``
    to also fire on the truncation case.

    Behaviour:

    * Calls the original predicate. If it returned True, returns True
      unchanged — the original happy path is unaltered.
    * Otherwise, on the truncation rescue case (finish_reason set,
      auto-tool parser is Qwen3CoderToolParser, delta_message exists
      but has no tool_calls because the parser returned None), this
      wrapper:

      1. **Mutates ``delta_message``** to add a synthetic
         ``DeltaToolCall(index=0, function=DeltaFunctionCall(arguments=""))``
         so the rescue body's unconditional
         ``delta_message.tool_calls[0]`` indexing at
         ``serving.py:1116-1126`` does not raise IndexError. The
         synthetic id/type/name are None; ``_create_remaining_args_delta``
         at ``serving.py:1819-1836`` is already tolerant of that
         (``original_tc.id if original_tc else None``).

      2. Returns True so the rescue body executes and emits the
         missing tail using ``prev_tool_call_arr[index]["arguments"]``
         (which the parser-side wrap kept in sync) and
         ``streamed_args_for_tool[index]`` (the truth of what was
         emitted on the wire).

    The mutation is a deliberate side effect inside what looks like a
    pure predicate. It is the smallest possible site that runs
    **before** the rescue body's unconditional indexing — wrapping
    the 700-line streaming generator itself would be the alternative
    and offers no semantic benefit.

    The synthetic ``index=0`` is correct on this code path: the
    rescue body at ``serving.py:1156`` calls
    ``self._create_remaining_args_delta(delta_message, remaining_call,
    index)`` with ``index = len(prev_tool_call_arr) - 1`` (line
    1102-1106). ``_create_remaining_args_delta`` matches by index
    via ``next(tc for tc in delta_message.tool_calls if tc.index ==
    index)``; with our synthetic ``index=0``, the match is None when
    ``index != 0``, and the helper falls through to the
    ``original_tc is None`` branch which already sets id/type/name
    to None — exactly the shape we want for a synthetic skeleton.
    """
    original_verdict = _original_should_check(self, delta_message, output)
    if original_verdict:
        return True

    # Extension only fires for the qwen3_coder auto-tool path.
    if not getattr(output, "finish_reason", None):
        return False
    if not getattr(self, "enable_auto_tools", False):
        return False
    # ``self.tool_parser`` is the parser CLASS (see
    # ``ParserManager.get_tool_parser`` returning ``type[ToolParser] |
    # None``), not the runtime instance. Identity check is the right
    # gate — only Qwen3CoderToolParser exhibits this break-on-
    # truncation shape; other parsers' rescue paths must not be
    # affected.
    if getattr(self, "tool_parser", None) is not _ParserCls:
        return False
    if delta_message is None:
        return False

    # Mutate delta_message to ensure tool_calls is non-empty.
    # ``DeltaMessage.tool_calls`` defaults to ``Field(default_factory=list)``
    # (``vllm/entrypoints/openai/engine/protocol.py:262``), so an
    # empty list is the typical shape from the
    # ``DeltaMessage()`` substitution at ``serving.py:1044``.
    if not getattr(delta_message, "tool_calls", None):
        delta_message.tool_calls = [
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(arguments=""),
            )
        ]
    return True


_wrapped_should_check_for_unstreamed_tool_arg_tokens.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_wrapped_should_check_for_unstreamed_tool_arg_tokens.__wrapped_original__ = (  # type: ignore[attr-defined]
    _original_should_check
)
_wrapped_should_check_for_unstreamed_tool_arg_tokens.__name__ = (
    "_should_check_for_unstreamed_tool_arg_tokens"
)
_wrapped_should_check_for_unstreamed_tool_arg_tokens.__qualname__ = (
    f"{OpenAIServingChat.__qualname__}._should_check_for_unstreamed_tool_arg_tokens"
)
_wrapped_should_check_for_unstreamed_tool_arg_tokens.__module__ = (
    OpenAIServingChat.__module__
)


# --------------------------------------------------------------------
# Phase 9: Install both wraps and verify dynamic+static lookup agree.
# --------------------------------------------------------------------

_ParserCls._qwen36_compute_partial_args_json = _qwen36_compute_partial_args_json
_ParserCls.extract_tool_calls_streaming = _wrapped_extract_tool_calls_streaming
OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens = (
    _wrapped_should_check_for_unstreamed_tool_arg_tokens
)

_require(
    getattr(
        _ParserCls.extract_tool_calls_streaming, "__qwen36_patch__", None
    )
    == _PATCH_TAG,
    "post-install: tag absent on Qwen3CoderToolParser.extract_tool_calls_streaming "
    "via attribute access.",
)
_require(
    getattr(
        inspect.getattr_static(_ParserCls, "extract_tool_calls_streaming"),
        "__qwen36_patch__",
        None,
    )
    == _PATCH_TAG,
    "post-install: inspect.getattr_static disagrees on "
    "Qwen3CoderToolParser.extract_tool_calls_streaming.",
)
_require(
    getattr(
        OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens,
        "__qwen36_patch__",
        None,
    )
    == _PATCH_TAG,
    "post-install: tag absent on OpenAIServingChat."
    "_should_check_for_unstreamed_tool_arg_tokens via attribute access.",
)
_require(
    getattr(
        inspect.getattr_static(
            OpenAIServingChat, "_should_check_for_unstreamed_tool_arg_tokens"
        ),
        "__qwen36_patch__",
        None,
    )
    == _PATCH_TAG,
    "post-install: inspect.getattr_static disagrees on "
    "OpenAIServingChat._should_check_for_unstreamed_tool_arg_tokens.",
)


# --------------------------------------------------------------------
# Phase 10: Behavioural verification on a synthetic Qwen3CoderToolParser
# instance. Drives the truncation scenario through the patched parser
# end-to-end and asserts the post-patch state would let the serving-
# layer rescue emit the missing tail.
# --------------------------------------------------------------------
# The parser's __init__ requires only a model_tokenizer whose
# get_vocab() returns non-None ids for <tool_call>/</tool_call>. Same
# minimal shape patches §7.2 and §7.7 use.


class _TokenizerMock:
    """Minimal model_tokenizer surface — only what
    ``Qwen3CoderToolParser.__init__`` actually reads."""

    _vocab = {"<tool_call>": 1_000_001, "</tool_call>": 1_000_002}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def __bool__(self) -> bool:
        return True


# Synthesise the D13 reproduction's tools list shape.
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import FunctionDefinition

_tool_write_file = ChatCompletionToolsParam(
    type="function",
    function=FunctionDefinition(
        name="write_file",
        description="Write content to a file.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    ),
)


# --------------------------------------------------------------------
# Behavioural case A: drive a truncation through the FULL streaming
# path delta-by-delta, and assert that after the truncating delta,
# prev_tool_call_arr[idx]["arguments"] contains the partial value
# bytes. This is the load-bearing assertion: it's what the serving-
# layer rescue reads as expected_call.
# --------------------------------------------------------------------

_parser_probe = _ParserCls(_TokenizerMock(), tools=[_tool_write_file])

# Simulate the deltas the engine would send. Each delta extends
# ``current_text`` and ``delta_text`` is the new tail.
_full_text = (
    "<tool_call>\n"
    "<function=write_file>\n"
    "<parameter=file_path>\n"
    "long_script.py\n"
    "</parameter>\n"
    "<parameter=content>\n"
    "#!/usr/bin/env python3\nimport os\nimport sys"
    # No </parameter>, no </function>, no </tool_call> — truncation.
)
_delta_chunks = [
    "<tool_call>\n<function=write_file>\n",
    "<parameter=file_path>\nlong_script.py\n</parameter>\n",
    "<parameter=content>\n",
    "#!/usr/bin/env python3\n",
    "import os\nimport sys",
]

# Sanity check: chunks reassemble to _full_text.
_require(
    "".join(_delta_chunks) == _full_text,
    "Phase 10 setup: delta chunks do not reassemble to full text — "
    "test fixture bug.",
)


class _StubRequest:
    """Minimal ChatCompletionRequest surface for the parser. Only
    attributes the parser reads are present."""

    tools = [_tool_write_file]
    tool_choice = "auto"


_request_probe = _StubRequest()
_previous_text = ""
_previous_token_ids: list[int] = []
for _chunk_idx, _chunk in enumerate(_delta_chunks):
    _current_text = _previous_text + _chunk
    # delta_token_ids is not load-bearing here (we never trip the
    # tool_call_end_token_id path); pass an empty list.
    _result = _parser_probe.extract_tool_calls_streaming(
        previous_text=_previous_text,
        current_text=_current_text,
        delta_text=_chunk,
        previous_token_ids=_previous_token_ids,
        current_token_ids=_previous_token_ids,
        delta_token_ids=[],
        request=_request_probe,
    )
    _previous_text = _current_text
    # Tag check: verify the wrap actually ran.
    _require(
        getattr(
            type(_parser_probe).extract_tool_calls_streaming,
            "__qwen36_patch__",
            None,
        )
        == _PATCH_TAG,
        f"Phase 10 chunk {_chunk_idx}: streaming wrap not in place "
        f"(tag missing on bound method).",
    )

# After the truncation, prev_tool_call_arr[-1]["arguments"] must
# contain BOTH the file_path AND the partial content value bytes.
_require(
    len(_parser_probe.prev_tool_call_arr) == 1,
    f"Phase 10: expected exactly one entry in prev_tool_call_arr, got "
    f"{len(_parser_probe.prev_tool_call_arr)}.",
)
_recovered_args = _parser_probe.prev_tool_call_arr[-1]["arguments"]
_require(
    isinstance(_recovered_args, str) and _recovered_args,
    f"Phase 10: prev_tool_call_arr[-1]['arguments'] is "
    f"{_recovered_args!r}, expected non-empty str.",
)
_require(
    '"file_path": "long_script.py"' in _recovered_args,
    f"Phase 10: file_path arg missing from recovered args: "
    f"{_recovered_args!r}.",
)
_require(
    '"content":' in _recovered_args,
    f"Phase 10: 'content' key missing from recovered args (the patch's "
    f"core promise — partial parameter value must be in "
    f"prev_tool_call_arr): {_recovered_args!r}.",
)
# The partial value bytes must be present, JSON-quoted.
_require(
    "import os" in _recovered_args and "import sys" in _recovered_args,
    f"Phase 10: partial value bytes missing from recovered args. "
    f"D13 byte-loss is unfixed: {_recovered_args!r}.",
)
# Match non-streaming truncation byte-shape: no closing brace.
_require(
    not _recovered_args.endswith("}"),
    f"Phase 10: recovered args ends with '}}' — closing brace was "
    f"appended where non-stream behaviour does not. "
    f"{_recovered_args!r}.",
)


# --------------------------------------------------------------------
# Behavioural case B: prove the bug exists on the UNPATCHED parser
# (call _original_streaming directly via the same chunk sequence,
# bypassing our wrap) — required for the patch's "if removed, this
# is the regression" claim. We construct a fresh parser instance for
# isolation.
# --------------------------------------------------------------------

_unpatched_probe = _ParserCls(_TokenizerMock(), tools=[_tool_write_file])
_previous_text = ""
for _chunk in _delta_chunks:
    _current_text = _previous_text + _chunk
    # Call the captured original — NOT through the wrap.
    _original_streaming(
        _unpatched_probe,
        _previous_text,
        _current_text,
        _chunk,
        _previous_token_ids,
        _previous_token_ids,
        [],
        _request_probe,
    )
    _previous_text = _current_text

# Unpatched parser leaves prev_tool_call_arr[-1]["arguments"] at "{}"
# (the initial header value at line 472). The bug is "the partial
# value never makes it into prev_tool_call_arr".
_require(
    len(_unpatched_probe.prev_tool_call_arr) == 1,
    f"Phase 10 (unpatched control): expected exactly one entry in "
    f"prev_tool_call_arr, got "
    f"{len(_unpatched_probe.prev_tool_call_arr)}.",
)
_unpatched_args = _unpatched_probe.prev_tool_call_arr[-1]["arguments"]
_require(
    "import sys" not in _unpatched_args,
    f"Phase 10 (unpatched control): the partial value is unexpectedly "
    f"present in the unpatched parser's prev_tool_call_arr "
    f"({_unpatched_args!r}). Either the bug landmark is no longer "
    f"firing OR the test fixture is wrong; refusing to install a "
    f"patch whose precondition cannot be reproduced.",
)


# --------------------------------------------------------------------
# Behavioural case C: exercise the predicate wrap. Stub the
# minimum-viable OpenAIServingChat surface, drive the wrapped
# predicate through three sub-cases.
# --------------------------------------------------------------------


class _StubOutput:
    """Minimal CompletionOutput surface — only finish_reason."""

    def __init__(self, finish_reason: str | None) -> None:
        self.finish_reason = finish_reason


class _StubServingChat:
    """Minimal OpenAIServingChat surface — only the attributes the
    predicate's extension path reads."""

    enable_auto_tools = True
    tool_parser = _ParserCls

    # Bind the wrapper as an unbound method so __get__ works the same
    # way it does on a real instance.
    _should_check_for_unstreamed_tool_arg_tokens = (
        _wrapped_should_check_for_unstreamed_tool_arg_tokens
    )


_stub = _StubServingChat()

# Sub-case C1: original predicate already True → wrapper passes
# through, does NOT mutate delta_message.
_dm_c1 = DeltaMessage(
    tool_calls=[
        DeltaToolCall(
            index=0,
            function=DeltaFunctionCall(arguments="some args"),
        )
    ]
)
_orig_tool_calls_c1 = _dm_c1.tool_calls
_verdict_c1 = _stub._should_check_for_unstreamed_tool_arg_tokens(
    _dm_c1, _StubOutput(finish_reason="tool_calls")
)
_require(
    _verdict_c1 is True,
    f"Phase 10 case C1 (original True): wrapper returned "
    f"{_verdict_c1!r}, expected True.",
)
_require(
    _dm_c1.tool_calls is _orig_tool_calls_c1,
    "Phase 10 case C1: wrapper mutated tool_calls when original "
    "verdict was True (should pass through unchanged).",
)

# Sub-case C2: truncation rescue path — original False, extension
# fires, delta_message mutated.
_dm_c2 = DeltaMessage()  # tool_calls = [] by default
_require(
    _dm_c2.tool_calls == [],
    f"Phase 10 case C2 setup: DeltaMessage().tool_calls is "
    f"{_dm_c2.tool_calls!r}, expected empty list (Pydantic default).",
)
_verdict_c2 = _stub._should_check_for_unstreamed_tool_arg_tokens(
    _dm_c2, _StubOutput(finish_reason="length")
)
_require(
    _verdict_c2 is True,
    f"Phase 10 case C2 (truncation rescue): wrapper returned "
    f"{_verdict_c2!r}, expected True (extension should fire on "
    f"finish_reason='length' with auto_tools + Qwen3CoderToolParser).",
)
_require(
    isinstance(_dm_c2.tool_calls, list) and len(_dm_c2.tool_calls) == 1,
    f"Phase 10 case C2: tool_calls length is "
    f"{len(_dm_c2.tool_calls) if isinstance(_dm_c2.tool_calls, list) else None}, "
    f"expected 1 (synthetic skeleton mutation).",
)
_synthetic_tc = _dm_c2.tool_calls[0]
_require(
    _synthetic_tc.index == 0
    and _synthetic_tc.function is not None
    and _synthetic_tc.function.arguments == ""
    and _synthetic_tc.function.name is None
    and _synthetic_tc.id is None
    and _synthetic_tc.type is None,
    f"Phase 10 case C2: synthetic skeleton has wrong shape: "
    f"index={_synthetic_tc.index!r}, function.arguments="
    f"{_synthetic_tc.function.arguments if _synthetic_tc.function else None!r}, "
    f"function.name="
    f"{_synthetic_tc.function.name if _synthetic_tc.function else None!r}, "
    f"id={_synthetic_tc.id!r}, type={_synthetic_tc.type!r}.",
)

# Sub-case C3: not the qwen3_coder parser → extension does NOT fire.
class _StubServingChatOtherParser(_StubServingChat):
    tool_parser = type("OtherParser", (), {})  # not Qwen3CoderToolParser


_stub_other = _StubServingChatOtherParser()
_dm_c3 = DeltaMessage()
_verdict_c3 = (
    _StubServingChatOtherParser._should_check_for_unstreamed_tool_arg_tokens(
        _stub_other, _dm_c3, _StubOutput(finish_reason="length")
    )
)
_require(
    _verdict_c3 is False,
    f"Phase 10 case C3 (other parser class): wrapper returned "
    f"{_verdict_c3!r}, expected False (extension is gated on "
    f"tool_parser is Qwen3CoderToolParser).",
)
_require(
    _dm_c3.tool_calls == [],
    f"Phase 10 case C3: wrapper mutated tool_calls for a non-"
    f"Qwen3CoderToolParser request: {_dm_c3.tool_calls!r}.",
)

# Sub-case C4: finish_reason absent → extension does NOT fire (this
# is an intermediate delta, not a final one).
_dm_c4 = DeltaMessage()
_verdict_c4 = _stub._should_check_for_unstreamed_tool_arg_tokens(
    _dm_c4, _StubOutput(finish_reason=None)
)
_require(
    _verdict_c4 is False,
    f"Phase 10 case C4 (no finish_reason): wrapper returned "
    f"{_verdict_c4!r}, expected False.",
)


# --------------------------------------------------------------------
# Behavioural case D: prove the rescue-shape end-to-end. With our
# parser-side update applied (case A), the serving-layer rescue's
# math at serving.py:1149-1154 (expected.replace(actual, "", 1))
# must yield exactly the missing-tail bytes.
# --------------------------------------------------------------------

_recovered_expected = _parser_probe.prev_tool_call_arr[-1]["arguments"]
_recovered_actual = _parser_probe.streamed_args_for_tool[-1]
_recovered_remaining = _recovered_expected.replace(_recovered_actual, "", 1)
_require(
    _recovered_remaining and _recovered_remaining != _recovered_expected,
    f"Phase 10 case D: rescue subtraction produced an empty or "
    f"unchanged string. expected={_recovered_expected!r}, "
    f"actual={_recovered_actual!r}, remaining={_recovered_remaining!r}.",
)
_require(
    "import os" in _recovered_remaining and "import sys" in _recovered_remaining,
    f"Phase 10 case D: rescue remaining tail does not contain the "
    f"partial value bytes that were lost on the wire. "
    f"expected={_recovered_expected!r}, actual={_recovered_actual!r}, "
    f"remaining={_recovered_remaining!r}.",
)
_require(
    _recovered_actual + _recovered_remaining == _recovered_expected,
    f"Phase 10 case D: actual + remaining != expected. The rescue "
    f"subtraction is not a strict-prefix difference; the wire would "
    f"reassemble incorrectly. actual={_recovered_actual!r}, "
    f"remaining={_recovered_remaining!r}, expected={_recovered_expected!r}.",
)


_logger.info(
    "[%s] applied: wrapped %s.extract_tool_calls_streaming and "
    "%s._should_check_for_unstreamed_tool_arg_tokens for vLLM commit %s "
    "(D13 streaming-truncation rescue: parser-side wrap keeps "
    "prev_tool_call_arr in sync with the in-progress partial; "
    "serving-side predicate fires on truncation+auto-tools+"
    "Qwen3CoderToolParser and synthesises the skeleton DeltaToolCall "
    "the rescue body indexes; combined effect makes the streaming-mode "
    "tool_calls[0].function.arguments byte-identical to the "
    "non-streaming counterpart on max_tokens-mid-value truncation).",
    _PATCH_TAG,
    _ParserCls.__qualname__,
    OpenAIServingChat.__qualname__,
    _PINNED_VLLM_COMMIT,
)
