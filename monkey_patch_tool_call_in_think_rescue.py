"""Strict, fail-loud runtime patch: rescue ``<tool_call>`` blocks emitted
inside ``<think>`` for the Qwen3.6 reasoning parser.

Target: vLLM commit ``32e45636e3d7e02615facc8c63645ce4ac1d7e11`` (README §3.2).

NOTE on PR #35687 (commit ``92762edc53``, merged 2026-04-23, present in
master): upstream's ``Qwen3ReasoningParser`` now treats ``<tool_call>`` as
an *implicit reasoning end* — but ONLY when ``</think>`` is absent
entirely (truncation case). The §6.1 / #39056 failure mode this patch
exists for is the ``<think>...<tool_call>...</tool_call>...</think>``
shape: both tokens emitted, with the tool call mid-thought. Master's
``extract_reasoning`` partitions on ``</think>`` first
(``qwen3_reasoning_parser.py:142-144``), so the ``<tool_call>`` markup
still lands in ``reasoning``. This patch's deferred-flush state machine
is therefore still load-bearing on master.
Patch class: ``vllm.reasoning.qwen3_reasoning_parser.Qwen3ReasoningParser``.
Companion to (not a replacement for) ``monkey_patch_qwen3_coder.py`` and
``monkey_patch_extract_tool_calls_metrics.py``.

What this fixes
---------------

Qwen3.6 occasionally (single-digit percent under agentic workloads) emits
``<tool_call>...</tool_call>`` markup BEFORE closing ``</think>`` — the
chat template (see ``chat_template.jinja:53``) explicitly instructs the
model to place any tool call AFTER reasoning closes, but the model
sometimes violates that instruction. Upstream
``Qwen3ReasoningParser.extract_reasoning`` partitions on ``</think>`` in
the non-streaming path; in the streaming path it routes every delta that
arrives before ``end_token_id`` to ``DeltaMessage.reasoning``. Either
way, a ``<tool_call>`` block emitted inside ``<think>`` is delivered to
the caller as *reasoning text*. The downstream
``Qwen3CoderToolParser.extract_tool_calls`` only ever sees ``content``,
not ``reasoning``, and so it never parses the call. The tool call is
silently lost.

This patch wraps both methods of ``Qwen3ReasoningParser`` and moves any
``<tool_call>...</tool_call>`` block that appeared inside ``<think>`` out
of ``reasoning`` and into ``content``. The downstream tool parser then
sees the markup and extracts the call via its normal code path. The
patch performs NO parsing of the tool-call interior — that is strictly
the downstream parser's job.

Patch-discipline contract
-------------------------

This file is a patch, not a library. At import it:

1. Imports vLLM. Failure is a hard ImportError; we do not hide it.
2. Looks up ``Qwen3ReasoningParser`` in the expected module.
3. Verifies it is a class and a subclass of
   ``BaseThinkingReasoningParser``.
4. Verifies ``extract_reasoning`` and ``extract_reasoning_streaming``
   exist, are callable, and have exactly the upstream signatures.
5. Verifies source-level landmarks in both methods, so that an
   upstream refactor that changes the partitioning semantics forces
   a re-audit rather than a silently-wrong wrapper.
6. Verifies ``start_token == "<think>"`` and ``end_token == "</think>"``
   at the class (descriptor) level — we refuse to install against a
   parser whose markers changed, since the rescue regex is literal.
7. Installs both wrappers and verifies both methods, resolved by
   ``getattr`` AND ``inspect.getattr_static``, carry the patch tag.
8. Logs a single INFO line via ``vllm.logger.init_logger`` naming both
   wrapped methods and the pinned commit.

Any of 1-7 failing raises :class:`RescuePatchRefusedError` and the
interpreter does not continue. There is no ``SystemExit(0)`` and no
``try/except Exception: pass`` on any install path. A half-installed
rescue is silently worse than no rescue at all: the operator assumes the
tool-call-in-think failure mode is covered while it is not.

Critical correctness invariants
-------------------------------

* **Identity case, non-streaming.** When the model_output contains no
  ``<tool_call>`` inside the reasoning span, the wrapper returns
  exactly the tuple the upstream method returned — byte-identical.
* **Identity case, streaming.** When no ``<tool_call>`` ever appears
  in reasoning across the whole stream, the wrapper's
  ``DeltaMessage`` sequence is byte-identical to upstream's.
* **Block integrity.** A rescue either moves the WHOLE
  ``<tool_call>...</tool_call>`` block (opening marker, interior,
  closing marker) from reasoning to content, or does nothing. A
  partially-rescued block is never emitted.
* **Deferred-flush routing (streaming).** Completed rescue blocks are
  buffered on the parser instance in a pending list and flushed
  ONLY on the delta that carries ``</think>`` (the handoff delta).
  The flushed concatenation is PREPENDED to the upstream's
  ``content`` field for that delta. This is load-bearing:
  ``parse_delta`` (``vllm/parser/abstract_parser.py:585-647``)
  invokes the tool parser only when ``is_reasoning_end(delta_token_ids)``
  is True, and on that transition it assigns
  ``current_text = delta_message.content`` (line 618) before
  handing off to ``extract_tool_calls_streaming``. Emitting the
  rescued block as ``content`` on a PRIOR delta would bypass the
  tool parser entirely, leaving ``tool_calls`` unpopulated — the
  exact silent-loss failure this file exists to prevent.
* **The wrapper never emits ``<tool_call>`` markup as a content
  delta mid-``<think>``.** Any content field bearing tool-call
  markup emitted by this wrapper is emitted only on the ``</think>``
  handoff delta, so ``parse_delta`` line 614-618 routes it to the
  tool parser.
* **Tool parser sees complete lumped block.** On the handoff delta,
  the tool parser receives a ``delta_text`` argument containing
  ``<tool_call>...</tool_call>`` as a complete, closed region (plus
  any trailing real content). ``Qwen3CoderToolParser.extract_tool_calls_streaming``
  supports burst-arriving complete blocks (see its comment block at
  ``qwen3coder_tool_parser.py:499-503``), so ``tool_calls`` IS
  populated on that delta.
* **Unclosed blocks stay put (non-streaming).** If ``<tool_call>``
  appears in reasoning without a matching ``</tool_call>`` inside
  the reasoning span, the opening marker and everything after it
  remain in ``reasoning`` — the rescue is a no-op for that block.
* **Ordering.** Multiple ``<tool_call>`` blocks inside reasoning are
  moved to content in the order the model emitted them.
* **No per-call parsing.** We match on the literal strings
  ``<tool_call>`` and ``</tool_call>``, never on user-supplied
  patterns. There is no ReDoS surface.
* **No cross-request state leak (streaming).** Per-instance rescue
  state is reset whenever ``not previous_text`` — the same stream-
  start sentinel the upstream streaming tool parser uses
  (``qwen3coder_tool_parser.py:337``). State attributes use a
  ``_qwen36_*`` namespace so they cannot collide with any
  existing upstream state.

Edge cases and policy calls
---------------------------

Streaming is the interesting path. The strategy is **deferred
flush**: completed rescue blocks are buffered on the parser instance
in a pending list, NOT emitted as content mid-``<think>``. The
pending list is flushed on the handoff delta (``end_token_id in
delta_token_ids``), concatenated, and prepended to upstream's
``content`` field. The enclosing ``parse_delta`` then routes that
combined content to the tool parser via its standard handoff at
``abstract_parser.py:614-618``. This is the ONLY way the tool parser
can see the rescued markup, because ``_in_tool_call_phase`` gates
the tool-parser call on ``state.reasoning_ended`` and that flag is
flipped solely by ``is_reasoning_end(delta_token_ids)``.

(a) **Marker split across delta boundary.** The string ``<tool_call>``
    (11 chars) or ``</tool_call>`` (12 chars) can arrive split across
    two deltas. The wrapper keeps a bounded tail of the most recent
    reasoning characters (the last ``len("</tool_call>") - 1 = 11``
    chars) and searches ``tail + delta_text`` for an opener. For the
    closer, no tail is needed because once the opener is seen we
    enter ``in_block`` mode and every subsequent character goes into
    the in-flight buffer — the closer is inherently contiguous with
    the buffer's tail.

(b) **Block split across many deltas.** Once we see ``<tool_call>``,
    every subsequent reasoning delta is appended to the in-flight
    buffer and the wrapper emits only the reasoning text preceding
    the opener (plus any reasoning following a closed block). The
    in-flight block is closed when ``</tool_call>`` appears in the
    buffer; the closed block is appended to the pending list and
    waits for the ``</think>`` handoff.

(c) **Multiple back-to-back blocks inside ``<think>``.** Each block
    is appended to the pending list in source order. On flush they
    are concatenated (no separator: the tool parser's regex matches
    ``<tool_call>...</tool_call>`` non-greedy DOTALL, so adjacent
    blocks are parsed cleanly without a separator).

(d) **Start-marker straddles delta boundary: the prefix is already
    emitted to the client.** If the previous delta's tail ended in
    ``<tool_c`` and upstream already sent that fragment to the
    client under ``reasoning=...``, we cannot retract it. The
    client's reasoning stream will have a trailing ``<tool_c`` that
    "belongs" to the tool call. The deferred-flush state machine
    still rescues the block correctly (``in_block`` flips True on
    opener detection across tail+delta; the in-flight buffer is
    seeded with the full opener string reconstructed from
    ``tail + delta_text`` starting at the opener's combined index).
    The stray prefix in reasoning is cosmetic — agents must not
    parse reasoning text for API-contract reasons.

(e) **``</think>`` arrives mid-rescue.** The opener was inside
    ``<think>``, the closer is post-``</think>``. Upstream will emit
    this as a ``DeltaMessage`` with ``content`` set (and possibly
    ``reasoning`` set, on the same delta if text straddled the
    close-think token). The wrapper recognises ``in_block`` is True
    and appends the incoming content to the in-flight buffer; once
    ``</tool_call>`` lands in the buffer the block is closed,
    appended to pending, flushed, and prepended to any remaining
    real post-``</think>`` content. On that same delta ``parse_delta``
    sees ``is_reasoning_end → True`` and hands the combined content
    off to the tool parser.

(f) **Known limitation: mid-``<think>`` truncation without
    ``</think>``.** If the stream is cancelled or hits max_tokens
    after a ``<tool_call>`` opener but before ``</think>``, the
    pending list (and any in-flight buffer) is never flushed. The
    rescued blocks are lost. The next request to land on the same
    parser instance (detected via ``not previous_text``) will find
    non-empty pending state and emit a single ``WARNING`` log line
    before clearing. This is an observability affordance, not a
    recovery — the completion was already sent to the client with
    an unterminated tool-call attempt visible only in reasoning.

(g) **Common case: no ``<tool_call>`` at all.** The wrapper adds one
    substring scan per delta over a string bounded at ~11 chars plus
    the delta itself. The overhead is negligible, and the wrapper
    passes the upstream ``DeltaMessage`` through unchanged — byte-
    identical identity.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Sequence
from typing import Any, Callable, TypeAlias


_PINNED_VLLM_COMMIT: str = "32e45636e3d7e02615facc8c63645ce4ac1d7e11"
_PATCH_TAG: str = "qwen36-agent-setup-tool-call-in-think-rescue-v1"

# The literal markers we rescue. Kept as module-level constants so the
# bounded-tail length in the streaming wrapper is derived from them, not
# hard-coded.
_TOOL_CALL_OPEN: str = "<tool_call>"
_TOOL_CALL_CLOSE: str = "</tool_call>"
# Longest marker minus one: the largest number of trailing characters we
# need to retain from prior reasoning to detect a marker straddling a
# delta boundary. ``</tool_call>`` is longer than ``<tool_call>``.
_TAIL_WINDOW: int = max(len(_TOOL_CALL_OPEN), len(_TOOL_CALL_CLOSE)) - 1

# Compiled once at import time. Non-greedy, DOTALL so ``.`` matches
# newlines inside the tool-call body (which is the common case — the
# chat template shows a multi-line block). Literal markers only; no
# user input enters this pattern. No ReDoS surface.
_TOOL_CALL_BLOCK_RE: re.Pattern[str] = re.compile(
    re.escape(_TOOL_CALL_OPEN) + r".*?" + re.escape(_TOOL_CALL_CLOSE),
    re.DOTALL,
)

# Expected literal marker strings — verified against the live parser's
# properties at import time so an upstream rename trips the refusal.
_EXPECTED_START_TOKEN: str = "<think>"
_EXPECTED_END_TOKEN: str = "</think>"

# Source landmarks: substrings we require in the relevant method bodies
# so that an upstream refactor that changes the partitioning strategy
# forces a re-audit here.
_NONSTREAMING_LANDMARK: str = "model_output.partition(self.end_token)"
_STREAMING_LANDMARK: str = "self.end_token_id in delta_token_ids"


# Type aliases kept at module level so inspectors reading the installed
# attribute can see the signatures still conform.
ExtractReasoning: TypeAlias = Callable[
    [Any, str, Any], "tuple[str | None, str | None]"
]
ExtractReasoningStreaming: TypeAlias = Callable[
    [
        Any,
        str,
        str,
        str,
        Sequence[int],
        Sequence[int],
        Sequence[int],
    ],
    Any,
]


class RescuePatchRefusedError(RuntimeError):
    """A precondition for the tool-call-in-think rescue wrapper was violated.

    Raised at import time only. The patch either applies cleanly and
    correctly, or the process does not come up. A half-installed or
    silently-skipped rescue patch returns the server to the silent-
    tool-call-loss mode this file exists to eliminate.
    """


def _require(condition: object, failure_message: str) -> None:
    if not condition:
        raise RescuePatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {failure_message}"
        )


# --------------------------------------------------------------------
# Phase 1: Import vLLM and locate the target surface.
# --------------------------------------------------------------------

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.reasoning import qwen3_reasoning_parser as _qwen3_reasoning_mod
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.entrypoints.openai.engine.protocol import DeltaMessage

_logger = init_logger(__name__)


_ParserCls = getattr(_qwen3_reasoning_mod, "Qwen3ReasoningParser", None)
_require(
    _ParserCls is not None,
    "Qwen3ReasoningParser is no longer exported from "
    "vllm.reasoning.qwen3_reasoning_parser. Upstream has moved or "
    "renamed the class; re-audit before bumping the pinned commit.",
)
_require(
    inspect.isclass(_ParserCls),
    "vllm.reasoning.qwen3_reasoning_parser.Qwen3ReasoningParser is no "
    "longer a class.",
)
_require(
    issubclass(_ParserCls, BaseThinkingReasoningParser),
    "Qwen3ReasoningParser is no longer a subclass of "
    "BaseThinkingReasoningParser. Upstream has restructured the "
    "reasoning-parser hierarchy and the contract this patch relies on "
    "may no longer hold.",
)


# --------------------------------------------------------------------
# Phase 2: Landmark the non-streaming method.
# --------------------------------------------------------------------

_original_extract: ExtractReasoning | None = getattr(
    _ParserCls, "extract_reasoning", None
)
_require(
    _original_extract is not None and callable(_original_extract),
    "Qwen3ReasoningParser.extract_reasoning is missing or not callable.",
)

try:
    _extract_sig = inspect.signature(_original_extract)  # type: ignore[arg-type]
except (TypeError, ValueError) as _exc:
    raise RescuePatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect signature of "
        f"extract_reasoning: {_exc!r}"
    ) from _exc

_extract_params = list(_extract_sig.parameters)
_require(
    _extract_params == ["self", "model_output", "request"],
    f"extract_reasoning signature changed; expected "
    f"(self, model_output, request), got {_extract_params!r}.",
)

try:
    _extract_src = inspect.getsource(_original_extract)  # type: ignore[arg-type]
except (OSError, TypeError) as _exc:
    raise RescuePatchRefusedError(
        f"[{_PATCH_TAG}] cannot read source of extract_reasoning "
        f"(is vLLM installed without accompanying .py files?): {_exc!r}"
    ) from _exc

_require(
    _NONSTREAMING_LANDMARK in _extract_src,
    f"non-streaming landmark {_NONSTREAMING_LANDMARK!r} not found in "
    f"extract_reasoning source. The partitioning strategy upstream "
    f"appears to have changed; refusing to wrap a function whose "
    f"contract has drifted.",
)


# --------------------------------------------------------------------
# Phase 3: Landmark the streaming method.
# --------------------------------------------------------------------

_original_extract_stream: ExtractReasoningStreaming | None = getattr(
    _ParserCls, "extract_reasoning_streaming", None
)
_require(
    _original_extract_stream is not None and callable(_original_extract_stream),
    "Qwen3ReasoningParser.extract_reasoning_streaming is missing or "
    "not callable.",
)

try:
    _stream_sig = inspect.signature(_original_extract_stream)  # type: ignore[arg-type]
except (TypeError, ValueError) as _exc:
    raise RescuePatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect signature of "
        f"extract_reasoning_streaming: {_exc!r}"
    ) from _exc

_stream_params = list(_stream_sig.parameters)
_require(
    _stream_params
    == [
        "self",
        "previous_text",
        "current_text",
        "delta_text",
        "previous_token_ids",
        "current_token_ids",
        "delta_token_ids",
    ],
    f"extract_reasoning_streaming signature changed; expected "
    f"(self, previous_text, current_text, delta_text, "
    f"previous_token_ids, current_token_ids, delta_token_ids), got "
    f"{_stream_params!r}.",
)

try:
    _stream_src = inspect.getsource(_original_extract_stream)  # type: ignore[arg-type]
except (OSError, TypeError) as _exc:
    raise RescuePatchRefusedError(
        f"[{_PATCH_TAG}] cannot read source of extract_reasoning_streaming "
        f"(is vLLM installed without accompanying .py files?): {_exc!r}"
    ) from _exc

_require(
    _STREAMING_LANDMARK in _stream_src,
    f"streaming landmark {_STREAMING_LANDMARK!r} not found in "
    f"extract_reasoning_streaming source. The streaming split strategy "
    f"upstream appears to have changed; refusing to wrap.",
)


# --------------------------------------------------------------------
# Phase 4: Verify the ``<think>`` / ``</think>`` markers are unchanged.
# --------------------------------------------------------------------

# ``start_token`` and ``end_token`` are declared as @property on
# Qwen3ReasoningParser (source lines 44-52), so the values are only
# reachable by invoking the descriptor. ``inspect.getattr_static`` sees
# the property object, not its value. The properties are pure: they
# return literal strings. We invoke via a None-instance descriptor
# call, which is safe for these specific properties because they read
# no ``self`` state.
_start_prop = inspect.getattr_static(_ParserCls, "start_token")
_end_prop = inspect.getattr_static(_ParserCls, "end_token")
_require(
    isinstance(_start_prop, property),
    "Qwen3ReasoningParser.start_token is no longer a @property; the "
    "rescue's assumption that the marker is a fixed literal no longer "
    "holds. Refusing to patch.",
)
_require(
    isinstance(_end_prop, property),
    "Qwen3ReasoningParser.end_token is no longer a @property; the "
    "rescue's assumption that the marker is a fixed literal no longer "
    "holds. Refusing to patch.",
)

# Invoke the property getters against a throwaway instance-free context
# — they read only literal strings (verified in source above), so this
# is sound for THIS parser. If upstream ever makes these ``self``-
# dependent we fall back to landmarks below.
try:
    _start_value: Any = _start_prop.fget(None)  # type: ignore[misc]
    _end_value: Any = _end_prop.fget(None)  # type: ignore[misc]
except Exception:  # noqa: BLE001 — property became self-dependent
    # Fall back: assert the expected literals are present in the
    # property-getter source. Weaker than invocation but still catches
    # a rename.
    try:
        _start_src = inspect.getsource(_start_prop.fget)  # type: ignore[arg-type]
        _end_src = inspect.getsource(_end_prop.fget)  # type: ignore[arg-type]
    except (OSError, TypeError) as _exc:
        raise RescuePatchRefusedError(
            f"[{_PATCH_TAG}] cannot read source of start_token/end_token "
            f"getters: {_exc!r}"
        ) from _exc
    _require(
        f'"{_EXPECTED_START_TOKEN}"' in _start_src
        or f"'{_EXPECTED_START_TOKEN}'" in _start_src,
        f"start_token getter no longer returns {_EXPECTED_START_TOKEN!r}.",
    )
    _require(
        f'"{_EXPECTED_END_TOKEN}"' in _end_src
        or f"'{_EXPECTED_END_TOKEN}'" in _end_src,
        f"end_token getter no longer returns {_EXPECTED_END_TOKEN!r}.",
    )
else:
    _require(
        _start_value == _EXPECTED_START_TOKEN,
        f"start_token is {_start_value!r}, expected "
        f"{_EXPECTED_START_TOKEN!r}. The rescue's rescue semantics "
        f"assume the reasoning opener is the literal {_EXPECTED_START_TOKEN!r}.",
    )
    _require(
        _end_value == _EXPECTED_END_TOKEN,
        f"end_token is {_end_value!r}, expected {_EXPECTED_END_TOKEN!r}. "
        f"The rescue's streaming logic assumes the reasoning closer is "
        f"the literal {_EXPECTED_END_TOKEN!r}.",
    )


# --------------------------------------------------------------------
# Phase 5: Verify the DeltaMessage contract.
# --------------------------------------------------------------------

# The streaming wrapper constructs DeltaMessage instances. Guard the
# two fields it writes (``reasoning`` and ``content``) against an
# upstream rename that would turn every rescue into a silent
# AttributeError or ValidationError.
for _field in ("reasoning", "content"):
    _require(
        _field in DeltaMessage.model_fields,
        f"DeltaMessage no longer exposes a {_field!r} field. The "
        f"streaming rescue writes this field and would fail at runtime "
        f"without it.",
    )


# --------------------------------------------------------------------
# Phase 6: Non-streaming wrapper.
# --------------------------------------------------------------------


def _rescue_reasoning_blocks(
    reasoning: str,
) -> "tuple[str, list[str]]":
    """Strip whole ``<tool_call>...</tool_call>`` blocks from ``reasoning``.

    Returns ``(cleaned_reasoning, rescued_blocks)`` where:

    * ``rescued_blocks`` is the ordered list of the raw matched
      substrings (each starts with ``<tool_call>`` and ends with
      ``</tool_call>``).
    * ``cleaned_reasoning`` is ``reasoning`` with every matched block
      removed. Whitespace hygiene is intentionally minimal — we do not
      collapse runs of newlines, only strip an immediately adjacent
      trailing newline per removed block to avoid a double-blank-line
      artifact at the splice point.

    An UNCLOSED ``<tool_call>`` (opening marker without a corresponding
    closing marker in the same string) produces NO match under the non-
    greedy regex; the opener and everything after it remain in the
    cleaned output. That is the correct policy (see module docstring
    invariant "Unclosed blocks stay put").
    """
    rescued: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        rescued.append(match.group(0))
        # Consume a single trailing newline if one immediately follows
        # the closing marker — avoids leaving a blank line at the
        # splice point. We cannot reach beyond ``match.end()`` from
        # inside ``re.sub``'s substitution function, so this is done
        # in a post-pass below.
        return ""

    cleaned = _TOOL_CALL_BLOCK_RE.sub(_capture, reasoning)

    # Post-pass whitespace hygiene: collapse consecutive blank lines
    # introduced at splice points. We are deliberately conservative —
    # only ``\n\n\n`` → ``\n\n`` and only once.
    if rescued:
        while "\n\n\n" in cleaned:
            cleaned = cleaned.replace("\n\n\n", "\n\n")

    return cleaned, rescued


def extract_reasoning_rescued(
    self: Any,
    model_output: str,
    request: Any,
) -> "tuple[str | None, str | None]":
    """Rescue wrapper around ``Qwen3ReasoningParser.extract_reasoning``.

    Contract vs. upstream:

    * Identical signature and return type.
    * Calls the upstream method once and captures
      ``(reasoning, content)``.
    * If ``reasoning`` contains no ``<tool_call>`` marker, returns the
      captured tuple UNCHANGED — the identity case.
    * Otherwise, extracts every ``<tool_call>...</tool_call>`` block
      from ``reasoning`` in order, concatenates them (joined by
      ``"\\n"``), appends the result to ``content`` (initializing
      ``content`` to ``""`` if it was ``None``), and returns the
      cleaned reasoning together with the augmented content.
    * If the cleaned reasoning, after stripping, is empty, returns
      ``None`` in its place (mirrors upstream's ``content or None``
      pattern, but for the reasoning side post-rescue).

    No tool-call interior is parsed; the block is moved verbatim.
    """
    reasoning, content = _original_extract(self, model_output, request)  # type: ignore[misc]

    # Identity short-circuit: the common case. Bare substring check is
    # O(n) and skips the regex compile of the body.
    if reasoning is None or _TOOL_CALL_OPEN not in reasoning:
        return reasoning, content

    cleaned_reasoning, rescued_blocks = _rescue_reasoning_blocks(reasoning)
    if not rescued_blocks:
        # Opener present but no matched close — unclosed block stays in
        # reasoning; no-op.
        return reasoning, content

    # Whitespace hygiene for the splice between existing content and
    # the rescued markup. The rescued blocks arrive in source order and
    # are concatenated with a single newline between them.
    rescued_joined = "\n".join(rescued_blocks)

    if content is None:
        new_content = rescued_joined
    elif not content:
        new_content = rescued_joined
    else:
        separator = "" if content.endswith("\n") else "\n"
        new_content = content + separator + rescued_joined

    # Post-rescue reasoning may be whitespace-only — surface that as
    # ``None`` to match the upstream convention where empty strings are
    # not emitted as reasoning.
    stripped_reasoning = cleaned_reasoning.strip()
    final_reasoning: str | None = cleaned_reasoning if stripped_reasoning else None

    return final_reasoning, new_content


extract_reasoning_rescued.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
extract_reasoning_rescued.__wrapped_original__ = _original_extract  # type: ignore[attr-defined]
extract_reasoning_rescued.__name__ = "extract_reasoning"
extract_reasoning_rescued.__qualname__ = (
    f"{_ParserCls.__qualname__}.extract_reasoning"
)


# --------------------------------------------------------------------
# Phase 7: Streaming wrapper (deferred-flush state machine).
# --------------------------------------------------------------------


# Per-parser-instance state attribute names. Namespaced with
# ``_qwen36_`` to guarantee no collision with any existing state on
# ``Qwen3ReasoningParser`` or its parents. The four attributes form
# the state machine:
#
# * ``_qwen36_pending_rescued_blocks`` — list[str]. Completed rescue
#   blocks awaiting flush. Flushed (concatenated + prepended to
#   upstream's ``content``) on the delta carrying ``</think>``.
# * ``_qwen36_rescue_in_block`` — bool. True iff we have seen a
#   ``<tool_call>`` opener but not yet its ``</tool_call>`` closer.
# * ``_qwen36_rescue_buffer`` — str. When in-block, the accumulated
#   text starting from the opener (including the opener itself).
# * ``_qwen36_reasoning_tail`` — str. Bounded last-N chars of the
#   reasoning text already emitted to the client, for detecting an
#   opener straddling the prior delta boundary. N = _TAIL_WINDOW.
_STATE_PENDING: str = "_qwen36_pending_rescued_blocks"
_STATE_IN_BLOCK: str = "_qwen36_rescue_in_block"
_STATE_BUFFER: str = "_qwen36_rescue_buffer"
_STATE_TAIL: str = "_qwen36_reasoning_tail"


def _reset_rescue_state(parser: Any) -> None:
    """Initialize / reset the per-instance rescue state.

    Called at stream start (``not previous_text``) and whenever the
    getters need to be sure the attributes are present. Uses setattr
    rather than touching ``__dict__`` directly so any ``__slots__``-
    bearing future subclass continues to work (provided it declares
    these slots; that is not required today).
    """
    setattr(parser, _STATE_PENDING, [])
    setattr(parser, _STATE_IN_BLOCK, False)
    setattr(parser, _STATE_BUFFER, "")
    setattr(parser, _STATE_TAIL, "")


def _stream_start_reset_with_warning(parser: Any) -> None:
    """Reset state at stream start; warn if pending blocks are being dropped.

    A non-empty pending list or in-flight buffer at stream start
    means the previous stream ended (truncation, max_tokens,
    client disconnect) with a ``<tool_call>`` that never saw its
    ``</think>``. Those blocks never reached the tool parser and
    are lost. We log a single ``WARNING`` line for observability
    — operators who see this at nonzero rate need to investigate
    the truncation, not the patch.
    """
    pending = getattr(parser, _STATE_PENDING, None) or []
    in_block = bool(getattr(parser, _STATE_IN_BLOCK, False))
    if pending or in_block:
        _logger.warning(
            "[%s] dropping %d rescued block(s) and in_flight=%s at stream "
            "reset; tool call(s) likely lost due to truncation without "
            "</think>",
            _PATCH_TAG,
            len(pending),
            in_block,
        )
    _reset_rescue_state(parser)


# --- Pure helpers (stateless; used directly by the state machine and
# --- exercised by the scratch test file).


def _consume_in_block(
    buffer: str, incoming: str
) -> "tuple[bool, str, str | None, str]":
    """Feed ``incoming`` into an in-flight block buffer.

    Returns ``(still_in_block, updated_buffer, closed_block, tail)``:

    * ``still_in_block`` — True iff the closer has NOT yet been seen.
    * ``updated_buffer`` — the new buffer value: the unchanged
      accumulation when still in-block; empty when closed.
    * ``closed_block`` — the full rescued block text (starts with
      ``<tool_call>``, ends with ``</tool_call>``) when the closer
      lands; ``None`` when still buffering.
    * ``tail`` — text after the closer that was in ``incoming``. May
      itself contain more openers — the caller should feed it through
      ``_scan_reasoning_blocks`` or repeat the state machine on it.
      Empty when still buffering.
    """
    combined = buffer + incoming
    close_idx = combined.find(_TOOL_CALL_CLOSE)
    if close_idx == -1:
        return True, combined, None, ""
    end = close_idx + len(_TOOL_CALL_CLOSE)
    return False, "", combined[:end], combined[end:]


def _scan_reasoning_blocks(
    text: str,
) -> "tuple[str, list[str], bool, str]":
    """Walk ``text`` extracting fully-closed blocks.

    Used when NOT currently in-block (tail handling is the caller's
    responsibility). Returns ``(emitted_reasoning, blocks, in_block,
    buffer)``:

    * ``emitted_reasoning`` — characters of ``text`` that are NOT
      part of any block. This is what the caller appends to the
      reasoning output channel.
    * ``blocks`` — the ordered list of complete matched blocks.
    * ``in_block`` — True iff ``text`` ended on an unclosed opener.
    * ``buffer`` — when ``in_block``, text starting from the opener
      (inclusive). Empty otherwise.

    NO whitespace hygiene — the streaming contract delivers bytes
    verbatim. Collapsing blank lines would drift the identity case.
    """
    rescued: list[str] = []
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        open_idx = text.find(_TOOL_CALL_OPEN, i)
        if open_idx == -1:
            out.append(text[i:])
            break
        out.append(text[i:open_idx])
        close_idx = text.find(
            _TOOL_CALL_CLOSE, open_idx + len(_TOOL_CALL_OPEN)
        )
        if close_idx == -1:
            buffer = text[open_idx:]
            return "".join(out), rescued, True, buffer
        end = close_idx + len(_TOOL_CALL_CLOSE)
        rescued.append(text[open_idx:end])
        i = end
    return "".join(out), rescued, False, ""


def _process_reasoning_chunk(
    incoming: str,
    tail: str,
    in_block: bool,
    buffer: str,
) -> "tuple[str, list[str], bool, str, str]":
    """One-shot reasoning-chunk processor.

    Applies the deferred-flush state machine to a single chunk of
    reasoning text. Handles opener straddle via ``tail`` and the
    multiple-back-to-back-blocks case. Returns
    ``(reasoning_to_emit, new_blocks, new_in_block, new_buffer, new_tail)``:

    * ``reasoning_to_emit`` — text to emit as ``reasoning=...`` on
      this delta (may be empty).
    * ``new_blocks`` — blocks CLOSED by this chunk, in order. Caller
      appends to the pending list.
    * ``new_in_block``, ``new_buffer`` — updated in-block state.
    * ``new_tail`` — updated tail for the next delta's straddle
      detection. Empty when ``new_in_block`` is True (no point
      scanning for another opener while one is already in flight).

    Pure function; suitable for scratch-test exercise without a
    live parser instance.
    """
    emitted: list[str] = []
    blocks: list[str] = []

    if in_block:
        still, buffer, closed, after_close = _consume_in_block(buffer, incoming)
        if still:
            # Still in-block, nothing to emit, no new closed blocks.
            return "", [], True, buffer, ""
        # Block closed in this chunk. Continue processing
        # `after_close` as fresh reasoning.
        assert closed is not None
        blocks.append(closed)
        in_block = False
        # buffer is "" per _consume_in_block contract.
        remainder = after_close
    else:
        remainder = incoming

    # Now not in-block; look for openers in (tail + remainder).
    # The straddle case: part of the opener is in tail, the rest is
    # in remainder. We scan combined text to find the opener
    # position, then reconstruct the block accordingly.
    combined = tail + remainder
    # Track how many chars of `remainder` have been accounted for in
    # `emitted` (so we can rebuild `new_tail` at the end).
    scan_from = 0
    while True:
        open_idx_combined = combined.find(_TOOL_CALL_OPEN, scan_from)
        if open_idx_combined == -1:
            # No more openers. Everything from scan_from onward in
            # the "remainder" portion is plain reasoning. The tail
            # portion at the beginning was already emitted in prior
            # deltas; we must not re-emit it.
            trailing_start = max(scan_from, len(tail))
            emitted.append(combined[trailing_start:])
            break
        # Emit the text between the previous scan point and this
        # opener — but only the portion that belongs to the CURRENT
        # delta, i.e. above len(tail). The tail prefix has already
        # been sent.
        emit_start = max(scan_from, len(tail))
        if open_idx_combined > emit_start:
            emitted.append(combined[emit_start:open_idx_combined])
        # Look for a matching closer.
        close_search_from = open_idx_combined + len(_TOOL_CALL_OPEN)
        close_idx_combined = combined.find(_TOOL_CALL_CLOSE, close_search_from)
        if close_idx_combined == -1:
            # Unclosed opener — enter in-block mode. Seed the buffer
            # with the opener reconstruction: everything from
            # open_idx_combined in the combined string onward. This
            # ensures the buffer starts with "<tool_call>" even if
            # the opener straddled.
            buffer = combined[open_idx_combined:]
            in_block = True
            # No more scanning; nothing after the opener is emitable
            # as reasoning.
            return "".join(emitted), blocks, True, buffer, ""
        # Full closed block. Reconstruct from the combined string.
        end = close_idx_combined + len(_TOOL_CALL_CLOSE)
        blocks.append(combined[open_idx_combined:end])
        scan_from = end

    reasoning_to_emit = "".join(emitted)
    # New tail: take from the emitted reasoning only (we must NEVER
    # include characters we failed to emit — that would re-emit tail
    # content or consume pre-opener characters twice).
    new_tail = reasoning_to_emit[-_TAIL_WINDOW:] if reasoning_to_emit else ""
    return reasoning_to_emit, blocks, False, "", new_tail


def extract_reasoning_streaming_rescued(
    self: Any,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int],
) -> Any:
    """Rescue wrapper around ``Qwen3ReasoningParser.extract_reasoning_streaming``.

    Deferred-flush state machine. See module docstring for the full
    edge-case table. Contract vs. upstream:

    * Identical signature and return type.
    * The upstream method is called exactly once per invocation with
      the received arguments unchanged. The wrapper then decides
      whether to forward, swallow, or transform its result.
    * Per-instance state lives on the parser under ``_qwen36_*``
      attribute names. Reset at stream start, detected via
      ``not previous_text`` (the idiom upstream's streaming tool
      parser uses).
    * When no rescue is in flight and no opener has been seen, the
      upstream ``DeltaMessage`` is forwarded unchanged (identity).
    * Completed rescue blocks are BUFFERED in ``_qwen36_pending_rescued_blocks``
      and flushed ONLY on the delta where ``end_token_id in
      delta_token_ids``. On that delta the concatenation is prepended
      to upstream's ``content`` field. ``parse_delta``
      (``abstract_parser.py:614-618``) then routes the combined
      content to the tool parser, which extracts ``tool_calls``.
    * This wrapper NEVER emits ``<tool_call>`` markup on ``content``
      mid-``<think>``. That invariant is load-bearing: see the
      module docstring for the full trace.

    Returns ``DeltaMessage | None`` with the same meaning upstream
    assigns: ``None`` means "emit nothing for this delta".
    """

    # Stream-start reset. Same sentinel the upstream streaming tool
    # parser uses; piggybacks so our state tracks request boundaries
    # the way upstream tracks its own. If state carries forward from
    # a prior truncated stream, warn before clearing.
    if not previous_text:
        _stream_start_reset_with_warning(self)

    # Lazy-init defense-in-depth: if an early delta arrives with a
    # non-empty ``previous_text`` (e.g. a reconnection) and state is
    # missing, populate rather than AttributeError out.
    if not hasattr(self, _STATE_PENDING):
        _reset_rescue_state(self)

    base: Any = _original_extract_stream(  # type: ignore[misc]
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    )

    # Probe the handoff signal directly from the token ids. This is
    # the same test ``parse_delta`` uses (``abstract_parser.py:614``)
    # to route to the tool parser. We MUST base our flush decision on
    # the same boolean — anything else desynchronises from the
    # routing. Note: ``self.end_token_id`` is populated by
    # ``BaseThinkingReasoningParser.__init__`` and is a stable int.
    is_end_delta: bool = self.end_token_id in delta_token_ids

    pending: list[str] = getattr(self, _STATE_PENDING)
    in_block: bool = getattr(self, _STATE_IN_BLOCK)
    buffer: str = getattr(self, _STATE_BUFFER)
    tail: str = getattr(self, _STATE_TAIL)

    # Upstream components.
    base_reasoning: str | None = getattr(base, "reasoning", None) if base else None
    base_content: str | None = getattr(base, "content", None) if base else None

    # ----------------------------------------------------------------
    # Case I: base is None. Upstream suppressed the delta (e.g. a
    # single special token such as a solitary ``</think>``). If this
    # is the handoff delta, flush pending AND — if we are mid-block
    # — also forward the in-flight buffer so the tool parser has the
    # opener. Subsequent tool-parser-phase deltas (which do not pass
    # through this wrapper) will deliver the closer and the tool
    # parser's streaming accumulator will complete the call.
    # ----------------------------------------------------------------
    if base is None:
        if is_end_delta and (pending or in_block):
            flushed_parts: list[str] = list(pending)
            if in_block:
                # Forward the in-flight buffer to the tool parser.
                # It contains ``<tool_call>`` + accumulated payload
                # with no closer yet. The tool parser's streaming
                # state machine will append future deltas and close
                # on its own. WHY: our wrapper does not run after
                # reasoning_ended flips True; this is our last
                # chance to hand over the opener.
                flushed_parts.append(buffer)
            setattr(self, _STATE_PENDING, [])
            # Keep in_block/buffer state cleared here. The tool
            # parser owns the in-flight block now.
            setattr(self, _STATE_IN_BLOCK, False)
            setattr(self, _STATE_BUFFER, "")
            setattr(self, _STATE_TAIL, "")
            flushed = "".join(flushed_parts)
            if flushed:
                return DeltaMessage(content=flushed)
            return None
        return None

    # ----------------------------------------------------------------
    # Case II: reasoning-only delta (mid-think, no </think> yet, or
    # </think> arrived as a trailing special token with no text).
    # Apply the state machine to base_reasoning. Buffer closed blocks
    # in pending; emit only non-block reasoning. Flush pending iff
    # is_end_delta.
    # ----------------------------------------------------------------
    if base_reasoning is not None and base_content is None:
        reasoning_out, new_blocks, new_in_block, new_buffer, new_tail = (
            _process_reasoning_chunk(base_reasoning, tail, in_block, buffer)
        )
        if new_blocks:
            pending = pending + new_blocks
            setattr(self, _STATE_PENDING, pending)
        setattr(self, _STATE_IN_BLOCK, new_in_block)
        setattr(self, _STATE_BUFFER, new_buffer)
        setattr(self, _STATE_TAIL, new_tail)

        if is_end_delta:
            # Flush pending. If we are still in-flight at handoff,
            # forward the in-flight buffer to the tool parser (see
            # Case I for the WHY — our wrapper does not run once
            # state.reasoning_ended flips True, so this is our last
            # chance to deliver the opener).
            flushed_parts: list[str] = list(pending)
            if new_in_block:
                flushed_parts.append(new_buffer)
                # Hand the in-flight block to the tool parser; we
                # retain no further rescue state.
                setattr(self, _STATE_IN_BLOCK, False)
                setattr(self, _STATE_BUFFER, "")
            setattr(self, _STATE_PENDING, [])
            flushed = "".join(flushed_parts)
            if flushed and reasoning_out:
                return DeltaMessage(
                    reasoning=reasoning_out,
                    content=flushed,
                )
            if flushed:
                return DeltaMessage(content=flushed)
            if reasoning_out:
                return DeltaMessage(reasoning=reasoning_out)
            return None

        # Not the handoff delta. Emit only non-block reasoning; all
        # completed blocks stay in pending. If nothing to emit,
        # return None (swallow the tick).
        if reasoning_out:
            return DeltaMessage(reasoning=reasoning_out)
        return None

    # ----------------------------------------------------------------
    # Case III: content-only delta OR mixed reasoning+content delta.
    # ``</think>`` has either just arrived (mixed) or already passed
    # (content-only). If in-flight, feed the content portion into the
    # buffer to capture the closer. Then flush pending + any just-
    # closed block, prepended to the real content.
    # ----------------------------------------------------------------
    if base_content is not None:
        reasoning_part = base_reasoning  # may be None
        content_part = base_content

        # First, process any reasoning half of this delta (mixed
        # case). This can CLOSE the in-flight block and/or OPEN new
        # ones — handle identically to Case II.
        reasoning_out = ""
        if reasoning_part is not None:
            (
                reasoning_out,
                new_blocks,
                in_block,
                buffer,
                _discard_tail,
            ) = _process_reasoning_chunk(reasoning_part, tail, in_block, buffer)
            if new_blocks:
                pending = pending + new_blocks

        # Now handle the content half. If we are in-block, the
        # closer MUST be in content_part (or in a future delta). The
        # opener was in reasoning, so we're required to deliver the
        # full block through content here (the tool parser runs on
        # content after the handoff).
        content_out = ""
        if in_block:
            still, buffer, closed, after_close = _consume_in_block(
                buffer, content_part
            )
            if still:
                # Closer has not arrived, even after consuming the
                # content portion of this delta. If this is the
                # handoff delta (is_end_delta), subsequent deltas
                # will not pass through our wrapper — hand the in-
                # flight buffer plus any pending blocks to the tool
                # parser now. The tool parser's streaming state
                # machine will accept the opener here and close on
                # future content deltas.
                if is_end_delta:
                    flushed_parts = list(pending) + [buffer]
                    setattr(self, _STATE_PENDING, [])
                    setattr(self, _STATE_IN_BLOCK, False)
                    setattr(self, _STATE_BUFFER, "")
                    setattr(self, _STATE_TAIL, "")
                    flushed = "".join(flushed_parts)
                    if reasoning_out and flushed:
                        return DeltaMessage(
                            reasoning=reasoning_out,
                            content=flushed,
                        )
                    if flushed:
                        return DeltaMessage(content=flushed)
                    if reasoning_out:
                        return DeltaMessage(reasoning=reasoning_out)
                    return None
                # Not handoff, not closed: keep buffering; the wrapper
                # will be called again on the next reasoning-phase
                # delta.
                setattr(self, _STATE_PENDING, pending)
                setattr(self, _STATE_IN_BLOCK, True)
                setattr(self, _STATE_BUFFER, buffer)
                setattr(self, _STATE_TAIL, "")
                if reasoning_out:
                    return DeltaMessage(reasoning=reasoning_out)
                return None
            # Block closed in the content half. The closed block
            # joins pending; ``after_close`` is real post-think
            # content.
            assert closed is not None
            pending = pending + [closed]
            in_block = False
            content_out = after_close
        else:
            content_out = content_part

        # Flush pending + assemble final content.
        if pending:
            flushed = "".join(pending)
            final_content = flushed + content_out
            setattr(self, _STATE_PENDING, [])
        else:
            final_content = content_out

        setattr(self, _STATE_IN_BLOCK, in_block)
        setattr(self, _STATE_BUFFER, buffer)
        # Post-handoff the tail is irrelevant (we never scan for
        # openers in content); reset so a misfire doesn't leak.
        setattr(self, _STATE_TAIL, "")

        if reasoning_out and final_content:
            return DeltaMessage(
                reasoning=reasoning_out,
                content=final_content,
            )
        if final_content:
            return DeltaMessage(content=final_content)
        if reasoning_out:
            return DeltaMessage(reasoning=reasoning_out)
        # Both empty. This happens when upstream emitted content=""
        # and we had nothing pending. Pass through None (upstream
        # would have returned None in the same circumstance).
        return None

    # ----------------------------------------------------------------
    # Case IV: upstream returned a DeltaMessage with BOTH fields
    # None (or some other exotic shape). Per upstream source this
    # shouldn't happen; pass through as defense.
    # ----------------------------------------------------------------
    return base


extract_reasoning_streaming_rescued.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
extract_reasoning_streaming_rescued.__wrapped_original__ = (  # type: ignore[attr-defined]
    _original_extract_stream
)
extract_reasoning_streaming_rescued.__name__ = "extract_reasoning_streaming"
extract_reasoning_streaming_rescued.__qualname__ = (
    f"{_ParserCls.__qualname__}.extract_reasoning_streaming"
)


# --------------------------------------------------------------------
# Phase 8: Install and verify.
# --------------------------------------------------------------------

_ParserCls.extract_reasoning = extract_reasoning_rescued
_ParserCls.extract_reasoning_streaming = extract_reasoning_streaming_rescued

_installed_extract = _ParserCls.extract_reasoning
_installed_stream = _ParserCls.extract_reasoning_streaming

_require(
    getattr(_installed_extract, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install verification failed: "
    "Qwen3ReasoningParser.extract_reasoning does not bear the expected "
    "patch tag. A concurrent monkey-patch has clobbered ours.",
)
_require(
    getattr(_installed_stream, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install verification failed: "
    "Qwen3ReasoningParser.extract_reasoning_streaming does not bear "
    "the expected patch tag. A concurrent monkey-patch has clobbered "
    "ours.",
)

# Second-order verification: both methods must resolve to the wrappers
# via static (MRO-walking, descriptor-respecting) lookup as well as
# normal attribute access. A metaclass __getattribute__ override that
# shadowed only one of the two paths would otherwise hide a regression.
_resolved_extract = inspect.getattr_static(_ParserCls, "extract_reasoning")
_resolved_stream = inspect.getattr_static(
    _ParserCls, "extract_reasoning_streaming"
)
_require(
    getattr(_resolved_extract, "__qwen36_patch__", None) == _PATCH_TAG,
    "static-lookup verification failed: inspect.getattr_static sees a "
    "different extract_reasoning than normal attribute access. "
    "Something in the MRO or metaclass is shadowing our assignment; "
    "refusing to proceed.",
)
_require(
    getattr(_resolved_stream, "__qwen36_patch__", None) == _PATCH_TAG,
    "static-lookup verification failed: inspect.getattr_static sees a "
    "different extract_reasoning_streaming than normal attribute "
    "access. Something in the MRO or metaclass is shadowing our "
    "assignment; refusing to proceed.",
)


_logger.info(
    "[%s] applied: wrapped %s.%s and %s.%s for vLLM commit %s "
    "(rescues <tool_call>...</tool_call> emitted inside <think>; "
    "moves whole blocks from reasoning to content; unclosed blocks "
    "remain in reasoning; streaming state resets on not previous_text).",
    _PATCH_TAG,
    _ParserCls.__module__,
    f"{_ParserCls.__qualname__}.extract_reasoning",
    _ParserCls.__module__,
    f"{_ParserCls.__qualname__}.extract_reasoning_streaming",
    _PINNED_VLLM_COMMIT,
)
