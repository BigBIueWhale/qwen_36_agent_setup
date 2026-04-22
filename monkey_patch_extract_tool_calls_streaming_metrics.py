"""Server-side observability patch for vLLM ``qwen3_coder`` silent streaming tool-parser failures.

Target: vLLM commit ``8936118134d0547fa1cc78adab2d03edd6d3dc48`` (README §3.2).
Streaming counterpart to ``monkey_patch_extract_tool_calls_metrics.py``
(§7.6, non-streaming). This file is §7.7. Same fail-loud import-time
discipline as every other patch in this repo.

What this fixes
---------------

This is the **streaming counterpart** to the non-streaming markup-leak
observability patch. The non-streaming patch wraps
``Qwen3CoderToolParser.extract_tool_calls`` and fires when
``tools_called=False`` is returned alongside leaked tool-call markup in
``content``. This patch wraps
``Qwen3CoderToolParser.extract_tool_calls_streaming`` and fires when a
streamed ``DeltaMessage`` carries leaked tool-call markup in
``content`` while its ``tool_calls`` list is empty — the streaming
equivalent of the same silent-failure shape.

The two patches SHARE a single Prometheus counter
(``vllm_qwen3_coder_silent_tool_call_failures_total``) with labels
``(failure_kind, model)``; this patch uses
``failure_kind="markup_leak_streaming"`` so operators can query the
combined rate across both code paths or split them as needed.

README §6.13's catalog lists 6 streaming silent-failure paths against 8
non-streaming ones. The non-streaming patch's docstring noted that
observing the streaming shape required per-stream state and was
deliberately out of scope. This patch relaxes that stance for the
specific sub-case the existing streaming code ACTUALLY exhibits: a
single delta whose ``content`` already contains a complete
``<tool_call>`` / ``<function=`` / ``<parameter=`` marker. That is a
single-delta property, detectable without cross-delta accumulation.
Everything else — split markers, state-machine divergence across a
stream, the ``finish_reason`` / ``json_invalid`` / object-shape
classes — remains out of scope, unchanged from the non-streaming
patch's stance. Patch 3 (``client/validate_response.py``) is still the
source of truth for end-to-end detection.

Scope: ``markup_leak_streaming`` only
-------------------------------------

This patch detects exactly one failure class:

* ``markup_leak_streaming`` — a ``DeltaMessage`` returned from
  ``extract_tool_calls_streaming`` with a non-empty ``str`` ``content``
  whose text contains a COMPLETE ``<tool_call>``, ``<function=``, or
  ``<parameter=`` marker, AND an empty/falsy ``tool_calls`` list.
  **Detected here.**

Explicit non-scope:

* **Partial markers across delta boundaries.** A delta with content
  ``"<too"`` followed by a delta with ``"l_call>..."`` will NOT fire.
  Cross-delta accumulation is deliberately not implemented — the goal
  is "spot a leak at the moment it happens", not "never miss". Missed
  split-marker events are acceptable observability noise, NOT a
  correctness bug. Patch 3 still catches the aggregated symptom
  client-side.
* **Truncated / malformed tool calls, json_invalid, object-shape.**
  Same reasoning as the non-streaming patch: these are added by code
  downstream of ``extract_tool_calls_streaming`` and invisible from
  inside the parser. Patch 3 handles them.

Patch-discipline contract
-------------------------

This file is a patch, not a library. At import it:

1. Imports vLLM. Failure is a hard ImportError; we do not hide it.
2. Looks up ``Qwen3CoderToolParser`` in the expected module.
3. Verifies ``extract_tool_calls_streaming`` exists, is callable, and
   has exactly the signature ``(self, previous_text, current_text,
   delta_text, previous_token_ids, current_token_ids, delta_token_ids,
   request)``.
4. Verifies ``DeltaMessage`` is importable at the expected path and
   exposes the attributes (``content``, ``tool_calls``) the wrapper
   reads.
5. Reads the method's source and verifies the landmark
   ``return DeltaMessage(content=`` (proves the leaking emit shape
   still exists, line 416 in the pinned source) is present.
6. Optionally validates ``prometheus_client``. Degradation is
   acceptable: this patch is observation-only. On collision with an
   already-registered counter (the common case — the non-streaming
   patch beat us to registration), DISCOVER and REUSE the existing
   collector via ``prometheus_client.REGISTRY._names_to_collectors``
   so both patches increment the same series.
7. Installs the wrapper, tags it with ``__qwen36_patch__``, and
   verifies both ``getattr`` and ``inspect.getattr_static`` resolve
   to the wrapped function bearing the tag.
8. Logs a single INFO line via ``vllm.logger.init_logger`` naming the
   class, the wrapped method, the Prometheus counter status, and the
   pinned commit.

Any of 1-5 or 7 failing raises
:class:`StreamingMetricsPatchRefusedError` and the interpreter does
not continue. There is **no** ``SystemExit(0)`` or
``try/except Exception: pass`` on any install path. The single
exception is the per-delta instrumentation block inside the wrapper —
documented at its site — which catches ``Exception`` so that a fault
in observability never takes down a real request.

Critical correctness invariants
-------------------------------

* **Observation only.** The wrapper calls the original method first,
  captures the result, and **returns the captured result unchanged**
  on every code path. It never alters ``content`` or ``tool_calls``,
  nor does it construct or replace the ``DeltaMessage``. If this
  invariant is ever broken, the patch has failed its single purpose.
* **The original-method call is NOT inside the try/except.** Real
  parser exceptions (including the genuinely frequent streaming-path
  ``AttributeError`` / ``IndexError`` classes) propagate normally —
  same behavior as if this patch were not loaded.
* **Complete-marker-only detection.** Partial markers MUST NOT match;
  see :func:`_contains_complete_marker`. A delta is classified as a
  leak only when a full marker literal is substring-present in its
  ``content``.
* **Shared counter invariant.** On collision, we reuse the
  already-registered counter by name lookup in the default registry
  (``prometheus_client.REGISTRY._names_to_collectors``). Both the
  non-streaming and streaming patches increment the same ``Counter``
  instance; distinguishing the paths is the job of the
  ``failure_kind`` label, not of two separate counters.
* **Per-delta counting.** The metric counts *deltas* exhibiting the
  silent-failure pattern, not *number of dropped tool calls* and not
  *streams containing a leak*. A single leaked marker split across
  deltas (which we won't catch at all) versus a single delta carrying
  two markers (which we count once, but log ``marker_count=2``) is
  the granularity this metric exposes.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypeAlias


_PINNED_VLLM_COMMIT: str = "8936118134d0547fa1cc78adab2d03edd6d3dc48"
_PATCH_TAG: str = "qwen36-agent-setup-extract-tool-calls-streaming-metrics-v1"

# Markers whose presence, as a COMPLETE substring, inside a single
# delta's ``content`` constitutes a streaming markup leak when paired
# with an empty ``tool_calls``. Same literal set as the non-streaming
# patch and ``client/validate_response.py`` — intentional, so operators
# grepping for these see one consistent alphabet across all three
# detection surfaces.
_MARKUP_MARKERS: tuple[str, ...] = ("<tool_call>", "<function=", "<parameter=")

# Source landmark proving the leaking emit shape is still the one we
# expect. The pinned source at line 416 reads
# ``return DeltaMessage(content=delta_text)`` — the "normal content, no
# tool call" branch, which is precisely the branch that leaks when the
# state machine mis-classifies tool-call markup as plain content. One
# landmark, load-bearing for our heuristic: if this literal no longer
# appears, the streaming emit shape has changed enough that the
# wrapper cannot safely claim to observe what it claims to observe.
_STREAMING_EMIT_LANDMARK: str = "return DeltaMessage(content="


# Type alias for the method we wrap. Kept at module level so callers
# inspecting the installed attribute can tell by signature that it
# still conforms.
ExtractToolCallsStreaming: TypeAlias = Callable[
    [Any, str, str, str, Any, Any, Any, Any], Any
]


class StreamingMetricsPatchRefusedError(RuntimeError):
    """A precondition for the extract_tool_calls_streaming metrics wrapper was violated.

    Raised at import time only. The patch either applies cleanly and
    correctly, or the process does not come up. A half-installed or
    silently-skipped observability patch is the worst of both worlds:
    the operator believes they have visibility they do not actually
    have. We refuse to enter that state.
    """


def _require(condition: object, failure_message: str) -> None:
    if not condition:
        raise StreamingMetricsPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {failure_message}"
        )


# --------------------------------------------------------------------
# Phase 1: Import vLLM and locate the target surface.
# --------------------------------------------------------------------

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.logger import init_logger
from vllm.tool_parsers import qwen3coder_tool_parser as _qwen3coder_mod
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
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
# Phase 2: Landmark the method we intend to wrap.
# --------------------------------------------------------------------

_original: ExtractToolCallsStreaming | None = getattr(
    _ParserCls, "extract_tool_calls_streaming", None
)
_require(
    _original is not None and callable(_original),
    "Qwen3CoderToolParser.extract_tool_calls_streaming is missing or "
    "not callable.",
)

try:
    _sig = inspect.signature(_original)  # type: ignore[arg-type]
except (TypeError, ValueError) as _exc:
    raise StreamingMetricsPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect signature of "
        f"extract_tool_calls_streaming: {_exc!r}"
    ) from _exc

_param_names = list(_sig.parameters)
_EXPECTED_PARAMS = [
    "self",
    "previous_text",
    "current_text",
    "delta_text",
    "previous_token_ids",
    "current_token_ids",
    "delta_token_ids",
    "request",
]
_require(
    _param_names == _EXPECTED_PARAMS,
    f"extract_tool_calls_streaming signature changed; expected "
    f"{_EXPECTED_PARAMS!r}, got {_param_names!r}.",
)


# --------------------------------------------------------------------
# Phase 3: Landmark the result type we read from.
# --------------------------------------------------------------------

# We don't construct DeltaMessage; we only read two attributes
# (``content`` and ``tool_calls``) off the upstream-built instance.
# But verifying the type still carries those fields guards against an
# upstream rename turning every wrapper invocation into a silent
# AttributeError that the per-call try/except below would swallow.
for _attr in ("content", "tool_calls"):
    _require(
        _attr in DeltaMessage.model_fields,
        f"DeltaMessage no longer exposes a {_attr!r} field. The "
        f"wrapper's silent-failure heuristic reads this attribute and "
        f"would mis-observe without it.",
    )


# --------------------------------------------------------------------
# Phase 4: Landmark the leaking emit shape inside the method body.
# --------------------------------------------------------------------

try:
    _original_src = inspect.getsource(_original)  # type: ignore[arg-type]
except (OSError, TypeError) as _exc:
    raise StreamingMetricsPatchRefusedError(
        f"[{_PATCH_TAG}] cannot read source of "
        f"extract_tool_calls_streaming (is vLLM installed without "
        f"accompanying .py files?): {_exc!r}"
    ) from _exc

_require(
    _STREAMING_EMIT_LANDMARK in _original_src,
    f"streaming emit-shape landmark {_STREAMING_EMIT_LANDMARK!r} not "
    f"found in extract_tool_calls_streaming source. The return shape "
    f"this wrapper observes (a DeltaMessage whose content field "
    f"carries text directly emitted from the delta) appears to have "
    f"changed; refusing to wrap a function whose contract has drifted.",
)


# --------------------------------------------------------------------
# Phase 5: Optional Prometheus counter. Shared with the non-streaming
# patch; degrades cleanly if absent; reuses an existing registration on
# collision.
# --------------------------------------------------------------------

_COUNTER_NAME: str = "vllm_qwen3_coder_silent_tool_call_failures_total"
_COUNTER_DESCRIPTION: str = (
    "Responses where the qwen3_coder tool parser returned no tool "
    "calls but the model output contained <tool_call>/<function=/"
    "<parameter= markup (silent markup-leak failure, README §6.13). "
    "Labelled by failure_kind=markup_leak (non-streaming) or "
    "failure_kind=markup_leak_streaming (per-delta)."
)
_COUNTER_LABELS: tuple[str, ...] = ("failure_kind", "model")

# Prometheus exports the counter under ``<name>`` with a ``_total``
# suffix automatically appended when its metric type is Counter. So a
# ``Counter("..._failures_total", ...)`` registration stores collectors
# under the key ``..._failures`` (no ``_total``) in the default
# registry's ``_names_to_collectors`` mapping. We probe both forms on
# collision to be robust to the prometheus_client versioning quirk.
_COUNTER_LOOKUP_KEYS: tuple[str, ...] = (
    _COUNTER_NAME,
    _COUNTER_NAME.removesuffix("_total"),
)

_silent_failure_counter: Any | None
try:
    from prometheus_client import Counter as _PromCounter
    from prometheus_client import REGISTRY as _PromRegistry
except ImportError:
    _silent_failure_counter = None
    _prometheus_status: str = "unavailable"
    _logger.info(
        "[%s] prometheus_client not importable; the wrapper will "
        "install and emit structured log lines only. Counter %s "
        "will not be incremented by this patch.",
        _PATCH_TAG,
        _COUNTER_NAME,
    )
else:
    try:
        _silent_failure_counter = _PromCounter(
            _COUNTER_NAME,
            _COUNTER_DESCRIPTION,
            labelnames=_COUNTER_LABELS,
        )
        _prometheus_status = "active"
        _logger.info(
            "[%s] registered Prometheus counter %s "
            "(first registrant); labelnames=%r.",
            _PATCH_TAG,
            _COUNTER_NAME,
            _COUNTER_LABELS,
        )
    except ValueError as _exc:
        # Counter already registered — the expected case when the
        # non-streaming sibling patch
        # (monkey_patch_extract_tool_calls_metrics.py) was imported
        # first. Look up the existing collector by name so both patches
        # increment the same time series. The lookup is explicit and
        # load-bearing: if we cannot find the collector we registered
        # with, we must NOT silently fabricate a second ``Counter`` or
        # fall all the way back to logs-only, because both the operator
        # and the non-streaming patch's dashboards expect a live
        # counter with both ``failure_kind`` values observable.
        #
        # Lookup method: ``prometheus_client.REGISTRY._names_to_collectors``
        # is the documented (though underscore-prefixed) mapping of
        # metric-family names to collector instances used by the
        # default registry. The mapping is keyed by the metric name
        # WITHOUT the ``_total`` suffix that Counter appends. We probe
        # both forms (``_COUNTER_LOOKUP_KEYS``) to tolerate version
        # drift in prometheus_client's naming normalisation.
        _names_to_collectors = getattr(
            _PromRegistry, "_names_to_collectors", None
        )
        _found: Any | None = None
        if isinstance(_names_to_collectors, dict):
            for _key in _COUNTER_LOOKUP_KEYS:
                _candidate = _names_to_collectors.get(_key)
                if _candidate is not None:
                    _found = _candidate
                    break

        if _found is not None:
            _silent_failure_counter = _found
            _prometheus_status = "active_shared"
            _logger.info(
                "[%s] reused already-registered Prometheus counter %s "
                "discovered via REGISTRY._names_to_collectors after "
                "registration collided (%r); the non-streaming patch "
                "most likely registered it first.",
                _PATCH_TAG,
                _COUNTER_NAME,
                _exc,
            )
        else:
            _silent_failure_counter = None
            _prometheus_status = "unavailable"
            _logger.info(
                "[%s] Prometheus counter %s registration collided "
                "(%r) but the existing collector could not be "
                "recovered via REGISTRY._names_to_collectors "
                "(keys probed: %r). Falling back to structured-log-"
                "only observability for the streaming path.",
                _PATCH_TAG,
                _COUNTER_NAME,
                _exc,
                _COUNTER_LOOKUP_KEYS,
            )


# --------------------------------------------------------------------
# Phase 6: The wrapper.
# --------------------------------------------------------------------


def _contains_complete_marker(content: str) -> bool:
    """True iff ``content`` contains any COMPLETE markup marker.

    Partial-marker tolerance: a delta carrying ``"<too"`` or
    ``"<function"`` (no trailing ``=``) will return False. Python's
    ``str.__contains__`` is an exact substring test, which is exactly
    what we want here — any marker appearing in full is a leak; any
    prefix of a marker is not.

    Non-string input (e.g. ``None``) returns False so the wrapper
    never crashes on a ``DeltaMessage`` whose content field is left
    unset. Observability must not crash on unexpected input shapes.
    """
    if not isinstance(content, str) or not content:
        return False
    return any(marker in content for marker in _MARKUP_MARKERS)


def _count_markers(content: str) -> int:
    """Total occurrences of any complete markup marker in ``content``.

    Reported as ``marker_count`` on the structured log line so an
    operator can distinguish a single-marker leak from a catastrophic
    full-call-markup leak in a single delta without re-parsing the
    content.
    """
    return sum(content.count(marker) for marker in _MARKUP_MARKERS)


def _is_markup_leak_streaming(result: Any) -> bool:
    """Pure-function classification of the streaming silent-failure heuristic.

    Returns True iff:

    * ``result`` is a :class:`DeltaMessage` (not ``None``), AND
    * ``result.content`` is a non-empty ``str``, AND
    * ``result.tool_calls`` is empty or falsy (``[]`` is the default;
      a populated list means the parser correctly emitted tool
      calls), AND
    * ``result.content`` contains at least one COMPLETE
      ``<tool_call>`` / ``<function=`` / ``<parameter=`` marker.

    All four conditions must hold; any deviation is "not a leak".
    """
    if not isinstance(result, DeltaMessage):
        return False
    content = getattr(result, "content", None)
    if not isinstance(content, str) or not content:
        return False
    tool_calls = getattr(result, "tool_calls", None)
    if tool_calls:
        # Non-empty list, non-None, non-zero: the parser already
        # emitted structured tool-call data for this delta. Even if
        # content happens to contain leftover markup, it is not a
        # *silent* failure; structured calls are leaving the parser.
        return False
    return _contains_complete_marker(content)


def extract_tool_calls_streaming_observed(
    self: Any,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Any,
    current_token_ids: Any,
    delta_token_ids: Any,
    request: Any,
) -> Any:
    """Observation-only wrapper around ``Qwen3CoderToolParser.extract_tool_calls_streaming``.

    Behavior contract:

    * Calls the upstream ``extract_tool_calls_streaming`` exactly once
      with the arguments received. The call is **NOT** inside a
      try/except; any exception the parser raises propagates exactly
      as it would without this patch.
    * Captures the result (``DeltaMessage | None``).
    * If the streaming silent-failure heuristic (see
      :func:`_is_markup_leak_streaming`) matches, increments the
      shared Prometheus counter (when registered) with
      ``failure_kind="markup_leak_streaming"`` and emits a single
      structured WARNING log line.
    * Returns the captured result **unchanged** on every code path.

    The instrumentation block — Prometheus increment plus structured
    log — is wrapped in ``try / except Exception``. **This is the one
    place in the patch stack where catching ``Exception`` is correct**:
    observability MUST NOT take down request handling. A bug in the
    counter lookup, a logging formatter regression, an unexpected
    ``request`` shape (e.g. with no ``model``) — none of these may be
    allowed to convert a successful streamed chunk into a 500. The
    cost of this concession is local to instrumentation; the real
    parser exceptions are unaffected because the call to ``_original``
    is outside the guard.
    """
    result = _original(  # type: ignore[misc]
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    )

    try:
        if _is_markup_leak_streaming(result):
            # result is guaranteed a DeltaMessage with str content here.
            content: str = result.content  # type: ignore[assignment]
            content_len = len(content)
            marker_count = _count_markers(content)
            model_label = getattr(request, "model", None) or "unknown"

            if _silent_failure_counter is not None:
                _silent_failure_counter.labels(
                    failure_kind="markup_leak_streaming",
                    model=model_label,
                ).inc()

            # Stable, greppable, ELK-friendly format. Keys without
            # quotes; values numeric or simple strings; one line.
            # Intentionally identical layout to the non-streaming
            # patch's warning line, with only ``kind`` differing, so
            # log-based alerting rules work uniformly.
            _logger.warning(
                "silent_tool_call_failure kind=markup_leak_streaming "
                "content_len=%d marker_count=%d model=%s",
                content_len,
                marker_count,
                model_label,
            )
    except Exception as _exc:  # noqa: BLE001 — see docstring
        # Last-resort safety net. We log at DEBUG (not WARNING) to
        # avoid log-storm amplification if the failure is in the
        # logger itself; the surrounding except will catch a logger
        # exception too.
        try:
            _logger.debug(
                "[%s] instrumentation suppressed exception: %r",
                _PATCH_TAG,
                _exc,
            )
        except Exception:  # noqa: BLE001
            pass

    return result


extract_tool_calls_streaming_observed.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
extract_tool_calls_streaming_observed.__wrapped_original__ = _original  # type: ignore[attr-defined]
extract_tool_calls_streaming_observed.__name__ = "extract_tool_calls_streaming"
extract_tool_calls_streaming_observed.__qualname__ = (
    f"{_ParserCls.__qualname__}.extract_tool_calls_streaming"
)


# --------------------------------------------------------------------
# Phase 7: Install and verify.
# --------------------------------------------------------------------

_ParserCls.extract_tool_calls_streaming = extract_tool_calls_streaming_observed

_installed = _ParserCls.extract_tool_calls_streaming
_require(
    getattr(_installed, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install verification failed: "
    "Qwen3CoderToolParser.extract_tool_calls_streaming does not bear "
    "the expected patch tag. A concurrent monkey-patch has clobbered "
    "ours.",
)

# Second-order verification: the class itself must resolve the
# wrapper via normal attribute lookup (not just the instance __dict__),
# guarding against metaclass-level __getattribute__ overrides that
# could otherwise hide our assignment.
_resolved = inspect.getattr_static(_ParserCls, "extract_tool_calls_streaming")
_require(
    getattr(_resolved, "__qwen36_patch__", None) == _PATCH_TAG,
    "static-lookup verification failed: "
    "inspect.getattr_static sees a different "
    "extract_tool_calls_streaming than normal attribute access. "
    "Something in the MRO or metaclass is shadowing our assignment; "
    "refusing to proceed.",
)


_logger.info(
    "[%s] applied: wrapped %s.%s for vLLM commit %s "
    "(observation-only; prometheus=%s; counter=%s; "
    "failure_kind=markup_leak_streaming; fires per-delta on "
    "DeltaMessage with non-empty content containing a complete "
    "tool-call marker and empty tool_calls).",
    _PATCH_TAG,
    _ParserCls.__module__,
    _ParserCls.__qualname__,
    _PINNED_VLLM_COMMIT,
    _prometheus_status,
    _COUNTER_NAME,
)
