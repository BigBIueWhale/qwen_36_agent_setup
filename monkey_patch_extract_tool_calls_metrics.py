"""Server-side observability patch for vLLM ``qwen3_coder`` silent tool-parser failures.

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce`` (README §3.2).
Companion to (not replacement for) ``client/validate_response.py`` (patch 3).
Same fail-loud import-time discipline as ``monkey_patch_qwen3_coder.py`` (patch 4).

What this fixes
---------------

README §6.13 catalogs ~14 silent-degradation paths in
``Qwen3CoderToolParser`` (8 non-streaming, 6 streaming) where the parser
returns ``tools_called=False, tool_calls=[], content=<raw_xml_markup>``
under HTTP 200 with no error surfaced. Today the only place those
events are detected is **client-side**, in
``client/validate_response.py``, which raises
:class:`~client.validate_response.MarkupLeakError` after the response
has already crossed the wire. vLLM itself emits zero metrics, zero
logs, and zero counters for parser failure. An operator looking at
vLLM's Prometheus endpoint has no way to see how often the model
produced tool-call markup the parser silently dropped.

This patch closes that gap **server-side**. It wraps
``Qwen3CoderToolParser.extract_tool_calls`` so that each response with
the silent-failure shape (``tools_called=False`` *and* the input
``model_output`` contains tool-call markup) increments a Prometheus
counter and emits a single structured log line. The wrapped method's
return value is forwarded **unchanged**.

Scope: ``markup_leak`` only
---------------------------

The §6.13 catalog spans three failure classes:

* ``markup_leak`` — empty ``tool_calls`` plus leaked
  ``<tool_call>`` / ``<function=`` / ``<parameter=`` markup in
  ``content``. **Detected here.** This is the most important class
  because it is the one where the model intended to call a tool and
  the call was silently lost. It is also the only class detectable
  from ``extract_tool_calls``'s inputs and outputs alone.
* ``truncated_tool_call`` — non-empty ``tool_calls`` plus
  ``finish_reason == "length"``. ``finish_reason`` is added to the
  response by code downstream of ``extract_tool_calls``; it is not
  visible from inside the parser. **Out of scope here.** Patch 3
  catches it client-side.
* ``json_invalid`` / object-shape / required-field issues — also
  added by code downstream of this method (the parser produces a
  ``ToolCall`` whose ``arguments`` is a JSON string; structural
  validation of the parsed dict happens later). **Out of scope here.**
  Patch 3 catches it client-side.

The streaming path (``extract_tool_calls_streaming``) has a different
return shape (``DeltaMessage`` rather than ``ExtractedToolCallInformation``)
and the silent-failure shape across multiple deltas is not a single-
delta property; correctly observing it requires per-stream state. That
is **deliberately out of scope** for this patch and not attempted here.

Patch-discipline contract
-------------------------

This file is a patch, not a library. At import it:

1. Imports vLLM. Failure is a hard ImportError; we do not hide it.
2. Looks up ``Qwen3CoderToolParser`` in the expected module.
3. Verifies ``extract_tool_calls`` exists, is callable, and has
   exactly the signature ``(self, model_output, request)``.
4. Verifies ``ExtractedToolCallInformation`` is importable at the
   expected path and exposes the attributes
   (``tools_called``, ``tool_calls``, ``content``) the wrapper reads.
5. Reads the method's source and verifies the silent-failure-shape
   landmark — ``return ExtractedToolCallInformation(`` followed by
   ``tools_called=False`` and ``content=model_output`` — is present.
   Confirms our model of the function and refuses to wrap a
   contract-changed function blindly.
6. Imports ``prometheus_client.Counter`` and registers the counter.
   Both are HARD requirements — refuse loudly on either:

   * ``ImportError`` from ``prometheus_client`` → refuse. The patch's
     entire purpose is to surface silent tool-parser failures via
     Prometheus; a logs-only fallback would silently degrade exactly
     the visibility this patch was written to provide. ``vLLM``'s
     ``requirements/common.txt`` pins ``prometheus_client >= 0.18.0``,
     so missing this dependency means the deployment image is
     misconfigured — operator must know.
   * ``ValueError`` from ``Counter(name=…)`` (metric already
     registered) → refuse. Patch 6 is the FIRST registrant
     (``_PATCH_MODULES`` order in the launcher); a collision means
     patch 6 was imported twice OR another module registered our
     metric name OR the registry is in an inconsistent state. Each
     case is unexpected and demands operator attention.

7. Installs the wrapper, tags it with ``__qwen36_patch__``, and
   verifies both ``getattr`` and ``inspect.getattr_static`` resolve
   to the wrapped function bearing the tag.
8. Logs a single INFO line via ``vllm.logger.init_logger`` naming the
   class, the wrapped method, the Prometheus counter name, and the
   pinned commit.

Any of 1-7 failing raises :class:`MetricsPatchRefusedError` and the
interpreter does not continue. There is **no** ``SystemExit(0)``,
``try/except Exception: pass``, fallback to logs-only, or any other
silent-degradation path on any install path. The single exception is
the per-call instrumentation block inside the wrapper — documented at
its site — which catches ``Exception`` and logs at **WARNING** (not
DEBUG) so an instrumentation regression at request time is visible to
the operator without crashing the request.

Critical correctness invariants
-------------------------------

* **Observation only.** The wrapper calls the original method first,
  captures the result, and **returns the captured result unchanged**
  on every code path. It never alters ``tools_called``,
  ``tool_calls``, or ``content``. If this invariant is ever broken,
  the patch has failed its single purpose.
* **The original-method call is NOT inside the try/except.** Real
  parser exceptions propagate normally — same behavior as if this
  patch were not loaded.
* **No client-side recovery here.** Detecting ``markup_leak`` server-
  side cannot fix the lost tool call; we cannot retry the LLM
  mid-request from the parser. This patch and patch 3 cover the same
  symptom from opposite sides.
* **Per-response counting.** The metric counts *responses* exhibiting
  the silent-failure pattern, not *number of dropped tool calls* —
  the latter cannot be inferred from a ``tool_calls=[]`` result
  without re-parsing the markup, which is patch 3's job, not this
  patch's.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypeAlias


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-extract-tool-calls-metrics-v1"

# Markers whose presence in ``model_output`` constitutes a markup
# leak when paired with an empty ``tool_calls`` result. Mirrors the
# heuristic in ``client/validate_response.py``.
_MARKUP_MARKERS: tuple[str, ...] = ("<tool_call>", "<function=", "<parameter=")

# Source landmarks proving the silent-failure shape is the one we
# expect. We require all three substrings to appear in the source of
# ``extract_tool_calls`` so a refactor that changes the shape (e.g.
# switching to a different sentinel object, or returning ``None``
# instead of an empty list) trips the import-time refusal rather than
# silently mis-observing.
_SILENT_FAILURE_LANDMARKS: tuple[str, ...] = (
    "ExtractedToolCallInformation(",
    "tools_called=False",
    "content=model_output",
)


# Type alias for the method we wrap. Kept at module level so callers
# inspecting the installed attribute can tell by signature that it
# still conforms.
ExtractToolCalls: TypeAlias = Callable[[Any, str, Any], Any]


class MetricsPatchRefusedError(RuntimeError):
    """A precondition for the extract_tool_calls metrics wrapper was violated.

    Raised at import time only. The patch either applies cleanly and
    correctly, or the process does not come up. A half-installed or
    silently-skipped observability patch is the worst of both worlds:
    the operator believes they have visibility they do not actually
    have. We refuse to enter that state.
    """


def _require(condition: object, failure_message: str) -> None:
    if not condition:
        raise MetricsPatchRefusedError(
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
    ExtractedToolCallInformation,
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

_original: ExtractToolCalls | None = getattr(
    _ParserCls, "extract_tool_calls", None
)
_require(
    _original is not None and callable(_original),
    "Qwen3CoderToolParser.extract_tool_calls is missing or not callable.",
)

try:
    _sig = inspect.signature(_original)  # type: ignore[arg-type]
except (TypeError, ValueError) as _exc:
    raise MetricsPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect signature of "
        f"extract_tool_calls: {_exc!r}"
    ) from _exc

_param_names = list(_sig.parameters)
_require(
    _param_names == ["self", "model_output", "request"],
    f"extract_tool_calls signature changed; expected "
    f"(self, model_output, request), got {_param_names!r}.",
)


# --------------------------------------------------------------------
# Phase 3: Landmark the result type we read from.
# --------------------------------------------------------------------

# We don't construct ExtractedToolCallInformation; we only read three
# attributes off the upstream-built instance. But verifying the type
# carries those attribute names guards against an upstream rename
# turning every wrapper invocation into a silent AttributeError that
# the per-call try/except below would swallow.
for _attr in ("tools_called", "tool_calls", "content"):
    _require(
        _attr in ExtractedToolCallInformation.model_fields,
        f"ExtractedToolCallInformation no longer exposes a "
        f"{_attr!r} field. The wrapper's silent-failure heuristic "
        f"reads this attribute and would mis-observe without it.",
    )


# --------------------------------------------------------------------
# Phase 4: Landmark the silent-failure shape inside the method body.
# --------------------------------------------------------------------

try:
    _original_src = inspect.getsource(_original)  # type: ignore[arg-type]
except (OSError, TypeError) as _exc:
    raise MetricsPatchRefusedError(
        f"[{_PATCH_TAG}] cannot read source of extract_tool_calls "
        f"(is vLLM installed without accompanying .py files?): {_exc!r}"
    ) from _exc

for _landmark in _SILENT_FAILURE_LANDMARKS:
    _require(
        _landmark in _original_src,
        f"silent-failure-shape landmark {_landmark!r} not found in "
        f"extract_tool_calls source. The return shape this wrapper "
        f"observes (tools_called=False / tool_calls=[] / "
        f"content=model_output) appears to have changed; refusing to "
        f"wrap a function whose contract has drifted.",
    )


# --------------------------------------------------------------------
# Phase 5: Prometheus counter — REQUIRED. No silent fallback.
# --------------------------------------------------------------------
#
# The patch's entire reason for existing is server-side observability
# of qwen3_coder silent tool-parser failures. A logs-only degradation
# would silently undercut that purpose for an operator who scrapes the
# Prometheus endpoint. Both an ImportError on prometheus_client and a
# ValueError on registration (collision) are refused loudly.

_COUNTER_NAME: str = "vllm_qwen3_coder_silent_tool_call_failures_total"
_COUNTER_DESCRIPTION: str = (
    "Responses where the qwen3_coder tool parser returned no tool "
    "calls but the model output contained <tool_call>/<function=/"
    "<parameter= markup (silent markup-leak failure, README §6.13)."
)
_COUNTER_LABELS: tuple[str, ...] = ("failure_kind", "model")

try:
    from prometheus_client import Counter as _PromCounter
except ImportError as _exc:
    raise MetricsPatchRefusedError(
        f"[{_PATCH_TAG}] prometheus_client is not importable "
        f"({_exc!r}). vLLM's requirements/common.txt pins "
        f"prometheus_client >= 0.18.0, so this should not happen in a "
        f"vllm/vllm-openai container. The patch's entire purpose is to "
        f"expose silent tool-parser failures as a Prometheus counter; "
        f"there is no acceptable logs-only degradation. Refusing."
    ) from _exc

try:
    _silent_failure_counter: Any = _PromCounter(
        _COUNTER_NAME,
        _COUNTER_DESCRIPTION,
        labelnames=_COUNTER_LABELS,
    )
except ValueError as _exc:
    # Counter name already registered. Patch 6 is the FIRST registrant
    # in launch_with_patches.py:_PATCH_MODULES order; a collision here
    # means either (a) this module was imported twice (deployment
    # misconfiguration), (b) another piece of code registered our exact
    # metric name (collision with a future vLLM release that adds the
    # same metric — see README §12 trigger 9), or (c) the registry is
    # in an inconsistent state. None of these are acceptable silent
    # degradations.
    raise MetricsPatchRefusedError(
        f"[{_PATCH_TAG}] could not register Prometheus counter "
        f"{_COUNTER_NAME!r} ({_exc!r}). This patch is the first "
        f"registrant of this metric name; a collision means the patch "
        f"was imported twice, another component registered the same "
        f"name, or the registry is corrupt. Refusing rather than "
        f"falling back to logs-only — the operator must investigate."
    ) from _exc

_logger.info(
    "[%s] registered Prometheus counter %s (first registrant); "
    "labelnames=%r.",
    _PATCH_TAG,
    _COUNTER_NAME,
    _COUNTER_LABELS,
)


# --------------------------------------------------------------------
# Phase 6: The wrapper.
# --------------------------------------------------------------------


def _is_markup_leak(model_output: Any, result: Any) -> bool:
    """Pure-function classification of the silent-failure heuristic.

    Returns True iff:

    * ``result.tools_called`` is exactly ``False`` (not just falsy —
      a future change to ``None`` or ``0`` should not be silently
      reclassified as a leak), AND
    * ``model_output`` is a ``str`` containing at least one of
      ``<tool_call>`` / ``<function=`` / ``<parameter=``.

    Any malformed input (non-bool ``tools_called``, non-str
    ``model_output``) is treated as "not a leak" — observability must
    not crash on unexpected input shapes.
    """
    if not isinstance(model_output, str):
        return False
    if getattr(result, "tools_called", None) is not False:
        return False
    return any(marker in model_output for marker in _MARKUP_MARKERS)


def _count_markers(model_output: str) -> int:
    """Total occurrences of any markup marker in ``model_output``."""
    return sum(model_output.count(marker) for marker in _MARKUP_MARKERS)


def extract_tool_calls_observed(
    self: Any, model_output: str, request: Any
) -> Any:
    """Observation-only wrapper around ``Qwen3CoderToolParser.extract_tool_calls``.

    Behavior contract:

    * Calls the upstream ``extract_tool_calls`` exactly once with the
      arguments received. The call is **NOT** inside a try/except;
      any exception the parser raises propagates exactly as it would
      without this patch.
    * Captures the result.
    * If the silent-failure heuristic (see :func:`_is_markup_leak`)
      matches, increments the Prometheus counter (when registered)
      and emits a single structured WARNING log line.
    * Returns the captured result **unchanged** on every code path.

    The instrumentation block — Prometheus increment plus structured
    log — is wrapped in ``try / except Exception``. **This is the one
    place in the patch stack where catching ``Exception`` is correct**:
    observability MUST NOT take down request handling. A bug in the
    counter registration, a logging formatter regression, an
    unexpected ``request`` shape (e.g. with no ``model``) — none of
    these may be allowed to convert a successful response into a 500.
    The exception is logged at **WARNING** level (visible by default,
    not DEBUG) so an instrumentation regression at request time
    surfaces to the operator immediately. The cost of this concession
    is local to instrumentation; the real parser exceptions are
    unaffected because the call to ``_original`` is outside the guard.
    """
    result = _original(self, model_output, request)  # type: ignore[misc]

    try:
        if _is_markup_leak(model_output, result):
            content_len = len(model_output)
            marker_count = _count_markers(model_output)
            model_label = getattr(request, "model", None) or "unknown"

            _silent_failure_counter.labels(
                failure_kind="markup_leak",
                model=model_label,
            ).inc()

            # Stable, greppable, ELK-friendly format. Keys without
            # quotes; values numeric or simple strings; one line.
            _logger.warning(
                "silent_tool_call_failure kind=markup_leak "
                "content_len=%d marker_count=%d model=%s",
                content_len,
                marker_count,
                model_label,
            )
    except Exception as _exc:  # noqa: BLE001 — see docstring
        # Last-resort safety net. WARNING (not DEBUG) so the operator
        # sees instrumentation regressions immediately. The inner
        # try/except guards against the logger itself being broken;
        # if logging fails too there is nowhere to report from, and
        # we silently drop only that single exception report — never
        # the underlying request, which has already been served.
        try:
            _logger.warning(
                "[%s] instrumentation raised %s during markup-leak "
                "observation: %r",
                _PATCH_TAG,
                type(_exc).__name__,
                _exc,
            )
        except Exception:  # noqa: BLE001
            pass

    return result


extract_tool_calls_observed.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
extract_tool_calls_observed.__wrapped_original__ = _original  # type: ignore[attr-defined]
extract_tool_calls_observed.__name__ = "extract_tool_calls"
extract_tool_calls_observed.__qualname__ = (
    f"{_ParserCls.__qualname__}.extract_tool_calls"
)


# --------------------------------------------------------------------
# Phase 7: Install and verify.
# --------------------------------------------------------------------

_ParserCls.extract_tool_calls = extract_tool_calls_observed

_installed = _ParserCls.extract_tool_calls
_require(
    getattr(_installed, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install verification failed: "
    "Qwen3CoderToolParser.extract_tool_calls does not bear the "
    "expected patch tag. A concurrent monkey-patch has clobbered ours.",
)

# Second-order verification: the class itself must resolve the
# wrapper via normal attribute lookup (not just the instance __dict__),
# guarding against metaclass-level __getattribute__ overrides that
# could otherwise hide our assignment.
_resolved = inspect.getattr_static(_ParserCls, "extract_tool_calls")
_require(
    getattr(_resolved, "__qwen36_patch__", None) == _PATCH_TAG,
    "static-lookup verification failed: "
    "inspect.getattr_static sees a different extract_tool_calls than "
    "normal attribute access. Something in the MRO or metaclass is "
    "shadowing our assignment; refusing to proceed.",
)


_logger.info(
    "[%s] applied: wrapped %s.%s for vLLM commit %s "
    "(observation-only; counter=%s; "
    "fires on tools_called=False with leaked tool-call markup).",
    _PATCH_TAG,
    _ParserCls.__module__,
    _ParserCls.__qualname__,
    _PINNED_VLLM_COMMIT,
    _COUNTER_NAME,
)
