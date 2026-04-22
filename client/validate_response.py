"""Strict observation, tolerant reporting.

A validator that catches the silent-failure classes of vLLM's
``qwen3_coder`` tool parser without exporting the consequences of LLM
misbehavior to the caller as a typed-exception gauntlet.

Why this file exists (philosophy)
---------------------------------

vLLM's ``qwen3_coder`` tool parser has roughly 14 code paths
(8 non-streaming, 6 streaming) that return HTTP 200 while degrading
the response in one of two ways (README §6.13):

* ``tool_calls`` is empty and raw ``<tool_call>`` / ``<function=`` /
  ``<parameter=`` markup leaks into ``content`` as plain text.
* ``tool_calls`` is populated but ``function.arguments`` is malformed
  JSON, a non-object value, missing required fields, or refers to a
  hallucinated tool name.

Neither failure is the caller's fault. The caller wrote a valid
request against a valid tool schema. The model misbehaved. Designing
a validator that raises seven typed exceptions the caller must catch,
dispatch on, and write recovery code for is — bluntly — punishing
the caller for the model's behavior. The system's job is to absorb
this gracefully and hand back a result the caller can act on without
writing defensive boilerplate around every call site.

Design principle: strict where strictness belongs, tolerant where it
does not.

* **STRICT (always raises)**

  - ``MalformedResponseError`` for response-shape violations
    (missing ``choices``, wrong types on sub-fields, etc.). These
    are system-level wire-contract violations — either the caller
    passed the wrong object or the server regressed. There is no
    recovery a caller could meaningfully take, and "absorbing" the
    failure would just hide a real bug.
  - ``TypeError`` for caller mis-use (``tool_schemas_by_name`` is
    not a Mapping). The caller's actual mistake; raise.

* **TOLERANT (returns structured data)**

  - Every class of LLM misbehavior is recorded as a
    :class:`ToolCallIssue` on the returned :class:`ValidationResult`.
    The result also exposes ``tool_calls`` — only the calls that
    passed every check, dispatch-ready with parsed ``arguments``.
  - Callers that prefer raise-on-misbehavior semantics opt in by
    invoking ``result.raise_on_model_misbehavior()`` immediately
    after validation. The seven typed exception subclasses still
    exist for that path; they are not the default.

Truncation handling, specifically
---------------------------------

When ``finish_reason == "length"`` and ``tool_calls`` is non-empty,
the runtime patch ``monkey_patch_qwen3_coder.py`` has already dropped
any tool call whose parameter list was incomplete (returning ``None``
for those, which vLLM's filter at ``extract_tool_calls:313`` removes).
The tool calls that *do* survive into the response on a length-finish
are therefore complete and safe to dispatch. The validator marks the
response with a ``truncated_tool_call`` issue (recovery hint
``bump_max_tokens``) so the caller knows the model wanted to do more,
but it does NOT preemptively reject the surviving calls. Strict
callers can still refuse via ``raise_on_model_misbehavior``.

What never silently passes
--------------------------

A tool call is added to ``ValidationResult.tool_calls`` only after it
has cleared every individual check (JSON parse, top-level object
shape, no duplicate keys, known tool name, all required fields
present). Any failure routes the call to ``issues`` and excludes it
from ``tool_calls``. The caller cannot accidentally dispatch a
tool call that flunked validation by reading ``tool_calls`` naively.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    runtime_checkable,
)


# ---------------------------------------------------------------------------
# Landmark regex
# ---------------------------------------------------------------------------


_LEAK_RE: re.Pattern[str] = re.compile(
    r"<tool_call>|<function=|<parameter=",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Type aliases — the closed enums for issue kinds and recovery hints
# ---------------------------------------------------------------------------


IssueKind = Literal[
    "markup_leak",
    "truncated_tool_call",
    "invalid_tool_arguments_json",
    "tool_arguments_not_object",
    "unknown_tool",
    "missing_required_tool_argument",
    "duplicate_parameter",
]

RecoveryHint = Literal[
    # Re-run with a different sampler state. Right for OOD emissions
    # (markup leak, duplicate parameter) where the model just had a
    # bad sample and a fresh roll usually clears it.
    "retry_with_fresh_seed",
    # Re-run with a larger max_tokens. Right for finish_reason=length
    # cases where the model needed more budget than was allocated.
    "bump_max_tokens",
    # Re-prompt with a short correction ("your last tool call had
    # malformed arguments — re-emit"). Right for argument-shape
    # failures where the model knows the tool but emitted wrong JSON.
    "reprompt",
    # Refuse this tool call permanently. Right for hallucinated tool
    # names — re-prompting won't summon a tool that does not exist.
    "reject",
]


# ---------------------------------------------------------------------------
# ToolCallRef — a validated, dispatch-ready tool call
# ---------------------------------------------------------------------------


class ToolCallRef(NamedTuple):
    """A validated tool call.

    ``arguments`` is the parsed ``dict``, not the JSON-encoded
    string the SDK exposes. Callers who re-serialize for logging do
    so themselves; the common case is ``fn(**call.arguments)`` and
    re-parsing at every dispatch site is wasted work.
    """

    id: str
    name: str
    arguments: dict[str, Any]


# ---------------------------------------------------------------------------
# ToolCallIssue — diagnostic for one LLM-misbehavior event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallIssue:
    """One LLM-misbehavior event detected during validation.

    Fields:

    * ``kind``: the closed-enum classification.
    * ``detail``: a free-form string. Machine-readable code lives in
      ``kind``; ``detail`` is for logs and human inspection.
    * ``suggested_recovery``: the recovery action the agent loop
      should take. Pre-baked into the issue so the caller does not
      need a separate dispatch table.
    * ``tool_call_index``: position of the offending call in the
      original response, or ``None`` for response-level issues
      (``markup_leak``, ``truncated_tool_call``).
    * ``tool_name``: the function name from the offending call, when
      one exists. ``None`` for response-level issues.
    * ``raw_tool_call``: the dumped dict of the offending call, for
      diagnostic logging. ``None`` for response-level issues.
    """

    kind: IssueKind
    detail: str
    suggested_recovery: RecoveryHint
    tool_call_index: int | None = None
    tool_name: str | None = None
    raw_tool_call: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# ValidationResult — the tolerant return surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """The structured return of :func:`validate_chat_response`.

    Fields:

    * ``content``: the ``message.content`` string. Empty string when
      the response had no content (rather than ``None``, so simple
      string concatenation by the caller does not fail).
    * ``tool_calls``: the validated, dispatch-ready tool calls. Each
      one cleared every check; ``arguments`` is parsed to ``dict``.
      Read this naively to dispatch.
    * ``issues``: per-event diagnostics for everything that did NOT
      clear validation. Empty when the response was clean.
    * ``finish_reason``: as reported by the server.
    * ``raw_tool_calls``: the dumped form of every tool call from
      the original response, in order — including the rejected
      ones. For diagnostic logging; normal callers read
      ``tool_calls``.

    Call patterns:

    .. code-block:: python

        # tolerant default
        result = validate_chat_response(resp, schemas)
        for call in result.tool_calls:
            dispatch(call.name, **call.arguments)
        if result.issues:
            log.warning("model misbehavior: %s", result.issues)

        # opt-in strict
        result = validate_chat_response(resp, schemas)
        result.raise_on_model_misbehavior()
        for call in result.tool_calls:
            dispatch(call.name, **call.arguments)

        # selective dispatch on issue kind
        for issue in result.issues_by_kind("unknown_tool"):
            log.error("hallucinated tool: %s", issue.tool_name)
    """

    content: str
    tool_calls: list[ToolCallRef]
    issues: list[ToolCallIssue]
    finish_reason: str | None
    raw_tool_calls: list[dict[str, Any]]

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    @property
    def has_valid_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def has_usable_output(self) -> bool:
        """True when the response yielded *something* the caller can
        act on — at least one validated tool call, or non-empty
        content. False only in the degenerate case where every tool
        call flunked AND content is empty."""
        return bool(self.content) or self.has_valid_tool_calls

    def issues_by_kind(self, kind: IssueKind) -> list[ToolCallIssue]:
        """Filter ``issues`` by kind, for callers dispatching on
        specific misbehavior classes without writing the comprehension
        themselves."""
        return [i for i in self.issues if i.kind == kind]

    def raise_on_model_misbehavior(self) -> None:
        """Opt-in strict mode.

        If any issues were recorded, raise the first one as its
        typed :class:`ToolCallParseError` subclass. No-op when
        ``issues`` is empty.

        Agent loops that prefer dispatch-on-exception write::

            result = validate_chat_response(resp, schemas)
            result.raise_on_model_misbehavior()

        and proceed as if the call had raised directly.
        """
        if not self.issues:
            return
        issue = self.issues[0]
        cls = _ISSUE_KIND_TO_EXCEPTION[issue.kind]
        raise cls(issue)


# ---------------------------------------------------------------------------
# Exceptions — strict (always raised) and opt-in strict
# ---------------------------------------------------------------------------


class MalformedResponseError(Exception):
    """STRICT — always raised by :func:`validate_chat_response` on
    response-shape violations.

    The response object did not match the OpenAI Chat Completion
    wire contract: ``choices`` was missing, wrong type, or empty;
    or ``message`` / ``finish_reason`` was malformed; or a tool
    call's ``function.name`` / ``function.arguments`` was the wrong
    type. This is a system-level contract violation — the caller
    passed the wrong object, the SDK changed shape, or the server
    regressed. There is no useful recovery a retry could effect.

    Deliberately does NOT inherit from :class:`ToolCallParseError`.
    Callers ``except``-ing the LLM-misbehavior root will not also
    catch this; the two are categorically different.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"[malformed_response] {detail}")
        self.detail: str = detail


class ToolCallParseError(Exception):
    """OPT-IN strict-mode root. Raised only via
    :meth:`ValidationResult.raise_on_model_misbehavior`.

    Wraps a single :class:`ToolCallIssue` so callers can read the
    structured details on the exception:

    .. code-block:: python

        try:
            result.raise_on_model_misbehavior()
        except ToolCallParseError as exc:
            recover(exc.issue.suggested_recovery)
    """

    kind: ClassVar[str] = "tool_call_parse_error"

    def __init__(self, issue: ToolCallIssue) -> None:
        super().__init__(f"[{issue.kind}] {issue.detail}")
        self.issue: ToolCallIssue = issue


class MarkupLeakError(ToolCallParseError):
    """``<tool_call>`` / ``<function=`` / ``<parameter=`` markup in
    ``content`` with empty ``tool_calls``. Recovery:
    ``retry_with_fresh_seed``."""

    kind: ClassVar[str] = "markup_leak"


class TruncatedToolCallError(ToolCallParseError):
    """``finish_reason == "length"`` with non-empty ``tool_calls``.
    The model wanted to emit more tokens than the budget allowed.
    Recovery: ``bump_max_tokens``.

    Note: the surviving ``tool_calls`` in the result are individually
    complete and safe to dispatch (the runtime patch dropped any
    truncated-mid-param call). This issue is informational; the
    caller can choose to dispatch the surviving calls and re-prompt
    for the rest, or to reject the whole response."""

    kind: ClassVar[str] = "truncated_tool_call"


class InvalidToolArgumentsJsonError(ToolCallParseError):
    """``function.arguments`` did not parse as JSON, or arrived in
    an unsupported type. Recovery: ``reprompt``."""

    kind: ClassVar[str] = "invalid_tool_arguments_json"


class ToolArgumentsNotObjectError(ToolCallParseError):
    """``function.arguments`` parsed but the top-level value was not
    a JSON object. Arrays, scalars, and ``null`` violate the OpenAI
    contract and break ``fn(**arguments)``. Recovery: ``reprompt``."""

    kind: ClassVar[str] = "tool_arguments_not_object"


class UnknownToolError(ToolCallParseError):
    """``function.name`` was not in the caller-supplied
    ``tool_schemas_by_name``. The model has hallucinated a tool.
    Recovery: ``reject`` (re-prompting will not summon a tool that
    does not exist)."""

    kind: ClassVar[str] = "unknown_tool"


class MissingRequiredToolArgumentError(ToolCallParseError):
    """One or more fields the schema marks ``required`` were absent
    from the parsed ``arguments``. Recovery: ``reprompt``."""

    kind: ClassVar[str] = "missing_required_tool_argument"


class DuplicateParameterError(ToolCallParseError):
    """A duplicated key was found at any nesting level inside the
    parsed ``arguments``. The model re-emitted a parameter mid-call
    and the caller cannot tell which value was intended. Recovery:
    ``reprompt``."""

    kind: ClassVar[str] = "duplicate_parameter"


_ISSUE_KIND_TO_EXCEPTION: dict[str, type[ToolCallParseError]] = {
    "markup_leak": MarkupLeakError,
    "truncated_tool_call": TruncatedToolCallError,
    "invalid_tool_arguments_json": InvalidToolArgumentsJsonError,
    "tool_arguments_not_object": ToolArgumentsNotObjectError,
    "unknown_tool": UnknownToolError,
    "missing_required_tool_argument": MissingRequiredToolArgumentError,
    "duplicate_parameter": DuplicateParameterError,
}


# ---------------------------------------------------------------------------
# Shape protocols — landmarks for duck-typed inputs
# ---------------------------------------------------------------------------


@runtime_checkable
class _ChatCompletionLike(Protocol):
    """Minimal structural protocol: anything with ``choices``."""

    choices: Any


@runtime_checkable
class _ChoiceLike(Protocol):
    """Minimal structural protocol for a Choice: carries
    ``message`` and ``finish_reason``."""

    message: Any
    finish_reason: Any


# ---------------------------------------------------------------------------
# Duck-typing helpers
# ---------------------------------------------------------------------------


_SENTINEL: Any = object()


def _probe(obj: Any, attr: str) -> Any:
    """Read ``obj.attr`` or ``obj[attr]``. Returns ``_SENTINEL`` if
    truly absent — distinguishes "absent" from "explicit ``None``"."""
    if isinstance(obj, Mapping):
        if attr in obj:
            return obj[attr]
        return _SENTINEL
    return getattr(obj, attr, _SENTINEL)


def _dump_tool_call(tc: Any) -> dict[str, Any]:
    """Serialize an SDK ``ToolCall``-like object to a plain dict for
    diagnostic logging. Prefer pydantic's ``model_dump``; fall back
    to a hand-rolled shim that reads the three fields we care about.

    ``model_dump`` failure is silently fallen-through-from because
    the goal of this helper is *exception-payload assembly*, not
    correctness — we never want diagnostic-payload construction to
    raise its own exception while building the payload for an actual
    issue.
    """
    model_dump = getattr(tc, "model_dump", None)
    if callable(model_dump):
        try:
            return dict(model_dump())
        except Exception:  # noqa: BLE001 — payload-only path
            pass
    fn = _probe(tc, "function")
    fn_name = _probe(fn, "name") if fn is not _SENTINEL else None
    fn_args = _probe(fn, "arguments") if fn is not _SENTINEL else None
    tc_id = _probe(tc, "id")
    tc_type = _probe(tc, "type")
    return {
        "id": tc_id if tc_id is not _SENTINEL else None,
        "type": tc_type if tc_type is not _SENTINEL else "function",
        "function": {
            "name": fn_name if fn_name is not _SENTINEL else None,
            "arguments": fn_args if fn_args is not _SENTINEL else None,
        },
    }


def _normalize_schema(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either OpenAI envelope ``{"parameters": {...}}`` or a
    bare JSON-Schema object."""
    if "parameters" in schema and isinstance(schema["parameters"], Mapping):
        return schema["parameters"]
    return schema


def _required_fields(schema: Mapping[str, Any]) -> list[str]:
    inner = _normalize_schema(schema)
    required = inner.get("required", []) if isinstance(inner, Mapping) else []
    if not isinstance(required, list):
        return []
    return [r for r in required if isinstance(r, str)]


# ---------------------------------------------------------------------------
# Strict landmark gauntlet — raises MalformedResponseError on shape failure
# ---------------------------------------------------------------------------


def _landmark_response(response: Any) -> tuple[Any, str | None]:
    """Walk ``response.choices[0]`` and return
    ``(message, finish_reason)``. Raises
    :class:`MalformedResponseError` on any shape violation, naming
    the exact landmark that failed."""
    choices = _probe(response, "choices")
    if choices is _SENTINEL:
        raise MalformedResponseError(
            f"response of type {type(response).__name__!r} has no "
            f"'choices' attribute or key; not an OpenAI Chat "
            f"Completion shape"
        )
    if not isinstance(choices, (list, tuple)):
        raise MalformedResponseError(
            f"response.choices is of type {type(choices).__name__!r}, "
            f"not a list/tuple"
        )
    if not choices:
        raise MalformedResponseError(
            "response.choices is empty (a moderation refusal or "
            "backend rejection — distinct from a tool-parser silent "
            "failure and not addressable by retry of the same request)"
        )

    choice = choices[0]
    message = _probe(choice, "message")
    if message is _SENTINEL or message is None:
        raise MalformedResponseError(
            f"choices[0] of type {type(choice).__name__!r} carries "
            f"no 'message'"
        )
    finish_reason_raw = _probe(choice, "finish_reason")
    if finish_reason_raw is _SENTINEL:
        raise MalformedResponseError(
            f"choices[0] of type {type(choice).__name__!r} carries "
            f"no 'finish_reason' field"
        )
    if finish_reason_raw is not None and not isinstance(finish_reason_raw, str):
        raise MalformedResponseError(
            f"choices[0].finish_reason is of type "
            f"{type(finish_reason_raw).__name__!r}, expected str or None"
        )
    return message, finish_reason_raw


def _landmark_message(message: Any) -> tuple[str, list[Any]]:
    """Validate the message shape and return
    ``(content_str, tool_calls_list)``. Raises
    :class:`MalformedResponseError`."""
    content_raw = _probe(message, "content")
    if content_raw is _SENTINEL or content_raw is None:
        content: str = ""
    elif isinstance(content_raw, str):
        content = content_raw
    else:
        raise MalformedResponseError(
            f"message.content is of type "
            f"{type(content_raw).__name__!r}, expected str or None"
        )

    tool_calls_raw = _probe(message, "tool_calls")
    if tool_calls_raw is _SENTINEL or tool_calls_raw is None:
        tool_calls_list: list[Any] = []
    elif isinstance(tool_calls_raw, (list, tuple)):
        tool_calls_list = list(tool_calls_raw)
    else:
        raise MalformedResponseError(
            f"message.tool_calls is of type "
            f"{type(tool_calls_raw).__name__!r}, expected list/tuple/None"
        )
    return content, tool_calls_list


def _landmark_tool_call(tc: Any, idx: int) -> tuple[str, str, Any]:
    """Validate a single tool-call object's shape and return
    ``(id, name, arguments_raw)``. Raises
    :class:`MalformedResponseError`."""
    fn = _probe(tc, "function")
    if fn is _SENTINEL or fn is None:
        raise MalformedResponseError(
            f"tool_calls[{idx}] of type {type(tc).__name__!r} has no "
            f"'function' field"
        )
    name = _probe(fn, "name")
    if name is _SENTINEL or not isinstance(name, str) or not name:
        raise MalformedResponseError(
            f"tool_calls[{idx}].function.name is missing, empty, or "
            f"non-string (got {type(name).__name__!r})"
        )
    arguments = _probe(fn, "arguments")
    if arguments is _SENTINEL:
        raise MalformedResponseError(
            f"tool_calls[{idx}].function.arguments is missing"
        )
    call_id = _probe(tc, "id")
    if call_id is _SENTINEL or call_id is None:
        call_id = ""
    elif not isinstance(call_id, str):
        raise MalformedResponseError(
            f"tool_calls[{idx}].id is of type "
            f"{type(call_id).__name__!r}, expected str or None"
        )
    return call_id, name, arguments


# ---------------------------------------------------------------------------
# The main validator
# ---------------------------------------------------------------------------


def validate_chat_response(
    response: Any,
    tool_schemas_by_name: Mapping[str, Mapping[str, Any]],
) -> ValidationResult:
    """Validate a Chat Completion response.

    Strict on system contracts; tolerant on LLM behavior.

    Args:
        response: an OpenAI SDK ``ChatCompletion`` object or a
            duck-typed equivalent (pydantic model, plain dict,
            custom wrapper).
        tool_schemas_by_name: mapping from tool name to its JSON
            schema. Either ``{"parameters": {"type": "object", ...}}``
            (OpenAI envelope) or a bare JSON-Schema object is
            accepted. Only ``required`` is read; full JSON-Schema
            validation is out of scope.

    Returns:
        :class:`ValidationResult`. ``tool_calls`` lists the validated,
        dispatch-ready calls; ``issues`` lists every LLM-misbehavior
        event detected. The two together account for every tool call
        in the original response.

    Raises:
        MalformedResponseError: response shape violated the OpenAI
            Chat Completion wire contract. Always raised on shape
            failure — never converted to an issue.
        TypeError: ``tool_schemas_by_name`` is not a Mapping.
            Caller-side type bug; raised rather than absorbed because
            the validator cannot meaningfully proceed.
    """
    if not isinstance(tool_schemas_by_name, Mapping):
        raise TypeError(
            f"tool_schemas_by_name must be a Mapping, got "
            f"{type(tool_schemas_by_name).__name__!r}"
        )

    message, finish_reason = _landmark_response(response)
    content, tool_calls_raw = _landmark_message(message)
    dumped_tool_calls: list[dict[str, Any]] = [
        _dump_tool_call(tc) for tc in tool_calls_raw
    ]
    issues: list[ToolCallIssue] = []

    # ---- Response-level issue: markup leaked into content ----
    # The §6.13 case, also produced intentionally by the runtime
    # patch when it drops a truncated tool call. Caller recovery:
    # retry with a fresh seed (model OOD), or run rescue
    # (client/rescue_tool_calls.py) against `content`.
    if not tool_calls_raw and _LEAK_RE.search(content):
        issues.append(
            ToolCallIssue(
                kind="markup_leak",
                detail="tool-call markup found in content with empty "
                "tool_calls",
                suggested_recovery="retry_with_fresh_seed",
            )
        )

    # ---- Response-level issue: truncation ----
    # The runtime patch already dropped any truncated-mid-param call.
    # Surviving calls in tool_calls_raw are individually complete and
    # safe to dispatch; we record the truncation as informational so
    # the caller knows the model wanted to do more, but we DO NOT
    # preemptively reject the surviving calls. Strict callers can
    # still refuse via raise_on_model_misbehavior().
    if finish_reason == "length" and tool_calls_raw:
        issues.append(
            ToolCallIssue(
                kind="truncated_tool_call",
                detail="finish_reason=length with non-empty tool_calls; "
                "the model wanted to emit more tokens than max_tokens "
                "allowed. Surviving tool calls are individually complete "
                "and safe to dispatch; the caller may dispatch them and "
                "re-prompt for the rest, or reject the whole response",
                suggested_recovery="bump_max_tokens",
            )
        )

    validated: list[ToolCallRef] = []

    for idx, (tc, tc_dump) in enumerate(
        zip(tool_calls_raw, dumped_tool_calls)
    ):
        # Shape gauntlet for this call. Shape failures stay strict
        # (raise) — they are wire-contract violations, not LLM
        # misbehavior.
        call_id, name, arguments_raw = _landmark_tool_call(tc, idx)

        # --- JSON parse ---
        if isinstance(arguments_raw, str):
            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError as exc:
                issues.append(
                    ToolCallIssue(
                        kind="invalid_tool_arguments_json",
                        detail=f"tool {name!r} arguments is not valid "
                        f"JSON: {exc.msg} at char {exc.pos}",
                        suggested_recovery="reprompt",
                        tool_call_index=idx,
                        tool_name=name,
                        raw_tool_call=tc_dump,
                    )
                )
                continue
        elif isinstance(arguments_raw, Mapping):
            # Some custom wrappers (and the rescue path) hand us a
            # dict directly. Accept; treat as already-parsed.
            arguments = dict(arguments_raw)
        else:
            issues.append(
                ToolCallIssue(
                    kind="invalid_tool_arguments_json",
                    detail=f"tool {name!r} arguments has unsupported "
                    f"type {type(arguments_raw).__name__!r}; expected "
                    f"a JSON-encoded string or a mapping",
                    suggested_recovery="reprompt",
                    tool_call_index=idx,
                    tool_name=name,
                    raw_tool_call=tc_dump,
                )
            )
            continue

        # --- Top-level JSON object check ---
        if not isinstance(arguments, dict):
            issues.append(
                ToolCallIssue(
                    kind="tool_arguments_not_object",
                    detail=f"tool {name!r} arguments parsed to "
                    f"{type(arguments).__name__!r}; OpenAI spec requires "
                    f"a JSON object at the top level",
                    suggested_recovery="reprompt",
                    tool_call_index=idx,
                    tool_name=name,
                    raw_tool_call=tc_dump,
                )
            )
            continue

        # --- Duplicate-key probe ---
        # Only meaningful on the string path; if arguments arrived as
        # a Mapping the duplicates have already collapsed and
        # reconstruction is impossible.
        if isinstance(arguments_raw, str):
            try:
                json.loads(
                    arguments_raw,
                    object_pairs_hook=_reject_duplicate_keys,
                )
            except _DuplicateKey as exc:
                issues.append(
                    ToolCallIssue(
                        kind="duplicate_parameter",
                        detail=f"tool {name!r} arguments contains "
                        f"duplicated key {exc.key!r}; the model likely "
                        f"re-emitted a parameter mid-call and the caller "
                        f"cannot tell which value was intended",
                        suggested_recovery="reprompt",
                        tool_call_index=idx,
                        tool_name=name,
                        raw_tool_call=tc_dump,
                    )
                )
                continue

        # --- Unknown tool ---
        if name not in tool_schemas_by_name:
            issues.append(
                ToolCallIssue(
                    kind="unknown_tool",
                    detail=f"tool {name!r} is not in tool_schemas_by_name "
                    f"(known: {sorted(tool_schemas_by_name)!r})",
                    suggested_recovery="reject",
                    tool_call_index=idx,
                    tool_name=name,
                    raw_tool_call=tc_dump,
                )
            )
            continue

        # --- Required-field presence ---
        required = _required_fields(tool_schemas_by_name[name])
        missing = [r for r in required if r not in arguments]
        if missing:
            issues.append(
                ToolCallIssue(
                    kind="missing_required_tool_argument",
                    detail=f"tool {name!r} is missing required "
                    f"arguments {missing!r}",
                    suggested_recovery="reprompt",
                    tool_call_index=idx,
                    tool_name=name,
                    raw_tool_call=tc_dump,
                )
            )
            continue

        # Cleared every check.
        validated.append(
            ToolCallRef(id=call_id, name=name, arguments=arguments)
        )

    return ValidationResult(
        content=content,
        tool_calls=validated,
        issues=issues,
        finish_reason=finish_reason,
        raw_tool_calls=dumped_tool_calls,
    )


# ---------------------------------------------------------------------------
# Duplicate-key probe internals
# ---------------------------------------------------------------------------


class _DuplicateKey(Exception):
    """Internal sentinel raised by ``_reject_duplicate_keys`` and
    caught inside :func:`validate_chat_response` so it never escapes
    as an opaque ``json.JSONDecodeError`` substitute."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key: str = key


def _reject_duplicate_keys(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    """``object_pairs_hook`` for ``json.loads``: raises
    :class:`_DuplicateKey` on the first duplicated key at any
    nesting level."""
    seen: set[str] = set()
    pair_list = list(pairs)
    for k, _ in pair_list:
        if k in seen:
            raise _DuplicateKey(k)
        seen.add(k)
    return dict(pair_list)


__all__ = [
    # Closed enums
    "IssueKind",
    "RecoveryHint",
    # Structured returns
    "ToolCallRef",
    "ToolCallIssue",
    "ValidationResult",
    # Always-raised exception (system-contract)
    "MalformedResponseError",
    # Opt-in strict-mode exception hierarchy (LLM behavior)
    "ToolCallParseError",
    "MarkupLeakError",
    "TruncatedToolCallError",
    "InvalidToolArgumentsJsonError",
    "ToolArgumentsNotObjectError",
    "UnknownToolError",
    "MissingRequiredToolArgumentError",
    "DuplicateParameterError",
    # Main entry
    "validate_chat_response",
]
