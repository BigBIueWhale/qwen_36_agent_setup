"""Strict rescue for ``<tool_call>`` blocks emitted inside ``<think>``.

Why this file exists
--------------------

README §6.1 documents a Class B failure mode: Qwen3.6 occasionally
emits a ``<tool_call>...</tool_call>`` block *inside*
``<think>...</think>`` as an out-of-distribution emission. vLLM's
reasoning parser runs first, greedily captures the entire think
range into ``reasoning_content`` (wire name ``reasoning`` — README
§6.12), and hands the response to the tool parser with no
``<tool_call>`` markup left to find. The response lands with
``tool_calls=[]`` and the swallowed markup embedded in
``reasoning_content`` as plain text.

This is **not** a vLLM bug. Qwen3.6's training contract places
tool calls strictly after ``</think>``; Alibaba's own evaluator
(``Qwen-Agent/benchmark/deepplanning``) strips everything up to
``</think>`` before parsing. The parser is correct-to-contract,
the model is wrong. Do not file an upstream patch making vLLM
tolerant of mid-think tool calls — README §6.1 closes that door
explicitly. This rescue is a client-side workaround for a model
OOD emission, not a parser fix.

Patch-discipline contract
-------------------------

Every public function in this module:

1. Validates the structural shape of its input via landmarks
   (presence of ``<tool_call>``, integrity of the
   ``<function=NAME>`` terminator, ``<parameter=KEY>VALUE``
   structure) before mutation.
2. Applies the rescue only when the landmarks pass.
3. Strips rescued markup from the source field in a single pass
   that preserves pre-block and post-block text verbatim.
4. Emits ``id="call_rescued_<N>"`` on rescued tool calls so the
   sibling validator (``client/validate_response.py``) can
   distinguish rescued-from-reasoning calls from natively-parsed
   ones for telemetry.

Crucial drop-vs-salvage policy
------------------------------

If a ``<tool_call>`` block is structurally salvageable but one of
its parameters is malformed (no ``>`` after ``<parameter=NAME``),
the entire block is **dropped**, not partially salvaged. This
mirrors the runtime patch's stance in ``monkey_patch_qwen3_coder.py``:
a ToolCall with silently-omitted fields is a correctness hazard,
because the agent loop cannot distinguish "model chose these
arguments" from "model was cut off mid-emission". Detection is
delegated to ``client/validate_response.py`` which sees the leftover
markup in ``reasoning_content`` (or ``content``) and raises
:class:`~client.validate_response.MarkupLeakError`.

Pairing
-------

This rescue is paired with a system-prompt guardrail ("You MUST
close ``</think>`` before emitting any ``<tool_call>``.") that
reduces incidence from single-digit-percent to a rarer residual.
This file documents the pairing but does NOT inject the system
prompt — that lives in the agent's system-prompt template, not in
a response post-processor.

What this file refuses to do silently
-------------------------------------

The two write-back paths in :func:`rescue_tool_calls_from_response`
(dict path and pydantic-attribute path) **raise** :class:`
RescueWriteError` if a setattr or item-assignment does not stick.
A frozen pydantic model that silently discards writes returns the
caller to the silent-failure mode this rescue exists to eliminate;
that outcome is strictly worse than a loud exception at the
boundary.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Landmark constants — mirror the pinned vLLM parser's regex shape.
# Reference: /tmp/qwen36_research/vllm_pinned_ref/vllm/tool_parsers/
#            qwen3coder_tool_parser.py:60-69
# ---------------------------------------------------------------------------


_RESCUED_ID_PREFIX: str = "call_rescued_"


# Outer-block regex tolerates a missing ``</function>`` tag (Qwen
# models occasionally drop it). The ``name`` capture rejects
# embedded ``>`` and newlines so a truncated ``<function=NAME``
# (no ``>``) does not match — patch-discipline says the regex
# itself is the first landmark and a non-match is a clean skip.
_TOOL_CALL_BLOCK_RE: re.Pattern[str] = re.compile(
    r"<tool_call>\s*"
    r"<function=(?P<name>[^>\n]+?)>"
    r"(?P<params>.*?)"
    r"(?:</function>\s*</tool_call>|</tool_call>)",
    re.DOTALL,
)

# The vLLM upstream parameter regex is:
#     <parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)
# The non-greedy ``.*?`` stops at the FIRST ``</parameter>``, which
# truncates values that legitimately contain the literal substring
# ``</parameter>`` (rare but real, e.g. arguments containing code
# samples — README §5.4 calls this out as a vLLM parser hazard).
# Our rescue corrects the anchor: per-parameter chunks are sliced
# by NEXT-boundary, then the closing ``</parameter>`` is located
# within each chunk by ``rfind`` so embedded ``</parameter>`` text
# survives.
_PARAM_START_RE: re.Pattern[str] = re.compile(r"<parameter=", re.DOTALL)
_PARAM_BOUNDARY_RE: re.Pattern[str] = re.compile(
    r"<parameter=|</function>", re.DOTALL
)

# Reasoning-field probe order: OpenAI standard first, vLLM wire
# name second. Same ordering as ``client/reasoning_field_shim.py``.
_REASONING_FIELDS: tuple[str, ...] = ("reasoning_content", "reasoning")


# ---------------------------------------------------------------------------
# Typed exceptions — patch-discipline failure surface
# ---------------------------------------------------------------------------


class RescueError(Exception):
    """Root exception for the rescue module."""


class RescueShapeError(RescueError):
    """An input did not match the structural shape the rescue
    operates on. Raised when a non-dict / non-message-like object
    is passed to a function whose contract requires one."""


class RescueWriteError(RescueError):
    """A post-rescue landmark check failed: the rescue believed
    it had written ``tool_calls`` (or stripped ``reasoning``) on
    an object, but a subsequent read returned a different value.
    Most commonly indicates a frozen pydantic model that silently
    discards writes."""


# ---------------------------------------------------------------------------
# Shape protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class _MessageLike(Protocol):
    """Minimal structural protocol for a pydantic SDK Message."""

    tool_calls: Any
    content: Any


# ---------------------------------------------------------------------------
# Type-aware parameter coercion (mirrors vLLM ``_convert_param_value``)
# ---------------------------------------------------------------------------


def _convert_param_value(
    param_value: str,
    param_name: str,
    param_config: Mapping[str, Any],
    func_name: str,
) -> Any:
    """Mirror of vLLM's ``Qwen3CoderToolParser._convert_param_value``.

    Reference: pinned source at
    ``/tmp/qwen36_research/vllm_pinned_ref/vllm/tool_parsers/
    qwen3coder_tool_parser.py`` around line 113. Behaviour is
    bit-for-bit equivalent to upstream's user-visible result, with
    one deliberate omission: vLLM's fall-through ``ast.literal_eval``
    branch is replaced with degrade-to-string, because rescued
    arguments are JSON-serialised downstream and ``ast`` tuples /
    sets do not survive that round-trip.
    """
    if param_value.lower() == "null":
        return None

    if param_name not in param_config:
        return param_value

    entry = param_config[param_name]
    if isinstance(entry, dict) and "type" in entry:
        param_type = str(entry["type"]).strip().lower()
    elif isinstance(entry, dict) and "anyOf" in entry:
        # anyOf has no top-level type; vLLM treats this as "object".
        param_type = "object"
    else:
        param_type = "string"

    if param_type in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value

    if (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            return param_value

    if param_type.startswith("num") or param_type.startswith("float"):
        try:
            f = float(param_value)
            return f if (f - int(f)) != 0 else int(f)
        except (ValueError, TypeError):
            return param_value

    if param_type in ("boolean", "bool", "binary"):
        return param_value.lower() == "true"

    if (
        param_type in ("object", "array", "arr")
        or param_type.startswith("dict")
        or param_type.startswith("list")
    ):
        try:
            return json.loads(param_value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return param_value

    return param_value


def _tool_schema_properties(
    tool_schemas_by_name: Mapping[str, Mapping[str, Any]] | None,
    function_name: str,
) -> Mapping[str, Any]:
    """Extract ``properties`` for a given tool name, defensively.

    Returns an empty mapping when the schema is absent, malformed,
    or missing a ``parameters.properties`` block. An empty
    properties map degrades gracefully to all-strings in
    :func:`_convert_param_value`.

    No exception is raised here for an absent / malformed schema:
    the rescue path's contract is "type-coerce when the schema is
    present, leave-as-string when it is not". Callers who *want*
    a raise on schema absence should validate the schema before
    invoking rescue.
    """
    if not tool_schemas_by_name:
        return {}
    schema = tool_schemas_by_name.get(function_name)
    if not isinstance(schema, Mapping):
        return {}
    params = schema.get("parameters")
    if not isinstance(params, Mapping):
        return {}
    props = params.get("properties")
    if not isinstance(props, Mapping):
        return {}
    return props


# ---------------------------------------------------------------------------
# Per-parameter body extraction (the corrected boundary algorithm)
# ---------------------------------------------------------------------------


def _iter_param_bodies(params_blob: str) -> Iterator[str]:
    """Yield raw ``KEY>VALUE`` bodies for every ``<parameter=...>``
    entry in ``params_blob``.

    Slices by consecutive ``<parameter=`` start positions; each
    slice is delimited at the NEXT ``<parameter=`` OR
    ``</function>``. Within each slice, ``rfind`` locates the
    closing ``</parameter>`` so values that legitimately contain
    the literal substring ``</parameter>`` survive verbatim. This
    is the deliberate correction relative to vLLM's non-greedy
    upstream regex (README §5.4).

    If a slice has no closing ``</parameter>``, the entire slice
    after the ``<parameter=`` marker is yielded (matching vLLM's
    ``|$`` fallback). The downstream ``_parse_param_body`` is
    responsible for catching the no-``>`` case in such slices.
    """
    starts = [m.start() for m in _PARAM_START_RE.finditer(params_blob)]
    for start in starts:
        inner_start = start + len("<parameter=")
        next_boundary = _PARAM_BOUNDARY_RE.search(params_blob, start + 1)
        slice_end = next_boundary.start() if next_boundary else len(params_blob)
        chunk = params_blob[inner_start:slice_end]
        close_idx = chunk.rfind("</parameter>")
        yield chunk if close_idx == -1 else chunk[:close_idx]


def _parse_param_body(body: str) -> tuple[str, str] | None:
    """Split a ``<parameter=KEY>VALUE`` body into ``(key, value)``.

    Mirrors the fixed ``_parse_xml_function_call`` in the runtime
    patch (``monkey_patch_qwen3_coder.py``): ``.find(">")``, not
    ``.index``; strips one leading and one trailing newline from
    the value. Returns ``None`` if the key terminator ``>`` is
    absent. Caller decides whether a ``None`` here drops the call
    (rescue policy: yes).
    """
    idx = body.find(">")
    if idx == -1:
        return None
    key = body[:idx]
    value = body[idx + 1 :]
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return key, value


# ---------------------------------------------------------------------------
# Rescue core
# ---------------------------------------------------------------------------


def _build_tool_call(
    function_name: str,
    params_blob: str,
    tool_schemas_by_name: Mapping[str, Mapping[str, Any]] | None,
    index: int,
) -> dict[str, Any] | None:
    """Build one rescued tool_call dict, or ``None`` to drop the call.

    Returns ``None`` when the block is structurally salvageable at
    the outer level but at least one of its parameters is
    malformed (no ``>`` in ``<parameter=…>``). Mirrors the
    runtime-patch contract: partial arguments are worse than no
    tool call.
    """
    properties = _tool_schema_properties(tool_schemas_by_name, function_name)
    arguments: dict[str, Any] = {}

    for body in _iter_param_bodies(params_blob):
        parsed = _parse_param_body(body)
        if parsed is None:
            return None
        key, raw_value = parsed
        arguments[key] = _convert_param_value(
            raw_value, key, properties, function_name
        )

    return {
        "id": f"{_RESCUED_ID_PREFIX}{index}",
        "type": "function",
        "function": {
            "name": function_name,
            # ensure_ascii=False matches the runtime patch's
            # _parse_xml_function_call output; Qwen tool payloads
            # are routinely non-ASCII and double-escaping them
            # corrupts prompts on subsequent turns.
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _rescue_from_text(
    reasoning_text: str,
    tool_schemas_by_name: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str]:
    """Extract tool calls from a reasoning blob and return the
    stripped text.

    Returns ``(rescued_calls, stripped_reasoning)``. The stripped
    reasoning preserves pre-block and post-block text around every
    rescued block; **unrescued blocks** (truncated ``<function=``,
    malformed parameter) are left in place so the validator can
    still see them and raise.
    """
    rescued: list[dict[str, Any]] = []
    pieces: list[str] = []
    cursor = 0
    index = 0

    for match in _TOOL_CALL_BLOCK_RE.finditer(reasoning_text):
        name = match.group("name")
        if not isinstance(name, str) or not name:
            # Defensive: the regex requires a non-empty name, but
            # an upstream regex edit could relax this. Skip rather
            # than crash — the markup will still be visible to the
            # validator.
            continue
        tool_call = _build_tool_call(
            function_name=name,
            params_blob=match.group("params") or "",
            tool_schemas_by_name=tool_schemas_by_name,
            index=index,
        )
        if tool_call is None:
            # Malformed parameter: leave the markup in place for
            # the validator to observe.
            continue
        pieces.append(reasoning_text[cursor : match.start()])
        cursor = match.end()
        rescued.append(tool_call)
        index += 1

    if not rescued:
        return [], reasoning_text

    pieces.append(reasoning_text[cursor:])
    stripped = "".join(pieces)
    return rescued, stripped


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def rescue_tool_calls_from_reasoning(
    message: dict[str, Any],
    tool_schemas_by_name: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rescue ``<tool_call>`` blocks from a single assistant message
    dict.

    Mutates and returns ``message``. Idempotent: if
    ``message["tool_calls"]`` is already non-empty, the message is
    returned unchanged — an upstream tool-parser success or a
    previous rescue pass wins.

    Reads reasoning from ``reasoning_content`` first (OpenAI
    standard), then ``reasoning`` (vLLM wire name). Whichever
    field was the source of a successful rescue has its
    tool_call markup stripped; pre-block and post-block reasoning
    text is preserved.

    NEVER rescues from ``content``. Markup-in-content is a
    distinct failure mode handled by
    ``client/validate_response.py`` (raise, not rescue).

    Args:
        message: an assistant-message dict — the lowest-common-
            denominator shape that crosses the wire boundary.
        tool_schemas_by_name: optional schema map. When provided,
            parameter values are type-coerced (int / float / bool
            / object via JSON). When absent, all values are
            strings. The schema map shape is the same as
            ``client/validate_response.py`` accepts.

    Raises:
        RescueShapeError: ``message`` is not a dict.
    """
    if not isinstance(message, dict):
        raise RescueShapeError(
            f"rescue_tool_calls_from_reasoning expects a dict; got "
            f"{type(message).__name__!r}. Use "
            f"rescue_tool_calls_from_response for SDK objects."
        )

    if message.get("tool_calls"):
        return message

    for field in _REASONING_FIELDS:
        text = message.get(field)
        if not isinstance(text, str) or "<tool_call>" not in text:
            continue
        rescued, stripped = _rescue_from_text(text, tool_schemas_by_name)
        if not rescued:
            continue
        message["tool_calls"] = rescued
        message[field] = stripped
        # Post-write landmark.
        if message.get("tool_calls") is not rescued:
            raise RescueWriteError(
                f"post-rescue landmark failed: assigned "
                f"message['tool_calls'] but read back a different "
                f"object. Dict subclass with __setitem__ override?"
            )
        if message.get(field) != stripped:
            raise RescueWriteError(
                f"post-rescue landmark failed: assigned "
                f"message[{field!r}] but read back a different value."
            )
        return message

    return message


def rescue_tool_calls_from_response(
    response: Any,
    tool_schemas_by_name: Mapping[str, Mapping[str, Any]] | None = None,
) -> Any:
    """Rescue tool calls on the first choice of a ChatCompletion
    response.

    Works for both pydantic OpenAI SDK ``ChatCompletion`` objects
    and plain dicts. Mutates the first choice's message in place
    and returns the same response object.

    Contract:

    1. ``response.choices`` exists and is iterable. Otherwise
       :class:`RescueShapeError`.
    2. ``choices[0].message`` exists. Otherwise
       :class:`RescueShapeError`.
    3. If the message is a dict, delegates to
       :func:`rescue_tool_calls_from_reasoning`.
    4. If the message is a pydantic-like object, builds a shadow
       dict, runs the rescue, then writes back via ``setattr``.
       A ``setattr`` that raises (frozen model) — or one that
       silently discards (read-back returns a different value) —
       raises :class:`RescueWriteError`. There is no silent
       fallback that would leave the model in the unrescued state
       it began in.
    """
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if choices is None:
        raise RescueShapeError(
            f"response of type {type(response).__name__!r} has no "
            f"'choices' attribute or key; not a Chat Completion shape."
        )
    if not choices:
        # Empty choices: nothing to rescue, but not an error — a
        # moderation refusal can produce this. Return unchanged.
        return response

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    if message is None:
        raise RescueShapeError(
            f"choices[0] of type {type(first).__name__!r} carries no "
            f"'message' field."
        )

    if isinstance(message, dict):
        rescue_tool_calls_from_reasoning(message, tool_schemas_by_name)
        return response

    # Pydantic / SDK object path.
    if not isinstance(message, _MessageLike):
        raise RescueShapeError(
            f"choices[0].message of type {type(message).__name__!r} "
            f"matches neither dict nor _MessageLike; cannot rescue."
        )

    shadow: dict[str, Any] = {}
    for field in ("tool_calls", *_REASONING_FIELDS, "content"):
        shadow[field] = getattr(message, field, None)

    before = len(shadow.get("tool_calls") or [])
    rescue_tool_calls_from_reasoning(shadow, tool_schemas_by_name)
    after = len(shadow.get("tool_calls") or [])

    if after == before:
        return response

    # Write back. Every setattr is verified by read-after-write;
    # a failed write raises rather than degrading silently.
    _set_and_verify(message, "tool_calls", shadow["tool_calls"])
    for field in _REASONING_FIELDS:
        new_value = shadow.get(field)
        # Only write fields that actually changed (the rescue may
        # have stripped only one of the two field names).
        current_value = getattr(message, field, None)
        if new_value == current_value:
            continue
        if new_value is None:
            continue
        _set_and_verify(message, field, new_value)

    return response


def _set_and_verify(obj: Any, attr: str, value: Any) -> None:
    """Write an attribute and verify the write took effect.

    Raises :class:`RescueWriteError` if the attribute either
    refuses to be set (e.g. frozen pydantic) or returns a
    different value on read-back. Does not catch the exception
    from ``setattr`` and silently retry — patch discipline says
    a write that does not stick is a failure of the rescue's
    contract, not something to paper over.
    """
    try:
        setattr(obj, attr, value)
    except (AttributeError, TypeError) as exc:
        raise RescueWriteError(
            f"setattr({type(obj).__name__!r}, {attr!r}, ...) raised "
            f"{exc!r}. The target object is read-only or has a "
            f"validator that rejects the rescue's value. Convert to "
            f"a mutable shape (model_dump on pydantic v2) before "
            f"invoking rescue, or rescue at the dict boundary."
        ) from exc
    after = getattr(obj, attr, _MISSING)
    if after is _MISSING or after != value:
        raise RescueWriteError(
            f"post-write landmark failed on {type(obj).__name__!r}: "
            f"set {attr!r}={value!r} but read back "
            f"{'<missing>' if after is _MISSING else repr(after)}. "
            f"The rescue cannot guarantee its mutation took effect."
        )


_MISSING: Any = object()


__all__ = [
    "RescueError",
    "RescueShapeError",
    "RescueWriteError",
    "rescue_tool_calls_from_reasoning",
    "rescue_tool_calls_from_response",
]
