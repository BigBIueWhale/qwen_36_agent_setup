"""Bidirectional rename for the vLLM ``reasoning`` ↔ OpenAI ``reasoning_content`` mismatch.

Scope (this is the entire problem this file solves)
---------------------------------------------------

vLLM's wire format for thinking-mode models carries the reasoning
string under the field name ``reasoning`` on both sides of the wire:

* **Egress from vLLM**: responses populate
  ``choices[i].message.reasoning`` (non-streaming) and
  ``choices[i].delta.reasoning`` (streaming). The OpenAI-standard
  field name is ``reasoning_content``.
* **Ingest into vLLM**: on replay of prior assistant turns, vLLM
  reads ``message.reasoning`` only
  (``vllm/entrypoints/chat_utils.py:1519`` in the pinned commit) and
  ignores ``message.reasoning_content``.

That is the whole mismatch. It is a field name, nothing more.
There is no library to monkey-patch, no version to pin, no
lifecycle to install. Two transformation functions are sufficient,
one per wire direction. The caller invokes them at the exact two
points where Chat Completion payloads cross the boundary — after
``client.chat.completions.create(...)`` returns and before the next
request is sent.

Patch-discipline contract
-------------------------

Every call to every public function in this module:

1. Validates the structural shape of its argument against a
   :class:`_Shape` contract before touching any field. A response
   object that does not look like a Chat Completion (or a message
   that does not look like an OpenAI assistant message) raises
   :class:`ReasoningFieldShimError` with the specific landmark
   that failed. There is no silent "object did not match, returned
   unchanged" branch — a bad argument is a caller bug, not
   something to paper over at the boundary this file exists to
   enforce.

2. Applies the minimum-possible mutation: copy one field to another
   on one object.

3. Post-verifies the mutation succeeded on every object it touched.
   A write that silently no-ops (frozen pydantic model, dict with a
   ``__setitem__`` override that discards keys, custom wrapper that
   swallows ``setattr``) raises
   :class:`ReasoningFieldShimWriteError`. The caller needs to know
   if the rename did not take effect, because the downstream
   consequence — silent reasoning loss on the next turn — is
   exactly the failure class this shim exists to eliminate.

Nothing silent. No warnings. Either the field got renamed and the
post-verify passed, or the caller gets a specific, typed exception
naming the field that did not stick.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, TypeAlias, runtime_checkable


# ---------------------------------------------------------------------------
# Typed exceptions — patch-discipline failure surface
# ---------------------------------------------------------------------------


class ReasoningFieldShimError(Exception):
    """Structural precondition for a reasoning-field rename was violated.

    Raised when an argument passed to :func:`normalize_response_reasoning`
    or :func:`mirror_request_reasoning` does not match the shape the
    function is contractually allowed to operate on. The exception
    message names the exact landmark that failed (missing attribute,
    wrong type on a sub-field, etc.). The caller should not recover
    from this by retrying the same object; it indicates that either
    the response did not come from a Chat Completion endpoint or the
    message list is shaped wrong for the OpenAI contract.
    """


class ReasoningFieldShimWriteError(ReasoningFieldShimError):
    """A post-transformation landmark check failed.

    The shim believed it had written ``reasoning_content`` or
    ``reasoning`` on an object, but a subsequent read returned a
    different value. The most common cause is a frozen pydantic
    model or a custom wrapper whose ``__setattr__`` silently
    discards writes; the most dangerous cause is a downstream
    wrapper that accepts the write to ``__dict__`` but still
    serializes a stale value. Either way, the shim's contract
    cannot be upheld and the caller must be told.
    """


# ---------------------------------------------------------------------------
# Shape protocols — landmark surface for duck typing
# ---------------------------------------------------------------------------


JsonLikeMessage: TypeAlias = "dict[str, Any] | _MessageLike"


@runtime_checkable
class _MessageLike(Protocol):
    """Structural protocol for pydantic/SDK message-like objects.

    Matched by both ``openai.types.chat.ChatCompletionMessage`` and
    the streaming ``ChoiceDelta``. Must expose both ``reasoning``
    and ``reasoning_content`` as attributes readable via
    ``getattr`` (the SDK defaults them to ``None`` when absent).
    """

    reasoning: Any
    reasoning_content: Any


# ---------------------------------------------------------------------------
# Shape probes — "looks like X" landmarks that raise on failure
# ---------------------------------------------------------------------------


_SENTINEL: Any = object()


def _is_mapping(obj: Any) -> bool:
    """True if ``obj`` responds to ``__getitem__`` / ``__setitem__``
    with string keys the way a plain dict does. Used as the first
    landmark before we try to treat a value as a dict."""
    return isinstance(obj, dict)


def _probe_field(obj: Any, field: str) -> Any:
    """Read a field from a mapping-or-attribute object.

    Returns ``_SENTINEL`` when the field is truly absent. Does NOT
    return ``None`` for absent because ``None`` is a valid
    reasoning value ("no reasoning this turn") and collapsing the
    two breaks the post-verify step — we have to be able to tell
    "wrote None and it stuck" apart from "wrote None and it
    silently vanished".
    """
    if _is_mapping(obj):
        if field in obj:
            return obj[field]
        return _SENTINEL
    value = getattr(obj, field, _SENTINEL)
    return value


def _write_field(obj: Any, field: str, value: Any) -> None:
    """Write a field to a mapping-or-attribute object, then
    landmark-verify the write stuck.

    Raises :class:`ReasoningFieldShimWriteError` if a read-after-
    write does not return the value we just set. ``None`` is a
    legal value; the post-verify uses ``is``-identity on the
    ``_SENTINEL`` absence marker rather than truthiness, so
    writing ``None`` is supported and is NOT confused with the
    write silently no-opping.
    """
    if _is_mapping(obj):
        obj[field] = value
    else:
        try:
            setattr(obj, field, value)
        except (AttributeError, TypeError) as exc:
            raise ReasoningFieldShimWriteError(
                f"cannot set attribute {field!r} on object of type "
                f"{type(obj).__name__!r}: {exc!r}. If this is a "
                f"frozen pydantic model, either re-construct it "
                f"(model_copy with update={{{field!r}: ...}}) or "
                f"operate on a dict view (model_dump()) before "
                f"invoking the shim."
            ) from exc

    after = _probe_field(obj, field)
    if after is _SENTINEL or after != value:
        raise ReasoningFieldShimWriteError(
            f"post-write verification failed on {type(obj).__name__!r}: "
            f"set {field!r}={value!r} but read back "
            f"{'<absent>' if after is _SENTINEL else repr(after)}. The "
            f"shim cannot guarantee its rename took effect; downstream "
            f"reasoning loss is certain if this proceeds."
        )


# ---------------------------------------------------------------------------
# Response-side rename: vLLM reasoning  →  OpenAI reasoning_content
# ---------------------------------------------------------------------------


def _iter_choice_fragments(response: Any) -> Iterator[Any]:
    """Yield every per-choice fragment where a reasoning field may
    live — i.e. each ``choice.message`` (non-streaming) and each
    ``choice.delta`` (streaming). A Chat Completion response never
    has both populated on the same choice at once, so yielding
    both shapes when present is safe.

    Landmark-verifies that ``response.choices`` exists and is
    iterable. An empty list is fine (some moderation refusals
    return ``choices=[]``); an absent ``choices`` attribute is a
    bug and raises :class:`ReasoningFieldShimError`.
    """
    choices = _probe_field(response, "choices")
    if choices is _SENTINEL:
        raise ReasoningFieldShimError(
            f"object of type {type(response).__name__!r} has no "
            f"'choices' attribute; it does not look like an OpenAI "
            f"Chat Completion response. The shim refuses to guess; "
            f"pass the actual response object returned by "
            f"client.chat.completions.create(...)."
        )
    if not isinstance(choices, (list, tuple)):
        raise ReasoningFieldShimError(
            f"response.choices is of type {type(choices).__name__!r}, "
            f"not a list/tuple; the OpenAI contract requires it to "
            f"be a sequence of Choice objects."
        )

    for idx, choice in enumerate(choices):
        found = False
        for fragment_field in ("message", "delta"):
            fragment = _probe_field(choice, fragment_field)
            if fragment is _SENTINEL or fragment is None:
                continue
            found = True
            yield fragment
        if not found:
            raise ReasoningFieldShimError(
                f"choices[{idx}] on object of type "
                f"{type(choice).__name__!r} carries neither 'message' "
                f"nor 'delta'. Every Chat Completion choice carries "
                f"one of the two; this object does not match the "
                f"contract."
            )


def normalize_response_reasoning(response: Any) -> Any:
    """Rename ``reasoning`` → ``reasoning_content`` on every choice.

    Reads ``choices[i].message.reasoning`` (non-streaming) or
    ``choices[i].delta.reasoning`` (streaming) and writes the
    same value into ``reasoning_content`` on the same fragment,
    so callers that follow the OpenAI standard on the
    *application* side see the reasoning under the field name
    they expect.

    Contract, in order:

    1. ``response.choices`` exists and is a list/tuple. Otherwise
       :class:`ReasoningFieldShimError`.
    2. Every choice carries a ``message`` or a ``delta``. Otherwise
       :class:`ReasoningFieldShimError`.
    3. For each fragment on each choice:
       * If ``reasoning_content`` is already set to a non-None
         value, the fragment is skipped — upstream has already
         normalized and re-applying would be a no-op. This is the
         ONLY shape of "skip" this function accepts, and it is
         landmark-verified rather than guessed: the incoming
         ``reasoning_content`` value is read back post-function to
         confirm nothing was lost.
       * Otherwise ``reasoning_content`` is written to the current
         value of ``reasoning`` (which may itself be ``None`` if
         the turn had no reasoning; that ``None`` is intentionally
         mirrored, not elided).
    4. After all fragments have been processed, every fragment is
       re-probed and must have ``reasoning_content`` and
       ``reasoning`` exposing the same value. Any mismatch raises
       :class:`ReasoningFieldShimWriteError`.

    Idempotent. Mutates and returns the same object (the SDK
    ``ChatCompletion`` / ``ChatCompletionChunk``).
    """
    touched: list[tuple[Any, Any]] = []

    for fragment in _iter_choice_fragments(response):
        existing_rc = _probe_field(fragment, "reasoning_content")
        existing_r = _probe_field(fragment, "reasoning")

        if existing_rc is _SENTINEL and existing_r is _SENTINEL:
            raise ReasoningFieldShimError(
                f"fragment of type {type(fragment).__name__!r} carries "
                f"neither 'reasoning' nor 'reasoning_content'. A "
                f"thinking-mode Chat Completion response from vLLM "
                f"always populates one; its absence indicates either "
                f"a non-thinking server or a response object that is "
                f"not what the shim was designed to handle."
            )

        # If reasoning_content is already populated with a non-None
        # value, respect it. The "non-None" gate is the landmark:
        # an explicit None on reasoning_content means "no reasoning
        # this turn" and we still want to mirror the vLLM field into
        # it if vLLM produced something.
        if existing_rc is not _SENTINEL and existing_rc is not None:
            touched.append((fragment, existing_rc))
            continue

        source_value = existing_r if existing_r is not _SENTINEL else None
        _write_field(fragment, "reasoning_content", source_value)
        # Mirror back too: the caller may pass this same fragment
        # to mirror_request_reasoning on a later turn, and we
        # want both names in sync regardless of direction.
        if existing_r is _SENTINEL:
            _write_field(fragment, "reasoning", source_value)
        touched.append((fragment, source_value))

    # Post-verify every fragment we touched carries both names
    # equal. An SDK wrapper that silently coerces one but not the
    # other is exactly the failure class we must not ship through.
    for fragment, expected in touched:
        rc = _probe_field(fragment, "reasoning_content")
        r = _probe_field(fragment, "reasoning")
        if rc is _SENTINEL or r is _SENTINEL or rc != r:
            raise ReasoningFieldShimWriteError(
                f"post-normalize landmark failed on "
                f"{type(fragment).__name__!r}: "
                f"reasoning_content={rc!r}, reasoning={r!r}, "
                f"expected both to equal {expected!r}"
            )

    return response


# ---------------------------------------------------------------------------
# Request-side mirror: OpenAI reasoning_content  →  vLLM reasoning
# ---------------------------------------------------------------------------


def mirror_request_reasoning(messages: list[JsonLikeMessage]) -> list[JsonLikeMessage]:
    """Copy ``reasoning_content`` → ``reasoning`` on every assistant
    message in an outgoing chat request, so vLLM's ingest reader
    (``vllm/entrypoints/chat_utils.py:1519``) sees the reasoning on
    replay of prior turns.

    Contract, in order:

    1. ``messages`` is a non-empty list. Otherwise
       :class:`ReasoningFieldShimError` — an empty request body is
       a caller bug and we refuse to silently forward it.
    2. Each element is either a ``dict`` or matches
       :class:`_MessageLike`. A bare string or a namedtuple would
       indicate the caller serialized wrong upstream; raise.
    3. Each element carries a ``role`` field readable via the same
       mapping-or-attribute probe the rest of the module uses.
       Absence raises.
    4. For each ``role == "assistant"`` message:
       * If ``reasoning_content`` is a non-None string, copy its
         value into ``reasoning``.
       * If ``reasoning_content`` is absent or ``None`` but
         ``reasoning`` is a non-None string, copy in the other
         direction so both names carry the same value. (Supports
         conversation histories that were already vLLM-shaped.)
       * If both are absent or both None, skip — no reasoning on
         this turn is a legal state.
    5. After mutation, re-probe and require
       ``reasoning == reasoning_content`` on every touched
       message. :class:`ReasoningFieldShimWriteError` otherwise.

    Idempotent. Mutates and returns the same list.
    """
    if not isinstance(messages, list):
        raise ReasoningFieldShimError(
            f"messages argument is of type {type(messages).__name__!r}, "
            f"not a list. The OpenAI Chat Completions contract requires "
            f"a list of message dicts/objects."
        )
    if not messages:
        raise ReasoningFieldShimError(
            "messages is empty. An empty messages list is a caller "
            "bug — vLLM would reject it with an HTTP 400 anyway, and "
            "the shim refuses to silently forward a broken payload."
        )

    touched: list[tuple[Any, Any]] = []

    for idx, message in enumerate(messages):
        if not (_is_mapping(message) or isinstance(message, _MessageLike)):
            raise ReasoningFieldShimError(
                f"messages[{idx}] is of type {type(message).__name__!r}, "
                f"which is neither a dict nor a structural match for "
                f"_MessageLike. The shim handles dicts and pydantic "
                f"SDK message objects; serialize to one of those before "
                f"invoking."
            )

        role = _probe_field(message, "role")
        if role is _SENTINEL:
            raise ReasoningFieldShimError(
                f"messages[{idx}] has no 'role' field. Every OpenAI "
                f"chat message must declare a role; refusing to guess."
            )

        if role != "assistant":
            continue

        rc = _probe_field(message, "reasoning_content")
        r = _probe_field(message, "reasoning")

        # Pick the canonical value. reasoning_content wins because
        # it is the OpenAI-standard name; this keeps double-application
        # deterministic (a message that already carries both equal
        # fields stays untouched).
        canonical: Any
        if rc is not _SENTINEL and rc is not None:
            canonical = rc
        elif r is not _SENTINEL and r is not None:
            canonical = r
        else:
            # Both absent or both None. Not an error; this turn simply
            # has no reasoning to mirror. Do not write anything; a
            # missing field is distinct from an explicitly-None field
            # and we do not want to inject None attributes onto SDK
            # objects that do not have them.
            continue

        if rc is _SENTINEL or rc != canonical:
            _write_field(message, "reasoning_content", canonical)
        if r is _SENTINEL or r != canonical:
            _write_field(message, "reasoning", canonical)
        touched.append((message, canonical))

    for message, expected in touched:
        rc = _probe_field(message, "reasoning_content")
        r = _probe_field(message, "reasoning")
        if rc != expected or r != expected:
            raise ReasoningFieldShimWriteError(
                f"post-mirror landmark failed on message of type "
                f"{type(message).__name__!r}: "
                f"reasoning_content={rc!r}, reasoning={r!r}, "
                f"expected both to equal {expected!r}"
            )

    return messages


__all__ = [
    "ReasoningFieldShimError",
    "ReasoningFieldShimWriteError",
    "normalize_response_reasoning",
    "mirror_request_reasoning",
]
