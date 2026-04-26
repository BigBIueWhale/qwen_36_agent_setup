"""Strict, fail-loud egress patch renaming vLLM's non-standard ``reasoning`` field
to the OpenAI-spec ``reasoning_content`` on the wire — at every nesting level
vLLM actually serialises.

Target: vLLM commit ``32e45636e3d7e02615facc8c63645ce4ac1d7e11`` (README §3.2).
Egress half of the §6.4 wire-format mismatch; the ingest half is
``monkey_patch_reasoning_field_ingest.py`` (§7.4). Same fail-loud
import-time discipline as every other patch in this repo.

What this fixes
---------------

The OpenAI Chat Completions spec names the chain-of-thought / reasoning
channel ``reasoning_content`` on both the full-response ``message`` object
and each streaming ``delta``. vLLM's pinned commit emits the bare
non-standard name ``reasoning`` instead. Concretely, at
``vllm/entrypoints/openai/chat_completion/protocol.py`` line 54-64::

    class ChatMessage(OpenAIBaseModel):
        role: str
        content: str | None = None
        ...
        # vLLM-specific fields that are not in OpenAI spec
        reasoning: str | None = None

and at ``vllm/entrypoints/openai/engine/protocol.py`` line 258-262::

    class DeltaMessage(OpenAIBaseModel):
        role: str | None = None
        content: str | None = None
        reasoning: str | None = None
        tool_calls: list[DeltaToolCall] = Field(default_factory=list)

OpenAI-conforming clients look under ``reasoning_content``, find nothing,
and silently lose the reasoning channel. Our deployment uses
**Qwen-Agent**, whose OAI client at
``Qwen-Agent/qwen_agent/llm/oai.py:111-112,126-127,169`` strictly checks
``reasoning_content`` with **no fallback**, so it sees no reasoning at
all under the bare-vLLM wire format. Qwen Code CLI has a
``reasoning_content ?? reasoning`` fallback at
``packages/core/src/core/openaiContentGenerator/converter.ts:818-819``;
without this patch it survives, but only by coupling itself to vLLM's
non-standard name.

The trace, end-to-end:

* vLLM emits ``{"choices":[{"message":{"role":"assistant",
  "reasoning":"<chain>","content":"...","tool_calls":[...]}}]}``.
* Qwen-Agent reads ``message.reasoning_content`` → None. Reasoning lost.
* On turn 2 it sends back ``{"role":"assistant",
  "reasoning_content":"<chain>", ...}``; the chat template at
  ``chat_template.jinja:91-92`` reads ``message.reasoning_content``
  for ``preserve_thinking=true`` rendering. With the field round-trip
  broken at *both* ends (egress here, ingest in §7.4), the historical
  ``<think>`` block never reaches the prompt.
* The model — RL-trained to expect prior-turn reasoning — re-derives
  context from scratch. Tool args degrade (``badlogic/pi-mono#3325``)
  after 2-3 turns. README §5.7 cites this exact failure mode.

Why patch v1 didn't actually fix it (and v2 does)
-------------------------------------------------

v1 of this patch (the previous file at this path) flipped
``serialize_by_alias=True`` and rebuilt the Pydantic schema on
``ChatMessage`` and ``DeltaMessage`` only — the two **leaf** classes.
That works when those classes are dumped *standalone*. **It does not
work when they are dumped nested inside the response wrappers**, which
is what every production code path in vLLM actually does. Empirically
confirmed against Pydantic 2.13.3 (which matches vLLM's
``pydantic >= 2.12.0`` pin in ``requirements/common.txt``):

    Inner.model_config["serialize_by_alias"] = True; Inner rebuilt.
    Outer(inner=Inner(reasoning="probe")).model_dump_json()
    → '{"inner":{"role":"assistant","reasoning":"probe"}}'    # WRONG

    # Then also flip+rebuild Outer:
    Outer.model_config["serialize_by_alias"] = True; Outer rebuilt.
    Outer(inner=Inner(reasoning="probe")).model_dump_json()
    → '{"inner":{"role":"assistant","reasoning_content":"probe"}}' # RIGHT

And critically, every intermediate level of nesting matters:
patching only Inner + Outer while leaving an intermediate ``Mid``
unrebuilt re-emits ``reasoning`` because ``Outer``'s regenerated
core_schema embeds ``Mid``'s old (unrebuilt) core_schema, which embeds
``Inner``'s old core_schema even though Inner has been rebuilt
elsewhere — Pydantic v2's compiled core schemas embed nested schemas by
snapshot at build time, not by reference to the live class.

The empirical conclusion: every class in the chain from leaf to
outermost wire-dump target must have its compiled schema regenerated
under ``serialize_by_alias=True``. v2 patches all six classes that
appear on any vLLM dump path that emits the reasoning channel:

* ``ChatMessage`` — leaf, sets the alias on the field.
* ``DeltaMessage`` — leaf, sets the alias on the field.
* ``ChatCompletionResponseChoice`` — intermediate (``message: ChatMessage``).
* ``ChatCompletionResponseStreamChoice`` — intermediate (``delta: DeltaMessage``).
* ``ChatCompletionResponse`` — outermost non-streaming wrapper
  (``choices: list[ChatCompletionResponseChoice]``); dumped at
  ``api_router.py:70``.
* ``ChatCompletionStreamResponse`` — outermost streaming wrapper
  (``choices: list[ChatCompletionResponseStreamChoice]``); dumped at
  ``serving.py:685, 721, 1208, 1233``.

The leaves carry the field alias. The wrappers carry only the
class-wide ``serialize_by_alias=True`` flag (no fields renamed). All
six get their compiled core schemas regenerated.

Patch-discipline contract
-------------------------

This file is a patch, not a library. At import it:

1. Imports vLLM. Failure is a hard ImportError; we do not hide it.
2. Imports all six target classes by their pinned paths and asserts
   each is a class and a subclass of ``pydantic.BaseModel``.
3. For each LEAF, verifies it declares the ``reasoning`` field with
   annotation exactly ``str | None``, plus the structural neighbors
   (``role``, ``content``, ``tool_calls``) we expect on a ChatMessage-
   shaped class. Refuses if neighbors are missing.
4. For each WRAPPER, verifies the nested field connecting it to the
   leaf has the expected concrete type (``message: ChatMessage`` /
   ``delta: DeltaMessage`` / ``choices: list[…Choice]``). Refuses
   on any drift — nesting rebuild is structurally load-bearing and a
   silent rename here would silently break the rebuild chain.
5. Audits ``serialization_alias`` on every field of every target. The
   class-wide ``serialize_by_alias=True`` flip activates **all**
   aliases on a class; if any non-target field already carries one,
   we refuse so the operator re-audits before changing wire behavior.
6. Refuses if any target already has ``model_config["serialize_by_alias"]``
   pre-set to a value other than False/absent — that would indicate
   another patch or a future upstream change is racing this one.
7. Refuses if any target already declares a ``reasoning_content`` field
   — upstream may have started its own implementation that this patch
   would collide with.
8. Installs:

   * For each leaf: mutate the ``reasoning`` field's
     ``serialization_alias`` to ``"reasoning_content"``, set
     ``cls.model_config["serialize_by_alias"] = True``, drop cached
     core_schema/validator/serializer, mark ``__pydantic_complete__``
     False, and call ``model_rebuild(force=True)``.
   * For each wrapper (in dependency order — leaves' direct parents
     before outermost containers): set
     ``cls.model_config["serialize_by_alias"] = True``, drop the
     cached schema attrs, mark incomplete, ``model_rebuild(force=True)``.
   * Stamp ``__qwen36_egress_patch__`` on every target.

9. Verifies, on each leaf class:

   * Internal Python attribute ``.reasoning`` still works.
   * ``cls(reasoning=probe).model_dump()`` emits ``reasoning_content``
     and not ``reasoning``.
   * Same for ``model_dump_json()``,
     ``model_dump(exclude_unset=True)``,
     ``model_dump_json(exclude_unset=True)``.

10. Verifies, **on real nested dumps** — this is the load-bearing
    addition over v1:

    * ``ChatCompletionResponse(id=…, model=…, usage=…,
      choices=[ChatCompletionResponseChoice(index=0,
      message=ChatMessage(role="assistant", reasoning=probe))]).model_dump_json()``
      contains ``"reasoning_content":"<probe>"`` and does NOT contain
      ``"reasoning":``.
    * ``ChatCompletionStreamResponse(id=…, model=…,
      choices=[ChatCompletionResponseStreamChoice(index=0,
      delta=DeltaMessage(reasoning=probe))]).model_dump_json(exclude_unset=True)``
      same property holds.

    **This is the regression v1's verifier missed.**

Any step 1-10 failing raises :class:`EgressPatchRefusedError` and the
interpreter does not continue. There is **no** ``SystemExit(0)``,
``try/except Exception: pass``, or silent fallback on any install path.
A refused patch is the desired behavior — the deployment treats refusal
as a hard boot failure.

Critical correctness invariants
-------------------------------

* **The Python attribute is NOT renamed.** vLLM's own code reads
  ``delta_message.reasoning`` (and the equivalent on ``ChatMessage``)
  as a plain attribute. This patch touches only the serialization
  alias and the class-wide ``serialize_by_alias`` flag.
* **No other field's behavior changes.** Phase 5 refuses if any other
  field on any of the six target classes has a pre-existing
  serialization alias.
* **Construction by the original field name still works.** Pydantic's
  ``serialization_alias`` (distinct from ``alias`` and
  ``validation_alias``) affects only egress.
* **No collateral effects on sibling classes.** Pydantic v2 makes a
  per-subclass copy of ``model_config`` at class creation (verified
  empirically against 2.13.3); flipping the flag on a target does not
  propagate to siblings via the shared ``OpenAIBaseModel`` parent.
* **Rebuild order matters.** Leaves first, then intermediates, then
  outermost wrappers. Pydantic v2 compiled core schemas embed nested
  schemas by snapshot at build time; rebuilding the inner first is the
  only way the outer's rebuild picks up the new inner schema.
"""

from __future__ import annotations

import inspect
from typing import Any


_PINNED_VLLM_COMMIT: str = "32e45636e3d7e02615facc8c63645ce4ac1d7e11"
_PATCH_TAG: str = "qwen36-agent-setup-reasoning-egress-v2"

# Source field name (Python attribute) and the OpenAI-spec alias we
# emit on the wire. The attribute name is NOT changed; only the
# serialized key.
_FIELD_NAME: str = "reasoning"
_SERIALIZATION_ALIAS: str = "reasoning_content"

# Sentinel value for post-install verification. Distinguishable from
# any legitimate content; short enough to substring-search without
# ambiguity.
_PROBE_VALUE: str = "__qwen36_egress_probe__"

# Structural-neighbor landmarks for LEAF classes. Their presence on
# each leaf confirms we are patching the class we think we are
# patching, not some unrelated class upstream happened to rename.
_LEAF_STRUCTURAL_NEIGHBORS: tuple[str, ...] = ("role", "content", "tool_calls")


class EgressPatchRefusedError(RuntimeError):
    """A precondition for the reasoning-field egress rename was violated.

    Raised at import time only. The patch either applies cleanly and
    correctly across all six target classes, or the process does not
    come up. A half-installed egress rename — leaves patched, wrappers
    not, like v1 of this patch — leaves the wire emitting the
    non-standard ``reasoning`` key while the patch's tag and verifier
    falsely indicate success. We refuse to enter that state.
    """


def _require(condition: object, failure_message: str) -> None:
    if not condition:
        raise EgressPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {failure_message}"
        )


# --------------------------------------------------------------------
# Phase 1: Import vLLM and pydantic, locate the six target classes.
# --------------------------------------------------------------------

import vllm  # noqa: F401  — availability landmark; must not be guarded

from pydantic import BaseModel

from vllm.logger import init_logger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    UsageInfo,
)

_logger = init_logger(__name__)


# Leaf targets: the two classes whose ``reasoning`` field gets the
# OpenAI-spec alias. Order is irrelevant.
_LEAF_TARGETS: tuple[tuple[str, type[BaseModel]], ...] = (
    (
        "vllm.entrypoints.openai.chat_completion.protocol.ChatMessage",
        ChatMessage,
    ),
    (
        "vllm.entrypoints.openai.engine.protocol.DeltaMessage",
        DeltaMessage,
    ),
)

# Wrapper targets, in REBUILD ORDER (leaves' immediate containers
# first, then their containers, then …). Each entry is
# ``(qualname, cls, nested_field_name, expected_inner_kind)``:
#
# * ``nested_field_name`` is the field on the wrapper whose type
#   transitively reaches a leaf. Phase 4 verifies its presence.
# * ``expected_inner_kind`` is one of:
#
#   - a class (``ChatMessage`` / ``DeltaMessage``): the field's
#     annotation must equal it.
#   - the literal ``list[ChatCompletionResponseChoice]`` /
#     ``list[ChatCompletionResponseStreamChoice]``: the field's
#     annotation must equal it.
#
#   We check against the live runtime annotation object, so PEP 604
#   union forms and ``typing.List[...]`` aliases compare correctly.
_WRAPPER_TARGETS: tuple[tuple[str, type[BaseModel], str, Any], ...] = (
    (
        "vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionResponseChoice",
        ChatCompletionResponseChoice,
        "message",
        ChatMessage,
    ),
    (
        "vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionResponseStreamChoice",
        ChatCompletionResponseStreamChoice,
        "delta",
        DeltaMessage,
    ),
    (
        "vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionResponse",
        ChatCompletionResponse,
        "choices",
        list[ChatCompletionResponseChoice],
    ),
    (
        "vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionStreamResponse",
        ChatCompletionStreamResponse,
        "choices",
        list[ChatCompletionResponseStreamChoice],
    ),
)

# Combined target list for the audits that apply to every target
# (Phase 3: alias audit; Phase 6: tag verification). NOT used for
# rebuild ordering — that uses the per-tier lists above.
_ALL_TARGETS: tuple[tuple[str, type[BaseModel]], ...] = tuple(
    [(qn, c) for qn, c in _LEAF_TARGETS]
    + [(qn, c) for (qn, c, _, _) in _WRAPPER_TARGETS]
)


# --------------------------------------------------------------------
# Phase 2: Per-target structural verification.
# --------------------------------------------------------------------

# (a) Every target must be a class, a BaseModel subclass.
for _qualname, _cls in _ALL_TARGETS:
    _require(
        inspect.isclass(_cls),
        f"{_qualname} is no longer a class (got {type(_cls).__name__!r}).",
    )
    _require(
        issubclass(_cls, BaseModel),
        f"{_qualname} is no longer a subclass of pydantic.BaseModel. "
        f"Upstream has restructured the response-protocol class "
        f"hierarchy and this patch's Pydantic v2 serialization-alias "
        f"approach may no longer apply.",
    )

# (b) Each leaf must declare the ``reasoning`` field with the expected
# annotation, plus the structural neighbors we expect on a ChatMessage-
# shaped class. ``reasoning_content`` must NOT already be a field
# (upstream collision check).
for _qualname, _cls in _LEAF_TARGETS:
    _fields = _cls.model_fields
    _require(
        _FIELD_NAME in _fields,
        f"{_qualname} no longer declares a {_FIELD_NAME!r} field. "
        f"Upstream has removed or renamed the reasoning channel; "
        f"re-audit the patch before bumping the pinned commit.",
    )
    _ann = _fields[_FIELD_NAME].annotation
    _expected_union = str | None
    _require(
        _ann == _expected_union,
        f"{_qualname}.{_FIELD_NAME} annotation changed: expected "
        f"{_expected_union!r}, got {_ann!r}. The wire-type of the "
        f"reasoning channel has changed; re-audit before patching.",
    )
    for _neighbor in _LEAF_STRUCTURAL_NEIGHBORS:
        _require(
            _neighbor in _fields,
            f"{_qualname} no longer declares the expected structural "
            f"neighbor {_neighbor!r}. This is a catastrophic schema "
            f"change; refusing to patch a class whose shape we no "
            f"longer recognize.",
        )
    _require(
        _SERIALIZATION_ALIAS not in _fields,
        f"{_qualname} already declares a {_SERIALIZATION_ALIAS!r} "
        f"field. Upstream appears to have added the OpenAI-spec field "
        f"on its own; this patch would collide. Remove this patch and "
        f"verify upstream's implementation is correct.",
    )

# (c) Each wrapper must expose the expected nested-field link to its
# inner. This is what makes the rebuild chain structurally sound — a
# rename or type drift here would silently break it.
for _qualname, _cls, _nested_name, _expected_inner in _WRAPPER_TARGETS:
    _fields = _cls.model_fields
    _require(
        _nested_name in _fields,
        f"{_qualname}.{_nested_name} field is missing. The rebuild "
        f"chain assumes this field connects {_qualname} to its "
        f"reasoning-bearing nested type. Upstream has restructured "
        f"the response wrapper; refusing to patch.",
    )
    _ann = _fields[_nested_name].annotation
    _require(
        _ann == _expected_inner,
        f"{_qualname}.{_nested_name} annotation changed: expected "
        f"{_expected_inner!r}, got {_ann!r}. The wrapper's nested "
        f"type no longer matches the rebuild chain's assumption.",
    )
    _require(
        _SERIALIZATION_ALIAS not in _fields,
        f"{_qualname} already declares a {_SERIALIZATION_ALIAS!r} "
        f"field; upstream collision — refusing.",
    )


# --------------------------------------------------------------------
# Phase 3: Audit existing serialization aliases on every target.
# --------------------------------------------------------------------
#
# Flipping ``serialize_by_alias=True`` at the class level activates ALL
# aliases on that class; if any non-target field already carries one,
# we must refuse so the operator re-audits before changing wire
# behavior of an unrelated field.

for _qualname, _cls in _ALL_TARGETS:
    for _fname, _finfo in _cls.model_fields.items():
        _existing_alias = getattr(_finfo, "serialization_alias", None)
        if _qualname.endswith(".ChatMessage") or _qualname.endswith(".DeltaMessage"):
            # On leaves, the ``reasoning`` field is expected to have
            # NO alias right now — we are about to set it.
            if _fname == _FIELD_NAME:
                _require(
                    _existing_alias is None,
                    f"{_qualname}.{_fname} already carries "
                    f"serialization_alias={_existing_alias!r}. Expected "
                    f"None at patch time; upstream may have already "
                    f"attempted this rename. Refusing.",
                )
                continue
        _require(
            _existing_alias is None,
            f"{_qualname}.{_fname} carries serialization_alias="
            f"{_existing_alias!r}. This patch flips the class-wide "
            f"``serialize_by_alias`` flag, which would activate that "
            f"alias on the wire. Refusing to silently change a second "
            f"field's serialization; re-audit before proceeding.",
        )


# --------------------------------------------------------------------
# Phase 4: Audit existing model_config["serialize_by_alias"] state.
# --------------------------------------------------------------------
#
# We are about to set this to True on every target. Refuse if any
# target already has it set to True — another patch or a future
# upstream may be racing this one and we should re-audit, not silently
# layer on top.

for _qualname, _cls in _ALL_TARGETS:
    _existing_flag = _cls.model_config.get("serialize_by_alias", False)
    _require(
        _existing_flag is False,
        f"{_qualname}.model_config['serialize_by_alias'] is already "
        f"{_existing_flag!r} at patch time. Another patch or future "
        f"upstream change is competing with this one; refusing.",
    )


# --------------------------------------------------------------------
# Phase 5: Install — leaves first (alias + flag + rebuild), then
#          wrappers in dependency order (flag + rebuild).
# --------------------------------------------------------------------

# Attributes Pydantic caches after class creation and that need to be
# dropped explicitly so model_rebuild(force=True) regenerates a fresh
# compiled schema rather than reusing the stale C-extension serializer.
_CACHED_SCHEMA_ATTRS: tuple[str, ...] = (
    "__pydantic_core_schema__",
    "__pydantic_validator__",
    "__pydantic_serializer__",
)


def _drop_cached_schema(cls: type[BaseModel]) -> None:
    """Remove the three cached schema attributes from ``cls.__dict__``.

    ``model_rebuild(force=True)`` alone is not sufficient in Pydantic
    2.13: the compiled C-extension serializer is cached and not
    regenerated unless the cached attributes are absent and
    ``__pydantic_complete__`` is False at the time of the rebuild call.
    Verified empirically.
    """
    for attr in _CACHED_SCHEMA_ATTRS:
        if attr in cls.__dict__:
            delattr(cls, attr)
    cls.__pydantic_complete__ = False


def _install_leaf(cls: type[BaseModel]) -> None:
    """Apply the alias + flag + rebuild on a leaf class.

    The leaf carries the actual field rename. The wrappers above it
    only carry the class-wide flag; without the flag set on every
    enclosing class up to the outermost wire-dump target, the
    compiled core_schema of the outermost class will not honor the
    leaf's alias when serialising nested instances.
    """
    cls.model_fields[_FIELD_NAME].serialization_alias = _SERIALIZATION_ALIAS
    cls.model_config["serialize_by_alias"] = True
    _drop_cached_schema(cls)
    cls.model_rebuild(force=True)
    setattr(cls, "__qwen36_egress_patch__", _PATCH_TAG)


def _install_wrapper(cls: type[BaseModel]) -> None:
    """Apply the flag + rebuild on a wrapper class.

    No FieldInfo mutation: the wrapper has no ``reasoning`` field of
    its own. The flag is what makes the wrapper's regenerated schema
    propagate ``serialize_by_alias=True`` into its embedded nested
    schemas — without it, dumping the wrapper bypasses the leaf's
    alias even though the leaf itself was rebuilt with the alias set.
    """
    cls.model_config["serialize_by_alias"] = True
    _drop_cached_schema(cls)
    cls.model_rebuild(force=True)
    setattr(cls, "__qwen36_egress_patch__", _PATCH_TAG)


# Install in dependency order. Leaves first so their compiled schemas
# carry the alias before any wrapper rebuilds and snapshots them.
for _qualname, _cls in _LEAF_TARGETS:
    _install_leaf(_cls)
for _qualname, _cls, _nested_name, _expected_inner in _WRAPPER_TARGETS:
    _install_wrapper(_cls)


# --------------------------------------------------------------------
# Phase 6: Per-leaf post-install verification.
# --------------------------------------------------------------------

def _verify_leaf(cls: type[BaseModel], qualname: str) -> None:
    """Assert that ``cls`` (a leaf) serializes ``reasoning`` as
    ``reasoning_content`` on every relevant standalone-dump path and
    that the Python attribute survives intact.

    Failures here raise :class:`EgressPatchRefusedError` via
    :func:`_require`. There is no fallback path; an assertion failure
    means the patch did not take effect on this leaf and boot must fail.
    """
    if "role" in cls.model_fields and cls.model_fields["role"].is_required():
        instance = cls(role="assistant", reasoning=_PROBE_VALUE)
    else:
        instance = cls(reasoning=_PROBE_VALUE)

    # (a) Internal Python attribute still works.
    _require(
        getattr(instance, _FIELD_NAME, None) == _PROBE_VALUE,
        f"post-install verification failed on {qualname}: "
        f".{_FIELD_NAME} attribute did not round-trip the probe value. "
        f"The patch has accidentally renamed or shadowed the Python "
        f"attribute — vLLM's own code that reads this attribute would "
        f"break. Refusing.",
    )

    # (b) model_dump() emits the alias, no bare reasoning key.
    dumped = instance.model_dump()
    _require(
        isinstance(dumped, dict),
        f"post-install verification failed on {qualname}: "
        f"model_dump() returned {type(dumped).__name__!r}, expected dict.",
    )
    _require(
        dumped.get(_SERIALIZATION_ALIAS) == _PROBE_VALUE,
        f"post-install verification failed on {qualname}: "
        f"model_dump()[{_SERIALIZATION_ALIAS!r}] is "
        f"{dumped.get(_SERIALIZATION_ALIAS)!r}, expected the probe "
        f"value {_PROBE_VALUE!r}. The serialization_alias did not "
        f"take effect on the leaf.",
    )
    _require(
        _FIELD_NAME not in dumped,
        f"post-install verification failed on {qualname}: "
        f"model_dump() still contains the bare key {_FIELD_NAME!r} "
        f"(value={dumped.get(_FIELD_NAME)!r}). The alias was applied "
        f"but the unaliased key is still being emitted alongside it — "
        f"this would double-emit the reasoning channel. Full dump: "
        f"{dumped!r}",
    )

    # (c) model_dump_json() emits the alias, no bare reasoning key.
    dumped_json = instance.model_dump_json()
    _require(
        isinstance(dumped_json, str),
        f"post-install verification failed on {qualname}: "
        f"model_dump_json() returned {type(dumped_json).__name__!r}, "
        f"expected str.",
    )
    _require(
        f'"{_SERIALIZATION_ALIAS}"' in dumped_json,
        f"post-install verification failed on {qualname}: "
        f"model_dump_json() output does not contain "
        f'"{_SERIALIZATION_ALIAS}" as a key. Raw: {dumped_json!r}',
    )
    _require(
        f'"{_FIELD_NAME}":' not in dumped_json,
        f"post-install verification failed on {qualname}: "
        f"model_dump_json() still contains "
        f'"{_FIELD_NAME}": as a key. Raw: {dumped_json!r}',
    )

    # (d) model_dump(exclude_unset=True) — streaming path.
    dumped_unset = instance.model_dump(exclude_unset=True)
    _require(
        dumped_unset.get(_SERIALIZATION_ALIAS) == _PROBE_VALUE,
        f"post-install verification failed on {qualname}: "
        f"model_dump(exclude_unset=True)[{_SERIALIZATION_ALIAS!r}] is "
        f"{dumped_unset.get(_SERIALIZATION_ALIAS)!r}, expected the "
        f"probe value {_PROBE_VALUE!r}. Streaming-delta path would "
        f"not rename correctly.",
    )
    _require(
        _FIELD_NAME not in dumped_unset,
        f"post-install verification failed on {qualname}: "
        f"model_dump(exclude_unset=True) still contains the bare key "
        f"{_FIELD_NAME!r}. Full dump: {dumped_unset!r}",
    )

    # (e) model_dump_json(exclude_unset=True) — same streaming path,
    # JSON form. The one vLLM actually invokes per streaming chunk.
    dumped_unset_json = instance.model_dump_json(exclude_unset=True)
    _require(
        f'"{_SERIALIZATION_ALIAS}"' in dumped_unset_json
        and f'"{_FIELD_NAME}":' not in dumped_unset_json,
        f"post-install verification failed on {qualname}: "
        f"model_dump_json(exclude_unset=True) did not cleanly rename. "
        f"Raw: {dumped_unset_json!r}",
    )


for _qualname, _cls in _LEAF_TARGETS:
    _verify_leaf(_cls, _qualname)


# --------------------------------------------------------------------
# Phase 7: Real-nested-dump verification — the load-bearing addition
#          v2 makes over v1.
# --------------------------------------------------------------------
#
# Constructs an actual ChatCompletionResponse and an actual
# ChatCompletionStreamResponse (the two outermost classes vLLM dumps
# on the wire), serializes them, and asserts the wire JSON contains
# ``"reasoning_content":"<probe>"`` and does NOT contain
# ``"reasoning":``. THIS IS THE CHECK V1's VERIFIER MISSED.

def _verify_nested_non_streaming() -> None:
    """Verify a real ChatCompletionResponse dump emits reasoning_content."""
    response = ChatCompletionResponse(
        id="probe-id",
        model="probe-model",
        usage=UsageInfo(
            prompt_tokens=0,
            total_tokens=0,
            completion_tokens=0,
        ),
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    reasoning=_PROBE_VALUE,
                ),
            ),
        ],
    )
    wire_json = response.model_dump_json()
    _require(
        f'"{_SERIALIZATION_ALIAS}":"{_PROBE_VALUE}"' in wire_json,
        f"Real ChatCompletionResponse dump did not emit "
        f'"{_SERIALIZATION_ALIAS}":"{_PROBE_VALUE}". This is the '
        f"production wire path (api_router.py:70). Full wire: "
        f"{wire_json!r}",
    )
    _require(
        f'"{_FIELD_NAME}":' not in wire_json,
        f"Real ChatCompletionResponse dump still contains "
        f'"{_FIELD_NAME}": as a key. The bare key would defeat '
        f"OpenAI-spec clients. Full wire: {wire_json!r}",
    )

    # Also verify model_dump() (the dict form invoked by FastAPI's
    # JSONResponse content= path).
    wire_dict = response.model_dump()
    _msg_dict = wire_dict["choices"][0]["message"]
    _require(
        _msg_dict.get(_SERIALIZATION_ALIAS) == _PROBE_VALUE,
        f"Real ChatCompletionResponse.model_dump() did not emit "
        f"reasoning_content nested. Full dump: {wire_dict!r}",
    )
    _require(
        _FIELD_NAME not in _msg_dict,
        f"Real ChatCompletionResponse.model_dump() still contains "
        f"bare {_FIELD_NAME!r} on nested message. Full dump: "
        f"{wire_dict!r}",
    )


def _verify_nested_streaming() -> None:
    """Verify a real ChatCompletionStreamResponse dump emits reasoning_content."""
    stream_response = ChatCompletionStreamResponse(
        id="probe-id",
        model="probe-model",
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(reasoning=_PROBE_VALUE),
            ),
        ],
    )
    # serving.py:685/721/1208/1233 all use model_dump_json(exclude_unset=True).
    wire_json = stream_response.model_dump_json(exclude_unset=True)
    _require(
        f'"{_SERIALIZATION_ALIAS}":"{_PROBE_VALUE}"' in wire_json,
        f"Real ChatCompletionStreamResponse dump did not emit "
        f'"{_SERIALIZATION_ALIAS}":"{_PROBE_VALUE}". This is the '
        f"production streaming wire path "
        f"(serving.py:685/721/1208/1233). Full wire: {wire_json!r}",
    )
    _require(
        f'"{_FIELD_NAME}":' not in wire_json,
        f"Real ChatCompletionStreamResponse dump still contains "
        f'"{_FIELD_NAME}": as a key. The bare key would defeat '
        f"OpenAI-spec streaming clients. Full wire: {wire_json!r}",
    )


_verify_nested_non_streaming()
_verify_nested_streaming()


# --------------------------------------------------------------------
# Phase 8: Tag verification on every target — defense against
#          metaclass shenanigans hiding the install.
# --------------------------------------------------------------------

for _qualname, _cls in _ALL_TARGETS:
    _require(
        getattr(_cls, "__qwen36_egress_patch__", None) == _PATCH_TAG,
        f"static-tag verification failed on {_qualname}: "
        f"__qwen36_egress_patch__ not resolvable via getattr. A "
        f"concurrent monkey-patch or metaclass override is hiding our "
        f"assignment; refusing to proceed.",
    )
    _static = inspect.getattr_static(_cls, "__qwen36_egress_patch__", None)
    _require(
        _static == _PATCH_TAG,
        f"static-tag verification failed on {_qualname}: "
        f"inspect.getattr_static(__qwen36_egress_patch__) returned "
        f"{_static!r}, expected {_PATCH_TAG!r}. MRO or metaclass is "
        f"shadowing our assignment; refusing.",
    )


_logger.info(
    "[%s] applied: renamed %r -> %r on serialization for "
    "ChatMessage, DeltaMessage, ChatCompletionResponseChoice, "
    "ChatCompletionResponseStreamChoice, ChatCompletionResponse, and "
    "ChatCompletionStreamResponse (vLLM commit %s). Verified: "
    "model_dump(), model_dump_json(), model_dump(exclude_unset=True), "
    "model_dump_json(exclude_unset=True) on each leaf, AND nested "
    "dumps of real ChatCompletionResponse / "
    "ChatCompletionStreamResponse instances all emit "
    "%r and never the bare %r key.",
    _PATCH_TAG,
    _FIELD_NAME,
    _SERIALIZATION_ALIAS,
    _PINNED_VLLM_COMMIT,
    _SERIALIZATION_ALIAS,
    _FIELD_NAME,
)
