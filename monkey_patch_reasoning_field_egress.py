"""Egress patch renaming vLLM's non-standard ``reasoning`` field to the
OpenAI-spec ``reasoning_content`` on the wire.

Target: vLLM commit ``8936118134d0547fa1cc78adab2d03edd6d3dc48`` (README §3.2).
Egress half of the §6.12 wire-format mismatch; the ingest half is
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

OpenAI-conforming clients — our client among them — look under
``reasoning_content``, find nothing, and silently lose the reasoning
channel. Clients that instead special-case vLLM's ``reasoning`` are
coupled to a non-standard field that upstream may rename at any point.
Either way the wire format is wrong; this patch fixes it.

The internal Python attribute stays named ``reasoning``. vLLM's own
code (``vllm/entrypoints/openai/chat_completion/serving.py`` around
lines 1036-1037 and elsewhere) reads ``delta_message.reasoning`` as
an attribute; renaming the attribute would break that. We only
rename the JSON key.

Patch-discipline contract
-------------------------

This file is a patch, not a library. At import it:

1. Imports vLLM. Failure is a hard ImportError; we do not hide it.
2. Imports the two target classes by their pinned paths
   (``vllm.entrypoints.openai.chat_completion.protocol.ChatMessage``,
   ``vllm.entrypoints.openai.engine.protocol.DeltaMessage``).
3. Verifies each class is a class and a subclass of ``pydantic.BaseModel``.
4. Verifies each class declares a ``reasoning`` field whose annotation
   is exactly ``str | None``.
5. Verifies each class ALSO declares the structural neighbors we expect
   (``content``, ``role``, ``tool_calls``) — guard against a catastrophic
   schema change that renamed the class but kept a ``reasoning`` field
   on something unrelated.
6. Verifies neither class already declares ``reasoning_content`` — if
   upstream has added the correct field on their own, this patch would
   collide and we refuse to install.
7. Verifies no other current field on either class already carries a
   ``serialization_alias``. Our approach flips the class-wide
   ``serialize_by_alias`` flag; doing so while other fields carry
   aliases would silently activate them. Refusing here means a future
   upstream that introduces another alias gets re-audited, not
   silently mis-patched.
8. Mutates ``model_fields["reasoning"].serialization_alias``, sets
   ``model_config["serialize_by_alias"] = True``, clears the cached
   core schema / validator / serializer, sets
   ``__pydantic_complete__ = False``, and calls
   ``model_rebuild(force=True)`` to regenerate the serializer from
   scratch with the new alias picked up.
9. Constructs a fresh instance of each class and asserts, via
   ``model_dump()`` AND ``model_dump_json()``, that the on-wire key is
   exactly ``reasoning_content`` and the bare ``reasoning`` key does
   NOT appear anywhere in the serialized blob.
10. Asserts the Python attribute ``.reasoning`` still resolves on an
    instance and returns the value passed at construction.

Any step 1-10 failing raises :class:`EgressPatchRefusedError` and the
interpreter does not continue. There is **no** ``SystemExit(0)``,
``try/except Exception: pass``, or silent fallback on any install
path. A refused patch is the desired behavior — the deployment
treats refusal as a hard boot failure.

Why ``serialize_by_alias=True`` + schema rebuild, and not the
alternatives
-----------------------------------------------------------

We investigated four Pydantic v2 strategies and chose the one above
because it is the only one that works across **every** serialization
path vLLM actually uses (``model_dump()``, ``model_dump(exclude_none=
True)``, ``model_dump(exclude_unset=True)``, ``model_dump_json()``,
``model_dump_json(exclude_unset=True)``, and nested serialization
when the patched class appears as a field of another model):

* **Mutating** ``FieldInfo.serialization_alias`` alone, with or
  without ``model_rebuild(force=True)``, does NOT take effect in
  Pydantic 2.13. The alias metadata is mutated but the compiled
  ``__pydantic_serializer__`` was built before the mutation and is
  not regenerated. Verified empirically.
* Installing a ``@model_serializer(mode='wrap')`` post-hoc on the
  class and re-running ``model_rebuild`` does NOT pick up the
  decorator either, for the same reason: ``model_rebuild`` does not
  re-scan the class body for decorator-annotated methods.
* Wrapping ``model_dump`` / ``model_dump_json`` with Python-level
  monkey-patches would miss ``TypeAdapter`` and
  ``__pydantic_serializer__`` paths. vLLM's response serialization
  may funnel through either (FastAPI/Starlette JSON responses invoke
  the serializer directly on some paths), so this approach is
  demonstrably incomplete.
* Setting ``model_config["serialize_by_alias"] = True`` together
  with ``FieldInfo.serialization_alias = "reasoning_content"``,
  then *forcibly* deleting the cached ``__pydantic_core_schema__``,
  ``__pydantic_validator__``, and ``__pydantic_serializer__``,
  setting ``__pydantic_complete__ = False``, and calling
  ``model_rebuild(force=True)`` causes Pydantic to regenerate the
  core schema from scratch. The new schema honors both the per-field
  alias and the class-wide flag. Verified empirically across every
  dump variant listed above and for nested use (ChatMessage as a
  field of ChatCompletionResponseChoice).

The class-wide ``serialize_by_alias`` flag is potentially too broad
— if any OTHER field on the same class carries a ``serialization_
alias``, that alias would now also activate. Step 7 above guards
against this: if a future upstream adds a second alias on either
target class, this patch refuses to install and the operator re-audits.

Critical correctness invariants
-------------------------------

* **The Python attribute is NOT renamed.** vLLM's own code reads
  ``delta_message.reasoning`` (and the equivalent on ``ChatMessage``)
  as a plain attribute. This patch touches only the serialization
  alias and the class-wide ``serialize_by_alias`` flag. Step 10
  verifies the attribute still works after install.
* **No other field's behavior changes.** Step 7 refuses to proceed
  if any other field on either class has a pre-existing
  serialization alias; this makes flipping the class-wide flag safe.
* **Construction by the original field name still works.** Pydantic's
  ``serialization_alias`` (distinct from ``alias`` and
  ``validation_alias``) affects only egress, not ingress.
  Construction via ``ChatMessage(reasoning="...")`` continues to
  work; we assert this implicitly in step 10.
* **Nested serialization follows.** When a ChatMessage is nested
  inside a ChatCompletionResponseChoice (the actual response shape),
  the nested serializer is invoked by the outer schema's recursion
  into the ChatMessage core schema. Regenerating the inner class's
  core schema is sufficient — the outer class picks up the new inner
  schema on its own invocation. Verified empirically in the scratch
  experiment before this patch was committed.
"""

from __future__ import annotations

import inspect


_PINNED_VLLM_COMMIT: str = "8936118134d0547fa1cc78adab2d03edd6d3dc48"
_PATCH_TAG: str = "qwen36-agent-setup-reasoning-egress-v1"

# Source field name (Python attribute) and the OpenAI-spec alias we
# emit on the wire. The attribute name is NOT changed; only the
# serialized key.
_FIELD_NAME: str = "reasoning"
_SERIALIZATION_ALIAS: str = "reasoning_content"

# Structural-neighbor landmarks. Their presence on each target class
# confirms we are patching the class we think we are patching, not
# some unrelated class upstream happened to rename to ``ChatMessage``
# or ``DeltaMessage``. ``ChatMessage`` has ``tool_calls``;
# ``DeltaMessage`` has ``tool_calls`` too; both have ``role`` and
# ``content``.
_STRUCTURAL_NEIGHBORS: tuple[str, ...] = ("role", "content", "tool_calls")


class EgressPatchRefusedError(RuntimeError):
    """A precondition for the reasoning-field egress rename was violated.

    Raised at import time only. The patch either applies cleanly and
    correctly, or the process does not come up. A half-installed
    egress rename would leave the two response paths (full response
    vs streaming delta) emitting different field names, which is
    strictly worse than the pre-patch state where both are
    consistently wrong. We refuse to enter that state.
    """


def _require(condition: object, failure_message: str) -> None:
    if not condition:
        raise EgressPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {failure_message}"
        )


# --------------------------------------------------------------------
# Phase 1: Import vLLM and pydantic, locate the target classes.
# --------------------------------------------------------------------

import vllm  # noqa: F401  — availability landmark; must not be guarded

from pydantic import BaseModel

from vllm.logger import init_logger
from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage
from vllm.entrypoints.openai.engine.protocol import DeltaMessage

_logger = init_logger(__name__)


# The two target classes, in a tuple so phases 2-6 can iterate uniformly.
# Order matters only for log readability.
_TARGETS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("vllm.entrypoints.openai.chat_completion.protocol.ChatMessage", ChatMessage),
    ("vllm.entrypoints.openai.engine.protocol.DeltaMessage", DeltaMessage),
)


# --------------------------------------------------------------------
# Phase 2: Landmark that each target is a class, a BaseModel subclass,
# and carries the field we intend to rename plus the structural
# neighbors we expect.
# --------------------------------------------------------------------

for _qualname, _cls in _TARGETS:
    _require(
        inspect.isclass(_cls),
        f"{_qualname} is no longer a class (got {type(_cls).__name__!r}).",
    )
    _require(
        issubclass(_cls, BaseModel),
        f"{_qualname} is no longer a subclass of pydantic.BaseModel. "
        f"Upstream has restructured the response-protocol class hierarchy "
        f"and this patch's Pydantic v2 serialization-alias approach may "
        f"no longer apply.",
    )

    _fields = _cls.model_fields
    _require(
        _FIELD_NAME in _fields,
        f"{_qualname} no longer declares a {_FIELD_NAME!r} field. "
        f"Upstream has removed or renamed the reasoning channel; "
        f"re-audit the patch before bumping the pinned commit.",
    )

    # Verify type annotation is exactly ``str | None``. We compare the
    # runtime annotation object to both the PEP 604 union ``str | None``
    # and the typing.Optional form to be robust to either.
    _ann = _fields[_FIELD_NAME].annotation
    _expected_union = str | None
    _require(
        _ann == _expected_union,
        f"{_qualname}.{_FIELD_NAME} annotation changed: "
        f"expected {_expected_union!r}, got {_ann!r}. The wire-type of "
        f"the reasoning channel has changed; re-audit before patching.",
    )

    for _neighbor in _STRUCTURAL_NEIGHBORS:
        _require(
            _neighbor in _fields,
            f"{_qualname} no longer declares the expected structural "
            f"neighbor {_neighbor!r}. This is a catastrophic schema "
            f"change; refusing to patch a class whose shape we no "
            f"longer recognize.",
        )

    _require(
        _SERIALIZATION_ALIAS not in _fields,
        f"{_qualname} already declares a {_SERIALIZATION_ALIAS!r} field. "
        f"Upstream appears to have added the OpenAI-spec field on its "
        f"own; this patch would collide with it. Remove this patch "
        f"from the boot sequence and verify upstream's implementation "
        f"is correct.",
    )


# --------------------------------------------------------------------
# Phase 3: Audit existing serialization_alias declarations on every
# field of every target class. Flipping ``serialize_by_alias = True``
# at the class level activates ALL aliases on that class; if any
# other field already carries one, we must refuse so the operator
# re-audits before changing wire behavior of an unrelated field.
# --------------------------------------------------------------------

for _qualname, _cls in _TARGETS:
    for _fname, _finfo in _cls.model_fields.items():
        if _fname == _FIELD_NAME:
            # Expected to be None right now — we are about to set it.
            _existing = getattr(_finfo, "serialization_alias", None)
            _require(
                _existing is None,
                f"{_qualname}.{_fname} already carries "
                f"serialization_alias={_existing!r}. Expected None "
                f"(unaliased) at patch time; upstream may have already "
                f"attempted this rename. Refusing to patch.",
            )
            continue
        _other_alias = getattr(_finfo, "serialization_alias", None)
        _require(
            _other_alias is None,
            f"{_qualname}.{_fname} carries serialization_alias="
            f"{_other_alias!r}. This patch flips the class-wide "
            f"``serialize_by_alias`` flag, which would activate that "
            f"alias on the wire. Refusing to silently change a second "
            f"field's serialization; re-audit before proceeding.",
        )


# --------------------------------------------------------------------
# Phase 4: Apply the alias and regenerate the core schema on each
# target class.
# --------------------------------------------------------------------

# Attributes Pydantic caches after class creation and that are re-
# populated during ``model_rebuild(force=True)`` only if they are
# absent or the class is marked incomplete. We delete them explicitly
# so the regenerated schema picks up the mutated FieldInfo instead of
# reusing the stale compiled serializer.
_CACHED_SCHEMA_ATTRS: tuple[str, ...] = (
    "__pydantic_core_schema__",
    "__pydantic_validator__",
    "__pydantic_serializer__",
)


def _install_serialization_alias(cls: type[BaseModel], qualname: str) -> None:
    """Install the ``reasoning`` -> ``reasoning_content`` serialization
    alias on ``cls`` and regenerate Pydantic's compiled schema so the
    new alias takes effect across every serialization path.

    This function intentionally does NOT live inside a try/except: any
    failure here must propagate so the outer import-time gate raises
    ``EgressPatchRefusedError`` (via the post-install verification).
    A partially-installed alias — e.g. mutated FieldInfo but failed
    rebuild — would leave the class in an inconsistent state, but
    the very next line of the regenerate step would surface that
    inconsistency via the post-install verification and the operator
    would get a clear failure.
    """
    # 1. Turn on the class-wide alias serialization flag. This is what
    #    makes model_dump() honor serialization_alias without the
    #    caller having to pass by_alias=True (vLLM callers do not).
    cls.model_config["serialize_by_alias"] = True

    # 2. Mutate the FieldInfo on the ``reasoning`` field.
    field_info = cls.model_fields[_FIELD_NAME]
    field_info.serialization_alias = _SERIALIZATION_ALIAS

    # 3. Drop the cached compiled schema / validator / serializer.
    #    model_rebuild(force=True) alone is not sufficient in Pydantic
    #    2.13 to regenerate these from the mutated FieldInfo; we have
    #    to explicitly remove them first.
    for attr in _CACHED_SCHEMA_ATTRS:
        if attr in cls.__dict__:
            delattr(cls, attr)

    # 4. Mark the class as incomplete so model_rebuild(force=True)
    #    takes the full regeneration path.
    cls.__pydantic_complete__ = False

    # 5. Regenerate.
    cls.model_rebuild(force=True)

    # Tag the class so other patches / introspection / operators can
    # confirm the egress rename was applied. The attribute name is
    # dunder-prefixed and qwen36-scoped to minimize collision risk
    # with any unrelated Pydantic-internal or vLLM-internal attribute.
    setattr(cls, "__qwen36_egress_patch__", _PATCH_TAG)


for _qualname, _cls in _TARGETS:
    _install_serialization_alias(_cls, _qualname)


# --------------------------------------------------------------------
# Phase 5: Post-install verification. Construct a fresh instance of
# each class, dump via every path that matters, and assert the wire
# shape is correct AND the internal Python attribute is intact.
# --------------------------------------------------------------------

# Sentinel value chosen to be distinguishable from any legitimate
# content and short enough to substring-search for without ambiguity.
_PROBE_VALUE: str = "__qwen36_egress_probe__"


def _verify_class(cls: type[BaseModel], qualname: str) -> None:
    """Assert that ``cls`` serializes ``reasoning`` as
    ``reasoning_content`` on every relevant dump path and that the
    Python attribute survives intact.

    Failures here raise ``EgressPatchRefusedError`` via ``_require``.
    There is no fallback path; an assertion failure means the patch
    did not take effect and boot must fail.
    """
    # Construct. ChatMessage requires ``role``; DeltaMessage does
    # not. Pass ``role`` when the class demands it; pass only
    # ``reasoning`` otherwise to exercise the minimal-object path.
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

    # (b) model_dump() emits the alias, and no bare "reasoning" key.
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
        f"take effect.",
    )
    _require(
        _FIELD_NAME not in dumped,
        f"post-install verification failed on {qualname}: "
        f"model_dump() still contains the bare key {_FIELD_NAME!r} "
        f"(value={dumped.get(_FIELD_NAME)!r}). The alias was applied "
        f"but the unaliased key is still being emitted alongside it — "
        f"this would double-emit the reasoning channel on the wire. "
        f"Full dump: {dumped!r}",
    )

    # (c) model_dump_json() emits the alias, and no bare "reasoning":
    #     key. We substring-search the JSON text directly rather than
    #     re-parse it, so we catch any case where the bare key would
    #     appear in the raw bytes (including in nested objects if any
    #     future field type embeds a ChatMessage).
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

    # (d) model_dump(exclude_unset=True) — vLLM's streaming path uses
    #     this (serving.py:678, 714, 1193). Must also emit the alias.
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
    #     JSON form. The one vLLM actually invokes per streaming chunk.
    dumped_unset_json = instance.model_dump_json(exclude_unset=True)
    _require(
        f'"{_SERIALIZATION_ALIAS}"' in dumped_unset_json
        and f'"{_FIELD_NAME}":' not in dumped_unset_json,
        f"post-install verification failed on {qualname}: "
        f"model_dump_json(exclude_unset=True) did not cleanly rename. "
        f"Raw: {dumped_unset_json!r}",
    )


for _qualname, _cls in _TARGETS:
    _verify_class(_cls, _qualname)


# --------------------------------------------------------------------
# Phase 6: Install tags and final landmark log line.
# --------------------------------------------------------------------

# Second-order verification that each target class carries the tag we
# attached in Phase 4, resolvable via both normal getattr and
# inspect.getattr_static (metaclass shenanigans would otherwise hide
# the assignment — same defense used in patch 5).
for _qualname, _cls in _TARGETS:
    _require(
        getattr(_cls, "__qwen36_egress_patch__", None) == _PATCH_TAG,
        f"static-tag verification failed on {_qualname}: "
        f"__qwen36_egress_patch__ not resolvable via getattr. "
        f"A concurrent monkey-patch or metaclass override is hiding "
        f"our assignment; refusing to proceed.",
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
    "%s and %s (vLLM commit %s). Verified: model_dump(), "
    "model_dump_json(), model_dump(exclude_unset=True), "
    "model_dump_json(exclude_unset=True) all emit %r; bare %r key "
    "absent from every dump; internal attribute .%s round-trips "
    "intact.",
    _PATCH_TAG,
    _FIELD_NAME,
    _SERIALIZATION_ALIAS,
    _TARGETS[0][0],
    _TARGETS[1][0],
    _PINNED_VLLM_COMMIT,
    _SERIALIZATION_ALIAS,
    _FIELD_NAME,
    _FIELD_NAME,
)
