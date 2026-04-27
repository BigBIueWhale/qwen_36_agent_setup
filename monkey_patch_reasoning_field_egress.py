"""Strict, fail-loud egress patch renaming vLLM's non-standard
``reasoning`` field to the OpenAI-spec ``reasoning_content`` on the wire.

Why this patch must exist
-------------------------

vLLM emits ``"reasoning":`` on the wire (since commit ``c5113f60f2``
deliberately removed ``reasoning_content``). Qwen-Agent's OAI client at
``Qwen-Agent/qwen_agent/llm/oai.py:111-112,126-127,169`` strict-checks
``reasoning_content`` with **no fallback**; without the alias every
multi-turn agent loop loses prior reasoning on egress and degrades
after 2-3 turns. Pydantic v2's compiled core schemas embed nested
schemas by snapshot at build time, so a leaves-only rebuild leaks
``reasoning`` through wrappers — every class on the dump chain must
rebuild under ``serialize_by_alias=True`` for the leaf alias to reach
the wire (negative control proven in tests Section 3b).

Target: vLLM commit ``8cd174fa358326d5cc4195446be2ebcd65c481ce``.
**Removal trigger**: vLLM ships native ``reasoning_content`` on Chat
Completions. Companion: ``monkey_patch_reasoning_field_ingest.py``.
"""

from __future__ import annotations

import inspect
from typing import Any

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
from vllm.entrypoints.openai.engine.protocol import DeltaMessage, UsageInfo

_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-reasoning-egress-v3"
_FIELD_NAME: str = "reasoning"
_ALIAS: str = "reasoning_content"
_PROBE: str = "__qwen36_egress_probe__"

_logger = init_logger(__name__)


class EgressPatchRefusedError(RuntimeError):
    """Precondition for the egress rename was violated."""


def _require(cond: object, msg: str) -> None:
    if not cond:
        raise EgressPatchRefusedError(f"[{_PATCH_TAG}] refusing to patch: {msg}")


# Targets in REBUILD ORDER — leaves first so wrappers' rebuild snapshots
# the new leaf schemas. Tuples: (qualname, cls, is_leaf, nested-link).
_TARGETS: tuple[tuple[str, type[BaseModel], bool, tuple[str, Any] | None], ...] = (
    ("ChatMessage", ChatMessage, True, None),
    ("DeltaMessage", DeltaMessage, True, None),
    ("ChatCompletionResponseChoice", ChatCompletionResponseChoice, False,
     ("message", ChatMessage)),
    ("ChatCompletionResponseStreamChoice", ChatCompletionResponseStreamChoice, False,
     ("delta", DeltaMessage)),
    ("ChatCompletionResponse", ChatCompletionResponse, False,
     ("choices", list[ChatCompletionResponseChoice])),
    ("ChatCompletionStreamResponse", ChatCompletionStreamResponse, False,
     ("choices", list[ChatCompletionResponseStreamChoice])),
)


# Phase 1: Per-target structural + collision verification.
for _qn, _cls, _is_leaf, _nested in _TARGETS:
    _require(
        inspect.isclass(_cls) and issubclass(_cls, BaseModel),
        f"{_qn} is no longer a pydantic.BaseModel.",
    )
    _require(
        _ALIAS not in _cls.model_fields,
        f"{_qn} already declares {_ALIAS!r}; upstream collision — refusing.",
    )
    _require(
        _cls.model_config.get("serialize_by_alias", False) is False,
        f"{_qn}.model_config['serialize_by_alias'] already set; refusing.",
    )
    for _fname, _finfo in _cls.model_fields.items():
        _require(
            getattr(_finfo, "serialization_alias", None) is None,
            f"{_qn}.{_fname} already carries a serialization_alias.",
        )
    if _is_leaf:
        _require(
            _FIELD_NAME in _cls.model_fields,
            f"{_qn}.{_FIELD_NAME} field is gone — upstream may have "
            f"removed or renamed the reasoning channel; re-audit.",
        )
        _ann = _cls.model_fields[_FIELD_NAME].annotation
        _require(_ann == (str | None), f"{_qn}.{_FIELD_NAME} annotation drifted: {_ann!r}.")
    else:
        assert _nested is not None
        _nname, _expected = _nested
        _require(_nname in _cls.model_fields, f"{_qn}.{_nname} field missing.")
        _ann = _cls.model_fields[_nname].annotation
        _require(
            _ann == _expected,
            f"{_qn}.{_nname} annotation drifted: expected {_expected!r}, got {_ann!r}.",
        )


# Phase 2: Install. Leaves first (alias + flag + rebuild), then wrappers
# in dependency order (flag + rebuild only).
for _qn, _cls, _is_leaf, _ in _TARGETS:
    if _is_leaf:
        _cls.model_fields[_FIELD_NAME].serialization_alias = _ALIAS
    _cls.model_config["serialize_by_alias"] = True
    for _attr in (
        "__pydantic_core_schema__",
        "__pydantic_validator__",
        "__pydantic_serializer__",
    ):
        if _attr in _cls.__dict__:
            delattr(_cls, _attr)
    _cls.__pydantic_complete__ = False
    _cls.model_rebuild(force=True)
    setattr(_cls, "__qwen36_egress_patch__", _PATCH_TAG)


# Phase 3: End-to-end nested-dump verification — load-bearing.
# A tag-only check passes even when the wrapper rebuild silently failed.
def _verify_wire(label: str, wire: str) -> None:
    _require(
        f'"{_ALIAS}":"{_PROBE}"' in wire,
        f"{label}: missing reasoning_content. Wire: {wire!r}",
    )
    _require(
        f'"{_FIELD_NAME}":' not in wire,
        f"{label}: bare reasoning leaks. Wire: {wire!r}",
    )


# api_router.py:70 uses .model_dump_json(); api_router.py:102 uses
# .model_dump() on the batch path. Both surfaces must rename — verify both.
_response = ChatCompletionResponse(
    id="p", model="p",
    usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0),
    choices=[ChatCompletionResponseChoice(
        index=0, message=ChatMessage(role="assistant", reasoning=_PROBE),
    )],
)
_verify_wire("response.model_dump_json", _response.model_dump_json())

_response_dump = _response.model_dump()
_msg_dict = _response_dump["choices"][0]["message"]
_require(
    _msg_dict.get(_ALIAS) == _PROBE,
    f"response.model_dump() did not emit nested {_ALIAS!r} on the "
    f"message dict (api_router.py:102 batch path would leak bare "
    f"{_FIELD_NAME!r}). Got: {_msg_dict!r}",
)
_require(
    _FIELD_NAME not in _msg_dict,
    f"response.model_dump() leaks bare {_FIELD_NAME!r} key on the "
    f"nested message dict. Got: {_msg_dict!r}",
)

# serving.py:685/721/1208/1233 use model_dump_json(exclude_unset=True).
_stream = ChatCompletionStreamResponse(
    id="p", model="p",
    choices=[ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(reasoning=_PROBE),
    )],
)
_verify_wire(
    "stream.model_dump_json(exclude_unset=True)",
    _stream.model_dump_json(exclude_unset=True),
)


# Phase 4: Tag verification on every target — defense against metaclass shims.
for _qn, _cls, _, _ in _TARGETS:
    _require(
        getattr(_cls, "__qwen36_egress_patch__", None) == _PATCH_TAG
        and inspect.getattr_static(_cls, "__qwen36_egress_patch__", None) == _PATCH_TAG,
        f"tag verification failed on {_qn}.",
    )


_logger.info(
    "[%s] applied: %r -> %r alias on 6 wire classes (vLLM commit %s).",
    _PATCH_TAG,
    _FIELD_NAME,
    _ALIAS,
    _PINNED_VLLM_COMMIT,
)
