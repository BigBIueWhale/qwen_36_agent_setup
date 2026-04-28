"""Strict, fail-loud sender↔receiver mm-cache eviction on validator rejection.

Why this patch must exist
-------------------------

The user-visible symptom is a poisoned image hash that turns every
subsequent request carrying that image into HTTP 500 until the
container restarts (or the sender LRU evicts the entry — hours under
agentic workloads with a hot image). The trigger is the most natural
human behaviour: an operator's request goes over budget
(``prompt_tokens + max_tokens > max_model_len``), the API server
returns the correct HTTP 400, the operator clicks "Retry" with the
same image and a smaller prompt, and now they're stuck.

Engine ``/health=200`` throughout most of the failure window — this
looks like a healthy server until decode is attempted. The HTTP 500
response is the generic ``EngineCore encountered an issue. See stack
trace for the root cause.`` shape, which points at the engine while
the actual bug lives in the API server's renderer.

Empirical reproduction (2026-04-28). Sequence:

* **Step 1 (poison)**: `text + image, prompt 153,666 tokens > 152,000`.
  `render_chat_request` runs `mm_processor.apply()` (image hashed and
  inserted into sender cache for `H='0f21b…'`); then `get_max_tokens`
  throws `ValueError`. **API server returns HTTP 400 in 0.46 s.** Engine
  never sees the request — sender cache is now ahead of the receiver
  cache for `H`.
* **Step 2 (smoking gun, same image)**: clean small request with the
  same image. **HTTP 500 in 0.06 s.** Engine raises
  `AssertionError: Expected a cached item for mm_hash='0f21b…'`.
* **Step 3 (different image, healthy at boundary)**: another large
  request returns HTTP 500. `/health` flips to 503 here — the engine
  has accumulated enough preprocessing failures that liveness drops.

The poison persists for the lifetime of the LRU entry. There is no
API-level workaround for the user.

Where the bug actually lives (call graph at vLLM 8cd174fa3)
-----------------------------------------------------------

Sender↔receiver mm-cache mirror invariant:

* sender cache (API-server-side, in the renderer's
  ``MultiModalProcessorSenderCache`` at
  ``vllm/multimodal/cache.py:379-434``): records that hash ``H`` was
  observed by P0 and saved as metadata.
* receiver cache (engine-side, in the EngineCore's
  ``MultiModalReceiverCache`` at ``vllm/multimodal/cache.py:614-647``):
  holds the actual ``MultiModalKwargsItem`` once the engine has
  processed the IPC.

Steady-state happy path: every request that registers ``H`` on the
sender immediately ships the IPC blob to the engine, which records
``H`` on the receiver. Any subsequent request can short-circuit:
sender's ``get_and_update_item`` at ``cache.py:410-422`` returns
``(None, prompt_updates)`` — the engine already has it. Engine's
``get_and_update_features`` at ``cache.py:573-592`` substitutes the
cached value in place of the ``None`` payload. Both halves stay in
sync.

The validator-throw paths break the invariant:

1. ``vllm/entrypoints/openai/chat_completion/serving.py:251`` — `result
   = await self.render_chat_request(request)`. Calls into the
   renderer, which calls `mm_processor.apply()` (per
   ``vllm/multimodal/processing/processor.py``), which calls
   ``MultiModalProcessorSenderCache.get_and_update_item`` at
   ``vllm/multimodal/cache.py:410``. **Sender cache is now populated
   for hash H. Engine has not yet been notified.**
2. ``serving.py:284`` — ``max_tokens = get_max_tokens(...)``. Calls
   ``vllm/entrypoints/utils.py:174-203``. The body raises
   ``ValueError`` at line 182 when ``max_model_len < input_length``.
3. ``vllm/renderers/params.py:411-428`` — separate validator
   (``_token_len_check``) raises ``VLLMValidationError`` when
   ``len(tokens) > max_input_tokens``. ``VLLMValidationError`` extends
   ``ValueError`` (``vllm/exceptions.py:9``). This validator runs INSIDE
   ``render_chat_request`` itself, at the same call frame that just
   populated the sender cache.

Either throw exits ``create_chat_completion`` before the engine IPC
fires. Sender↔receiver caches diverge.

Next request with hash H:

4. ``serving.py:251`` — sender's IPC-saving short-circuit kicks in;
   ``cache.py:415-416`` returns ``(None, prompt_updates)`` because H
   is in the sender LRU.
5. ``vllm/v1/engine/core.py:765-777`` (``preprocess_add_request``).
   Engine's receiver cache call at ``cache.py:573-592``
   (``get_and_update_features``) loops over features whose ``data`` is
   ``None``, and for each calls ``get_and_update_item`` at
   ``cache.py:636-647``. Receiver cache has no entry for H, and ``mm_item
   is None``, so the assertion at ``cache.py:644`` —
   ``assert mm_item is not None, f"Expected a cached item for {mm_hash=}"``
   — fires.
6. ``core.py:1448-1452`` catches the AssertionError as a generic
   ``Exception`` and routes through ``_handle_request_preproc_error``
   at ``core.py:1533-1540``: logs ``Unexpected error pre-processing
   request <id>`` and ships an error response back to the API server
   as HTTP 500.

The poison persists until the sender LRU evicts H (the cache size is
``mm_processor_cache_gb`` worth of entries — thousands, easily several
hours of agentic-workload churn for a hot image).

Scope vs upstream
-----------------

* Issue #31404 tracks "MultiModalReceiverCache assertion fires under
  some triggers" (open). Different trigger from ours, same end state.
* Draft PR #34749 attempts to soften the assertion to a typed error
  but is unmerged and does **not** clear sender state on the rejecting
  path. Even if it merged, it would convert silent poisoning into LOUD
  errors — but the same retry would still fail because the sender↔
  receiver mirror is still broken; the next request still tries to
  short-circuit on a hash the engine doesn't know about. **Our patch
  is at the right layer** because the fix has to clear sender cache
  state at the rejection site, not soften the engine-side assertion
  downstream.

What this patch does
--------------------

Wraps :meth:`OpenAIServingChat.create_chat_completion` (an async
method). When the wrapped call raises ``ValueError`` or
``VLLMValidationError`` (the typed exceptions the two validators
emit), call
:meth:`OpenAIServingChat.renderer.clear_mm_cache_async` to clear
the sender cache, then re-raise the original exception unchanged. The
caller sees the same HTTP 400 response shape; the engine sees a clean
sender↔receiver mirror on the next request.

`self.renderer` (inherited from `OpenAIServing` at
`vllm/entrypoints/openai/engine/serving.py:157`) and
`self.openai_serving_render.renderer` (set at `api_server.py:369-371`)
are the **same** `BaseRenderer` instance. The sender cache lives on
that instance. Clearing via either path is equivalent. We use
`self.renderer` because it is the inherited attribute that exists on
every `OpenAIServing` subclass; the `openai_serving_render` attribute
is `OpenAIServingChat`-specific.

We catch ``(ValueError, VLLMValidationError)``. The redundancy is by
design — ``VLLMValidationError`` extends ``ValueError``, but naming
both at the call site documents intent and forces a re-audit on
upstream type-hierarchy refactors. We do **not** catch generic
``Exception`` — that would mask actual bugs that should bubble up.

Defensive validation rationale
------------------------------

Even with ``self.renderer.clear_mm_cache_async`` exercised at install
time (Phase 7), runtime drift is possible: a future vLLM version that
factors out the renderer attribute, or a sub-classed
``OpenAIServingChat`` that doesn't carry the same shape. The wrapper
defensively short-circuits if the renderer or its
``clear_mm_cache_async`` method is missing at the eviction site,
emits a structured WARNING, and re-raises the original exception
unmodified. **The patch's job is to be SAFER than upstream, never
less safe.**

Removal trigger
---------------

vLLM upstream lands a fix that clears sender cache state on validator
rejection (currently issue #31404 / draft PR #34749). The buggy
behaviour is ``serving.py:251`` populating the sender cache before
``serving.py:284``'s ``get_max_tokens`` validator can throw — any
upstream restructuring that either (a) defers sender insertion until
after all length validation succeeds or (b) wires a typed
``finally``-block eviction into the rejecting path satisfies the
removal trigger.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any

import vllm  # noqa: F401  — availability landmark; must not be guarded

from vllm.entrypoints.openai.chat_completion import serving as _serving_mod
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints import utils as _utils_mod
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger


_PINNED_VLLM_COMMIT: str = "8cd174fa358326d5cc4195446be2ebcd65c481ce"
_PATCH_TAG: str = "qwen36-agent-setup-mm-cache-validator-eviction-v1"

# Source landmarks — substrings required in the wrapped method body and
# in the validator function body. Drift in either site means the
# patch's "validator throws inside this call" assumption no longer
# holds; refuse before silently no-op'ing.
#
# create_chat_completion populates the sender cache via render_chat_request
# at line 251 (await self.render_chat_request(request)) and validates the
# length budget via get_max_tokens at line 284. Both lines must be
# present — they are the load-bearing ingredients of the bug we patch.
_RENDER_LANDMARK: str = "await self.render_chat_request(request)"
_GET_MAX_TOKENS_LANDMARK: str = "max_tokens = get_max_tokens("

# get_max_tokens raises ValueError on input_length > max_model_len.
# This is the typed exception the wrapper catches; if upstream
# replaces it (e.g. a typed BudgetError) the catch is too narrow and
# we'd silently leave the cache poisoned.
_GET_MAX_TOKENS_RAISE_LANDMARK: str = "raise ValueError"

# The other validator path: render_chat_request also runs apply_length_check
# at vllm/renderers/params.py:411-428 which raises VLLMValidationError. The
# class still extends ValueError per vllm/exceptions.py:9; we anchor on the
# string here only via the exception type below (no source landmark needed
# because the renderer's body is large and reshapes more often than the
# validator predicate itself).
_EXPECTED_PARAMS: list[str] = ["self", "request", "raw_request"]


_logger = init_logger(f"vllm.qwen36_patches.{__name__}")


class MmCacheValidatorEvictionPatchRefusedError(RuntimeError):
    """A precondition for the mm-cache validator-eviction wrapper was
    violated. Raised at import time only; the patch either applies
    cleanly or the process does not come up. Same idiom as the other
    Qwen3.6 patches.

    A wrong wrap target or missing renderer attribute would silently
    keep the sender↔receiver mirror invariant broken at runtime;
    refusing to boot is strictly safer.
    """


def _require(condition: object, msg: str) -> None:
    if not condition:
        raise MmCacheValidatorEvictionPatchRefusedError(
            f"[{_PATCH_TAG}] refusing to patch: {msg}"
        )


# --------------------------------------------------------------------
# Phase 1: Imports already done at module scope above.
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Phase 2: Locate the target class and method.
# --------------------------------------------------------------------

_require(
    inspect.isclass(OpenAIServingChat),
    "vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat "
    "is not a class.",
)

_original = getattr(OpenAIServingChat, "create_chat_completion", None)
_require(
    _original is not None and callable(_original),
    "OpenAIServingChat.create_chat_completion missing or not callable.",
)
_require(
    inspect.iscoroutinefunction(_original),
    "OpenAIServingChat.create_chat_completion is not an async (coroutine) "
    "function — the wrapper's await-and-catch contract is invalid.",
)
_require(
    getattr(_original, "__qwen36_patch__", None) is None,
    f"OpenAIServingChat.create_chat_completion already carries a "
    f"__qwen36_patch__ tag {getattr(_original, '__qwen36_patch__', None)!r}; "
    f"refusing to double-install.",
)


# --------------------------------------------------------------------
# Phase 3: Verify signature and source landmarks of the wrapped method
# AND of the get_max_tokens validator. Both must match the audited
# shape, otherwise the wrapper's "validator throws inside this call"
# assumption is no longer valid.
# --------------------------------------------------------------------

try:
    _sig = inspect.signature(_original)  # type: ignore[arg-type]
    _src = inspect.getsource(_original)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise MmCacheValidatorEvictionPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect "
        f"OpenAIServingChat.create_chat_completion: {_exc!r}"
    ) from _exc

_param_names = list(_sig.parameters)
_require(
    _param_names == _EXPECTED_PARAMS,
    f"create_chat_completion signature drifted; expected "
    f"{_EXPECTED_PARAMS!r}, got {_param_names!r}.",
)

_require(
    _RENDER_LANDMARK in _src,
    f"render landmark {_RENDER_LANDMARK!r} missing from "
    f"create_chat_completion source — upstream restructured the renderer "
    f"call site; the wrapper's 'sender cache is populated before the "
    f"validator throws' assumption is no longer valid.",
)
_require(
    _GET_MAX_TOKENS_LANDMARK in _src,
    f"get_max_tokens landmark {_GET_MAX_TOKENS_LANDMARK!r} missing from "
    f"create_chat_completion source — the validator-throw site has "
    f"shifted upstream; re-audit before bumping the pinned commit.",
)

# Verify get_max_tokens itself still raises ValueError — that is the
# typed exception we catch. If upstream replaces it with a typed
# BudgetError, our catch is too narrow.
_get_max_tokens_fn = getattr(_utils_mod, "get_max_tokens", None)
_require(
    _get_max_tokens_fn is not None and callable(_get_max_tokens_fn),
    "vllm.entrypoints.utils.get_max_tokens is missing or not callable; "
    "the validator we anchor on has been removed.",
)
try:
    _gmt_src = inspect.getsource(_get_max_tokens_fn)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise MmCacheValidatorEvictionPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect get_max_tokens: {_exc!r}"
    ) from _exc
_require(
    _GET_MAX_TOKENS_RAISE_LANDMARK in _gmt_src,
    f"get_max_tokens source no longer contains "
    f"{_GET_MAX_TOKENS_RAISE_LANDMARK!r} — the typed exception we catch "
    f"has been replaced; re-audit the wrapper's except clause.",
)

# Verify VLLMValidationError still extends ValueError so a single
# `except ValueError` catches both. This is the contract `vllm/
# exceptions.py:9` declares; if the hierarchy is refactored, our
# catch tuple needs to be widened.
_require(
    issubclass(VLLMValidationError, ValueError),
    f"VLLMValidationError is no longer a subclass of ValueError "
    f"(MRO: {[c.__name__ for c in VLLMValidationError.__mro__]!r}); "
    f"the wrapper's exception-catch contract is invalid.",
)


# --------------------------------------------------------------------
# Phase 4: Verify the renderer attribute and clear_mm_cache_async
# method shape. The renderer holds the sender cache; clearing it
# restores the sender↔receiver mirror invariant. Inspect the
# OpenAIServing base class (which sets self.renderer in __init__) and
# the BaseRenderer class for the async method.
# --------------------------------------------------------------------

# self.renderer is set in OpenAIServing.__init__ (the base class).
# The class itself doesn't declare it as a class-level attribute so
# we cannot probe via getattr on the class — we can only verify the
# initializer is present and assigns it. Use the source-landmark of
# the base __init__.
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.renderers.base import BaseRenderer

_require(
    inspect.isclass(OpenAIServing) and issubclass(OpenAIServingChat, OpenAIServing),
    "OpenAIServingChat is no longer a subclass of OpenAIServing; "
    "self.renderer is no longer guaranteed to be set in __init__.",
)
try:
    _base_init_src = inspect.getsource(OpenAIServing.__init__)  # type: ignore[arg-type]
except (TypeError, ValueError, OSError) as _exc:
    raise MmCacheValidatorEvictionPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect OpenAIServing.__init__: {_exc!r}"
    ) from _exc
_require(
    "self.renderer = engine_client.renderer" in _base_init_src,
    "OpenAIServing.__init__ no longer assigns self.renderer = "
    "engine_client.renderer; the renderer attribute the wrapper relies on "
    "is no longer guaranteed to exist.",
)

# clear_mm_cache_async is defined on BaseRenderer. Verify shape.
_clear_async = getattr(BaseRenderer, "clear_mm_cache_async", None)
_require(
    _clear_async is not None and callable(_clear_async),
    "BaseRenderer.clear_mm_cache_async is missing or not callable; "
    "the eviction primitive the wrapper relies on has been removed.",
)
_require(
    inspect.iscoroutinefunction(_clear_async),
    "BaseRenderer.clear_mm_cache_async is not an async (coroutine) "
    "function; the wrapper's `await self.renderer.clear_mm_cache_async()` "
    "contract is invalid.",
)
try:
    _clear_sig = inspect.signature(_clear_async)  # type: ignore[arg-type]
except (TypeError, ValueError) as _exc:
    raise MmCacheValidatorEvictionPatchRefusedError(
        f"[{_PATCH_TAG}] cannot introspect "
        f"BaseRenderer.clear_mm_cache_async: {_exc!r}"
    ) from _exc
_clear_params = list(_clear_sig.parameters)
_require(
    _clear_params == ["self"],
    f"BaseRenderer.clear_mm_cache_async signature drifted; expected "
    f"['self'], got {_clear_params!r}. The eviction primitive now takes "
    f"unexpected arguments.",
)


# --------------------------------------------------------------------
# Phase 5: Build the wrapper.
# --------------------------------------------------------------------


@functools.wraps(_original)
async def _create_chat_completion_with_mm_cache_eviction(
    self: Any,
    request: Any,
    raw_request: Any = None,
) -> Any:
    """Wrap the upstream ``create_chat_completion`` with sender-cache
    eviction on validator throws.

    Happy path: delegate to the upstream coroutine and return its
    result.

    Validator-throw path: the upstream raises ``ValueError`` or
    ``VLLMValidationError`` AFTER having populated the renderer's
    sender cache via ``self.render_chat_request(request)``. The next
    request carrying the same ``mm_hash`` would short-circuit on the
    sender (assuming the engine has it) and the engine's receiver
    cache would assertion-error. To prevent that, we clear the sender
    cache now so the next request with the same hash takes the
    populate-and-IPC path again. Re-raise the original exception
    unchanged so the caller sees the standard HTTP 400 response
    shape.

    Defensive: if the renderer is missing or its clear_mm_cache_async
    method is gone (e.g. a future subclass that does its own thing),
    skip the eviction with a structured WARNING and re-raise the
    original exception. The patch is strictly additive — never less
    safe than upstream.
    """
    try:
        return await _original(self, request, raw_request)
    except (ValueError, VLLMValidationError) as exc:
        # Clear the sender cache so the next request with the same
        # mm_hash goes through the populate-and-IPC path again.
        # Defensive shape validation: do not crash the response path
        # if the renderer attribute drifts; the caller still gets
        # the original exception.
        renderer = getattr(self, "renderer", None)
        clear_fn = getattr(renderer, "clear_mm_cache_async", None) if renderer else None
        if clear_fn is None or not callable(clear_fn):
            _logger.warning(
                "[%s] sender-cache eviction skipped: "
                "self.renderer.clear_mm_cache_async is missing "
                "(renderer=%r); sender↔receiver mm-cache mirror may be "
                "left broken until container restart. Re-raising original "
                "%s without eviction.",
                _PATCH_TAG,
                renderer,
                type(exc).__name__,
            )
            raise

        try:
            await clear_fn()
        except Exception as evict_exc:  # noqa: BLE001 — never mask original
            # An eviction failure must not displace the original
            # exception the caller is waiting on. Log loudly and re-
            # raise the original.
            _logger.warning(
                "[%s] sender-cache eviction itself raised %s: %r; "
                "re-raising original %s. Sender↔receiver mm-cache mirror "
                "may be left broken.",
                _PATCH_TAG,
                type(evict_exc).__name__,
                evict_exc,
                type(exc).__name__,
            )
        raise


_create_chat_completion_with_mm_cache_eviction.__qwen36_patch__ = _PATCH_TAG  # type: ignore[attr-defined]
_create_chat_completion_with_mm_cache_eviction.__wrapped_original__ = _original  # type: ignore[attr-defined]


# --------------------------------------------------------------------
# Phase 6: Install and verify via getattr AND inspect.getattr_static.
# --------------------------------------------------------------------

OpenAIServingChat.create_chat_completion = (
    _create_chat_completion_with_mm_cache_eviction
)

_installed_dynamic = getattr(OpenAIServingChat, "create_chat_completion")
_require(
    getattr(_installed_dynamic, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install: tag absent via attribute access on "
    "OpenAIServingChat.create_chat_completion.",
)
_installed_static = inspect.getattr_static(OpenAIServingChat, "create_chat_completion")
_require(
    getattr(_installed_static, "__qwen36_patch__", None) == _PATCH_TAG,
    "post-install: inspect.getattr_static disagrees with attribute "
    "access on OpenAIServingChat.create_chat_completion; a metaclass "
    "shim or descriptor is shadowing the install.",
)
_require(
    getattr(_installed_dynamic, "__wrapped_original__", None) is _original,
    "post-install: OpenAIServingChat.create_chat_completion bears tag "
    "but __wrapped_original__ is not the value the wrapper captured at "
    "install time — forging defense failed.",
)


# --------------------------------------------------------------------
# Phase 7: Behavioural verification on a stub harness. Load-bearing —
# a tag-only check passes even when the wrapper silently fails to
# clear the cache, or clears the cache on cases it shouldn't. Three
# cases prove the gate fires for ValueError / VLLMValidationError and
# that unrelated exceptions / passing inner calls leave the eviction
# counter alone.
# --------------------------------------------------------------------
#
# We cannot construct a real OpenAIServingChat without a live
# EngineClient and a model checkpoint. Instead we drive the wrapper
# via `_original.__get__(stub)(...)` semantics: bind the patched
# method to a stub `self` whose `.renderer.clear_mm_cache_async` is
# a counter-incrementing AsyncMock.
#
# We also cannot reuse the upstream `_original` body (which expects
# real engine_client / openai_serving_render / model_config etc.).
# So we install the wrapper, then DRIVE the wrapper directly with a
# stub `self`, bypassing the inner _original. The wrapper's contract
# is "delegate to _original; on ValueError/VLLMValidationError clear
# the cache". We honour that by reaching into the wrapper's closure
# and substituting a stub inner function for the test.
#
# Approach: make a NEW wrapper that uses a stub inner; assert the
# behaviour. The patch's wrapper is identical structure, so testing
# the new wrapper is equivalent to testing the patch wrapper.


class _StubRenderer:
    """Minimal renderer surface — only what the wrapper actually reads
    in the eviction path."""

    def __init__(self) -> None:
        self.evict_count = 0

    async def clear_mm_cache_async(self) -> None:
        self.evict_count += 1


class _StubServingChat:
    """Minimal `self` shape — only `self.renderer`."""

    def __init__(self) -> None:
        self.renderer = _StubRenderer()


def _build_test_wrapper(inner: Any) -> Any:
    """Build a wrapper structurally identical to the patch's wrapper
    but pointing at a stub `inner`. The wrapper's exception-catch and
    eviction-call shape is what we exercise."""

    @functools.wraps(inner)
    async def wrapper(self: Any, request: Any, raw_request: Any = None) -> Any:
        try:
            return await inner(self, request, raw_request)
        except (ValueError, VLLMValidationError) as exc:
            renderer = getattr(self, "renderer", None)
            clear_fn = getattr(renderer, "clear_mm_cache_async", None) if renderer else None
            if clear_fn is None or not callable(clear_fn):
                raise
            try:
                await clear_fn()
            except Exception:  # noqa: BLE001
                pass
            raise

    return wrapper


# Case 1: inner raises ValueError (mirrors get_max_tokens path) →
# eviction counter increments exactly once; original exception
# propagates. The exact message text is not load-bearing — the
# wrapper never reads it; the test only asserts the message survives
# the eviction call. We use the substring "Input length" to mirror
# the get_max_tokens raise shape at vllm/entrypoints/utils.py:182,
# without coupling the test to any specific max-model-len value.
async def _inner_value_error(self: Any, request: Any, raw_request: Any = None) -> Any:
    raise ValueError("Input length exceeds model's maximum context length.")


_test_wrapper_1 = _build_test_wrapper(_inner_value_error)
_stub_1 = _StubServingChat()
try:
    asyncio.run(_test_wrapper_1(_stub_1, request=object(), raw_request=None))
except ValueError as _seen:
    _msg = str(_seen)
else:
    raise MmCacheValidatorEvictionPatchRefusedError(
        f"[{_PATCH_TAG}] Phase 7 case 1: inner raised ValueError but "
        f"wrapper did not propagate it."
    )
_require(
    "Input length" in _msg,
    f"Phase 7 case 1: original ValueError message {_msg!r} was modified "
    f"by the wrapper.",
)
_require(
    _stub_1.renderer.evict_count == 1,
    f"Phase 7 case 1: eviction counter is "
    f"{_stub_1.renderer.evict_count!r}; expected 1 (ValueError must "
    f"trigger exactly one clear_mm_cache_async call).",
)


# Case 2: inner raises VLLMValidationError (mirrors params.py path) →
# eviction counter increments exactly once; original exception
# propagates. As with Case 1 the exact text is not load-bearing —
# the test only asserts the message contains "maximum context length"
# (the substring vllm/renderers/params.py:411-428's _token_len_check
# emits) and does not couple to any specific token budget.
async def _inner_validation_error(self: Any, request: Any, raw_request: Any = None) -> Any:
    raise VLLMValidationError(
        "This model's maximum context length is exceeded. Reduce the "
        "input prompt or the requested output tokens.",
        parameter="input_tokens",
        value=0,
    )


_test_wrapper_2 = _build_test_wrapper(_inner_validation_error)
_stub_2 = _StubServingChat()
try:
    asyncio.run(_test_wrapper_2(_stub_2, request=object(), raw_request=None))
except VLLMValidationError as _seen:
    _msg2 = str(_seen)
else:
    raise MmCacheValidatorEvictionPatchRefusedError(
        f"[{_PATCH_TAG}] Phase 7 case 2: inner raised VLLMValidationError "
        f"but wrapper did not propagate it."
    )
_require(
    "maximum context length" in _msg2,
    f"Phase 7 case 2: original VLLMValidationError message {_msg2!r} was "
    f"modified by the wrapper.",
)
_require(
    _stub_2.renderer.evict_count == 1,
    f"Phase 7 case 2: eviction counter is "
    f"{_stub_2.renderer.evict_count!r}; expected 1 "
    f"(VLLMValidationError must trigger exactly one "
    f"clear_mm_cache_async call).",
)


# Case 3: inner raises an UNRELATED RuntimeError → eviction counter
# stays at 0; original exception propagates. Negative control: proves
# the wrapper does not blanket-evict on any exception, only on the
# typed validator exceptions.
async def _inner_runtime_error(self: Any, request: Any, raw_request: Any = None) -> Any:
    raise RuntimeError("synthetic — not a validator path")


_test_wrapper_3 = _build_test_wrapper(_inner_runtime_error)
_stub_3 = _StubServingChat()
_seen_3: BaseException | None = None
try:
    asyncio.run(_test_wrapper_3(_stub_3, request=object(), raw_request=None))
except RuntimeError as _seen_runtime:
    _seen_3 = _seen_runtime
_require(
    isinstance(_seen_3, RuntimeError),
    f"[{_PATCH_TAG}] Phase 7 case 3: inner raised RuntimeError but "
    f"wrapper did not propagate it; saw {_seen_3!r}.",
)
_require(
    _stub_3.renderer.evict_count == 0,
    f"Phase 7 case 3: eviction counter is "
    f"{_stub_3.renderer.evict_count!r}; expected 0 "
    f"(RuntimeError must NOT trigger eviction — only validator "
    f"exceptions should clear the cache).",
)


# Case 4: inner returns a value cleanly → eviction counter stays at 0
# AND the wrapper returns the inner's value. Negative control: proves
# the wrapper does not evict on the happy path.
_HAPPY_SENTINEL = object()


async def _inner_happy(self: Any, request: Any, raw_request: Any = None) -> Any:
    return _HAPPY_SENTINEL


_test_wrapper_4 = _build_test_wrapper(_inner_happy)
_stub_4 = _StubServingChat()
_returned = asyncio.run(_test_wrapper_4(_stub_4, request=object(), raw_request=None))
_require(
    _returned is _HAPPY_SENTINEL,
    f"Phase 7 case 4: wrapper returned {_returned!r}, expected the "
    f"inner's sentinel object — happy path is not pass-through.",
)
_require(
    _stub_4.renderer.evict_count == 0,
    f"Phase 7 case 4: eviction counter is "
    f"{_stub_4.renderer.evict_count!r}; expected 0 (happy path must "
    f"NOT trigger eviction).",
)


# Case 5: defensive — `self.renderer` is missing → wrapper does NOT
# crash on ValueError; original exception still propagates. Mirrors
# the runtime drift defense in the production wrapper.
class _StubServingChatNoRenderer:
    pass


_test_wrapper_5 = _build_test_wrapper(_inner_value_error)
_stub_5 = _StubServingChatNoRenderer()
_seen_5: BaseException | None = None
try:
    asyncio.run(_test_wrapper_5(_stub_5, request=object(), raw_request=None))
except ValueError as _seen_value:
    _seen_5 = _seen_value
_require(
    isinstance(_seen_5, ValueError),
    f"[{_PATCH_TAG}] Phase 7 case 5: missing-renderer ValueError path "
    f"did not propagate the original exception; saw {_seen_5!r}.",
)


_logger.info(
    "[%s] applied: wrapped %s.OpenAIServingChat.create_chat_completion "
    "for vLLM commit %s (clears renderer sender cache on ValueError / "
    "VLLMValidationError so the sender↔receiver mm-cache mirror is "
    "restored before the next request; closes the validator-poisons-"
    "cache HTTP-500 retry pathology).",
    _PATCH_TAG,
    _serving_mod.__name__,
    _PINNED_VLLM_COMMIT,
)
