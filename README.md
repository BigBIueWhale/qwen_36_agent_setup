# Qwen3.6 on a single RTX 5090 — production-grade agentic deployment

**Last updated: 2026-04-27.** `QuantTrio/Qwen3.6-27B-AWQ` (dense VL, AWQ INT4) on **vLLM** (llama.cpp rejected, §5.4). Multi-tool agentic pipelines with vision, reasoning, preserved-thinking. Host: single NVIDIA RTX 5090, 32 GB VRAM, Blackwell SM 12.0, Linux, CUDA 13.0, nvidia-container-toolkit v1.19.0.

This README documents a specific, pinned, reproducible deployment. Every choice below is deliberate.

### Deployment target & load-bearing pins

**`QuantTrio/Qwen3.6-27B-AWQ`** (selected 2026-04-25, replacing an earlier `RedHatAI/Qwen3.6-35B-A3B-NVFP4` target). The 27B dense model is more capable per forward pass than a 35B-A3B MoE that activates only ~3B per token, fits on the card cleanly, and the AWQ recipe preserves the load-bearing layers (vision, `linear_attn.in_proj_a/b`, `lm_head`, embeddings, layer 0, MTP) at BF16 while quantizing MLPs. NVFP4 builds for 27B-dense were triaged and rejected — see §3.1.

| Slot | Pin |
|---|---|
| Model | `QuantTrio/Qwen3.6-27B-AWQ` at revision `9b507bdc9afafb87b7898700cc2a591aa6639461` |
| Quantization format | AWQ INT4 (gemm, group_size=128, zero_point=true), data-free calibration. Vision encoder, `linear_attn.in_proj_a/b`, all `self_attn.{q,k,v}_proj`, layer 0, embeddings, `lm_head`, MTP, norms/conv1d kept BF16 |
| Runtime | vLLM Docker image `vllm/vllm-openai@sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba` (build `0.19.2rc1.dev212+g8cd174fa3`, commit `8cd174fa358326d5cc4195446be2ebcd65c481ce`, container CUDA 13.0.2, PyTorch `2.11.0+cu130`, FlashInfer `0.6.8.post1`, transformers `5.6.2`, pydantic `2.13.3`, image built 2026-04-26 05:19 UTC) |
| KV cache dtype | BF16 (FP8 KV scales are not shipped with this checkpoint; rationale §5.1) |
| `--max-model-len` | **65,536** |
| Disk size | 20.36 GiB |
| VRAM at boot (measured) | 19.78 GiB resident (MTP head auto-skipped via `qwen3_5.py:701-706`'s `skip_prefixes=["mtp."]`) |
| KV pool available | 6.89 GiB at gmu=0.92, image:2/video:1. Boot log: `GPU KV cache size: 111,328 tokens` (patch 3 installed; unpatched §6.3 would display ~28K) |
| Per-token attention KV at BF16 | `4 KV heads × 256 head_dim × 2 (K+V) × 16 attn layers × 2 bytes = 65,536 bytes/token` (16 GiB at the model's native 262K context) |
| `preserve_thinking` | Set as server-wide default via `--default-chat-template-kwargs '{"preserve_thinking": true}'` |
| MTP speculative decoding | OFF (head present in the AWQ checkpoint as BF16 but auto-skipped at load; §5.3) |
| Patches in this repo | 5 strict, fail-loud Python monkey-patches (§7), all server-side, loaded before `vllm serve` starts. Zero client-side code |

---

## 1. What this project is

A production deployment of **`QuantTrio/Qwen3.6-27B-AWQ`** (a 27-billion-parameter dense vision-language model from Alibaba, AWQ INT4 quantized) served via **vLLM** behind an OpenAI-compatible HTTP API at `http://127.0.0.1:8000/v1/chat/completions`, intended for agentic coding via the Qwen Code CLI on a single RTX 5090.

Five non-negotiable correctness requirements:

1. **Tool calling that actually works in multi-turn agent loops** — no silent failures.
2. **Preserved thinking across tool turns** — historical `<think>` blocks remain visible to the model on subsequent turns.
3. **Vision input at full preprocessing fidelity** — vision encoder kept BF16; HF `Qwen3VLImageProcessor` semantics preserved.
4. **65,536-token `--max-model-len` at BF16 KV cache precision** (FP8 KV rejected; §5.1).
5. **Stability over a marginal throughput gain** — MTP off; rationale §5.3.

Every software pin, every launch flag, every server-side monkey-patch in this repo exists to uphold them simultaneously.

---

## 2. Hardware

| Component | Spec | Why it matters |
|---|---|---|
| GPU | NVIDIA RTX 5090, 32 GB VRAM | Largest consumer Blackwell card. 32 GB is the binding constraint: dictates 4-bit AWQ weights and `--max-model-len 65536`. |
| GPU compute capability | SM 12.0 (Blackwell consumer) | Native FP4 tensor cores (unused by AWQ, used by FlashInfer where dispatched). |
| Host OS | Linux (not WSL2) | dxgkrnl on WSL2 does not expose Blackwell's native FP8 tensor cores. |
| Host CUDA | 13.0 (runtime) | Image's PyTorch links against CUDA 12.9 — runs on host CUDA 13.0 via NVIDIA's forward-compat layer. |
| Host driver | NVIDIA Linux driver ≥ 580.65.06 | Minimum for CUDA 13.0 forward-compatibility. Recommended: 580.95.05 (current R580 LTS). |
| nvidia-container-toolkit | 1.19.0 (released 2025-03-12) | Provides `--gpus all` GPU passthrough. |

### 2.1 Why not larger GPUs or multi-GPU

Designed for a single 32 GB card deliberately. Qwen's official BF16 weights are too large for a single 5090 even at 27B; staying on one GPU avoids an entire class of NCCL / expert-parallelism bugs that Blackwell vLLM has open against it.

---

## 3. Software pins (exact)

All versions are pinned for reproducibility. Floating tags (`latest`, `main`, `nightly`) are not used.

### 3.1 Model

| Field | Value |
|---|---|
| HuggingFace repo | `QuantTrio/Qwen3.6-27B-AWQ` |
| Revision (commit SHA) | `9b507bdc9afafb87b7898700cc2a591aa6639461` |
| Disk size | 20.36 GiB total (~9.34 GiB BF16 + ~10.60 GiB INT4-packed values + scales) |
| VRAM at boot | 19.78 GiB resident |
| Format | AWQ INT4 (gemm, `group_size=128`, `zero_point=true`), data-free calibration |
| Kept BF16 | vision encoder (27 Qwen3-VL layers), `embed_tokens`, `lm_head`, layer 0 entirely, `self_attn.{q,k,v}_proj` for all 16 full-attention layers, `linear_attn.in_proj_{a,b}` for the 47 GDN layers excluding layer 0, `linear_attn.conv1d`/`norm`/`A_log`/`dt_bias`, MTP head |
| Quantized INT4 | `linear_attn.{in_proj_qkv, in_proj_z, out_proj}` for the 47 GDN layers, `self_attn.o_proj` for the 16 full-attention layers, all MLPs |

Keeping `linear_attn.in_proj_{a,b}` in BF16 is likely load-bearing for thinking-mode loop resistance: `cpatonn` on HF `cyankiwi/Qwen3.5-27B-AWQ-4bit` discussion #2 reports that `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4` (linear-attn in BF16, rest INT4) was *"significantly better on the infinite loop issue"*. The MTP head ships in this checkpoint but is auto-skipped at load via `qwen3_5.py:701-706`'s `skip_prefixes=["mtp."]`.

**Why this quant over published NVFP4 alternatives**: every Qwen3.6-27B NVFP4 quant triaged either won't fit on 32 GiB (preserves full BF16 everything except MLPs), lacks a documented recipe, strips vision, or targets a different runtime. QuantTrio AWQ is the only option that fits on the card, preserves the load-bearing layers, works with vLLM's `qwen3_coder` tool parser, and has been empirically validated on the target hardware (210-prompt corpus, 0% true garbling).

### 3.2 Runtime

| Field | Value |
|---|---|
| Runtime | vLLM |
| Docker image (published) | `vllm/vllm-openai:nightly-8cd174fa358326d5cc4195446be2ebcd65c481ce` |
| Image digest (amd64) | `sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba` |
| Underlying vLLM commit | `8cd174fa358326d5cc4195446be2ebcd65c481ce` (`0.19.2rc1.dev212+g8cd174fa3`) |
| Image build timestamp | 2026-04-26 05:19:53 UTC |
| CUDA toolkit inside image | 13.0.2 |
| Image PyTorch | `2.11.0+cu130` |
| Image FlashInfer | `0.6.8.post1` |
| `transformers` pinned inside image | `5.6.2` (past vLLM PR #40331's brief 2026-04-21 boot regression, reverted by `3975eb6de6`) |
| `pydantic` pinned inside image | `2.13.3` (matches the version the §7.4 egress patch's mechanism was empirically validated against) |

vLLM CI publishes a commit-tagged variant for every nightly build (`nightly-<commit>`), digest-pinnable. We pin by digest because nightly tags can be re-pushed. Master HEAD when this pin was selected was `32e45636e3` (3 commits ahead, no patched surface touched).

### 3.3 Client

No client-side code in this repo. The deployment exposes an OpenAI-compatible HTTP API at `/v1/chat/completions`. Verified to connect unmodified: **OpenAI Python SDK**, **Qwen Code CLI** (Alibaba TypeScript), **Qwen-Agent** (Alibaba Python — `qwen_agent/llm/oai.py`), and any other Chat Completions client that reads/writes `choices[i].(message|delta).reasoning_content`.

The wire-level interop hazards that made third-party clients silently lose data — §6.1 (ingest), §6.4 (egress) — are closed server-side by the §7 patches. §6.5 (`<tool_call>`-in-`<think>`) is detected but the wire passes through unchanged so the agent's retry policy decides.

---

## 4. Model architecture background

Qwen3.6-27B is a **dense** vision-language model with a hybrid attention pattern.

- **64 decoder layers**, arranged at `full_attention_interval=4`:
  - **16 full (softmax) attention layers** at indices 3, 7, 11, …, 63 — the only layers whose KV cache grows with context.
  - **48 Gated DeltaNet** linear-attention layers — fixed-size recurrent state, does not grow with context.
- **Attention config of full layers**: `num_attention_heads=24`, `num_key_value_heads=4`, `head_dim=256`, `hidden_size=5120`, `intermediate_size=17408`. GQA with low KV-head count keeps per-token KV small.
- **Native context**: `max_position_embeddings=262144` (we cap at 65,536 for VRAM reasons).
- **Vocabulary**: `vocab_size=248320`.
- **Rotary positional embedding**: M-RoPE; used for both text and vision tokens.
- **Vision encoder**: 27-layer Qwen3-VL tower, BF16, bundled into the main checkpoint.
- **MTP (Multi-Token Prediction) head**: 1-layer module, BF16, present in the AWQ checkpoint (~0.68 GiB) but auto-skipped at load. See §5.3.

The hybrid attention pattern is why the KV-cache memory math diverges from a pure transformer: only 16 of 64 layers contribute to KV growth with context.

---

## 5. Decisions that shape the deployment

### 5.1 KV cache: BF16, not FP8

`--kv-cache-dtype auto` (resolves to BF16). FP8 KV halves KV memory but requires per-tensor or per-head scaling factors to preserve numerical range. Qwen3.6-27B-AWQ does not ship calibrated FP8 KV scales — vLLM falls back to scale=1.0, which is the documented cause of mild long-context quality drift. SGLang's docs flag the same hazard: *"these FP8 checkpoints do not include pre-calibrated KV cache scaling factors; SGLang defaults to scale 1.0, which may cause noticeable accuracy degradation on reasoning-heavy tasks."* For a reasoning model in an agentic pipeline this is unacceptable. Rotation-based low-bit KV (TurboQuant) does not yet support hybrid attention + GDN architectures in vLLM.

### 5.2 Max context length: 65,536 tokens

`--max-model-len 65536`. The byte budget on a 32 GiB card after weights + activations + vision profiling reservation leaves ~6.89 GiB for KV at gmu=0.92. At per-token attention KV of 65,536 bytes/token for 27B (4 KV heads × 256 head_dim × 2 (K+V) × 16 attn layers × 2 bytes), 65,536 tokens is comfortable headroom. Boot log reports `GPU KV cache size: 28,224 tokens` (under-reported by ~4× due to the §6.3 hybrid-KV bug; with patch 3 installed → 111,328 measured 2026-04-27). The model's native 262K context would require 16.0 GiB of attention-only KV — does not fit on this card without dropping precision.

`--gpu-memory-utilization 0.92` is the empirical knee: lower wastes pool, higher risks Triton-warmup OOM.

### 5.3 MTP speculative decoding: OFF

The MTP head ships in the QuantTrio AWQ checkpoint as BF16 (~0.68 GiB) but is auto-skipped at load via `qwen3_5.py:701-706`'s `skip_prefixes=["mtp."]`. Not exercised on the 27B-AWQ deployment. To enable: append `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` to the launch line. **Not validated on this stack** — vLLM issues #36872 and #38182 still apply in principle (MTP gibberish on quantized weights, prefix-cache hit rate regression).

### 5.4 Runtime: vLLM, not llama.cpp

Re-verified against `ggml-org/llama.cpp` master `0d0764df` (2026-04-22):

| Dimension | vLLM | llama.cpp |
|---|---|---|
| Dedicated Qwen3 tool parser | Yes (`qwen3_coder`) | **No.** Qwen3.x routes through generic `peg-native`; live OPEN failure modes #20260/20837/21771/22240. |
| Dedicated reasoning parser | Yes (`qwen3`) | Generic `--reasoning-format deepseek`; no Qwen3-specific handling. |
| MTP head | Loaded (disabled by §5.3) | **Dropped at GGUF conversion** (`convert_hf_to_gguf.py:4781-4782`). PR #20700 unmerged WIP. |
| Vision preprocessing | BICUBIC (matches HF `Qwen3VLImageProcessor`) | **BILINEAR** hardcoded at `tools/mtmd/clip.cpp:1357` for QWEN3VL. No upstream PR open. |
| Agentic prompt-cache stability | Stable | Live regression llama.cpp #21383 (OPEN, 2026-04-03): Qwen3.5-27B CUDA illegal-memory-access under `--cache-ram`. |
| Qwen official endorsement | Launch command in HF model card | "Supported" without a pinned command |

The tool-parsing, MTP-drop, vision-preprocessing, and agentic-prompt-cache gaps above are live on master with no merged fix.

### 5.5 API endpoint: `/v1/chat/completions`, not Responses API

vLLM issue #39584 asserts `len(tool_calls)==1` in the Responses API streaming path, which crashes on legitimate parallel tool calls. Chat Completions is universally supported across every Python and TypeScript OpenAI-SDK client; using it costs nothing and sidesteps the bug.

### 5.6 Sampling parameters

Per the Qwen3.6 model card "Best Practices" block. Thinking-mode agentic use:
`temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, max_tokens=16384`. Precise coding / low-variance tool args: `temperature=0.6, presence_penalty=0.0`, rest unchanged.

`presence_penalty=1.5` is Alibaba's shipped mitigation for the Qwen3-family thinking-mode loop pathology (LiveCodeBench: 17.4% of 1,400 outputs truncated with no `</think>`, 27.5% on hard problems). May cause language mixing and slight perf decrease per Qwen's docs. Community alternative: `min_p=0.2` with `temperature=1.0` (drop `presence_penalty` back to 0). `max_tokens=16384` gives headroom against the truncation crash mode patched in §7.2. Qwen3.6 does NOT support soft `/think` or `/nothink` switches — thinking mode is controlled exclusively via `chat_template_kwargs`.

### 5.7 `preserve_thinking=true` as a server-side default

`--default-chat-template-kwargs '{"preserve_thinking": true}'` sets `preserve_thinking=true` as a server-wide default. The Qwen3.6 chat template only emits `<think>` blocks from historical assistant turns when `preserve_thinking=true`; otherwise only the most recent assistant turn retains its `<think>`. The model was RL-trained expecting reasoning to persist across turns; stripping it degrades tool-argument correctness after 2–3 turns (`badlogic/pi-mono#3325`).

`enable_thinking` is *not* set server-side — clients pass `"chat_template_kwargs": {"enable_thinking": true}` in the request body when they want thinking on.

The separate `reasoning` vs `reasoning_content` wire-format mismatch (§6.1 ingest, §6.4 egress) is closed by the §7.1 and §7.4 patches.

### 5.8 Multimodal cost accounting

- **Vision encoder weights** (~0.83 GiB) are always resident in VRAM, whether or not any request contains an image. Subsumed into the boot weights footprint.
- **Vision encoder profiling reservation** (~1.56 GiB) is paid at boot if `--limit-mm-per-prompt` admits any image or video modality. Binary switch: `image: 1` and `image: 3` pay the same tax. Setting `{image: 0, video: 0, audio: 0}` reclaims it.
- **Per-request transient activations** are small (+2–10 MiB measured for 896×896 to 1792×1792 images); absorbed by safety headroom.
- **Image and video tokens occupy the KV pool**: ~256 tokens for 512×512, ~787 for 896×896, ~3,136 for 1792×1792, up to ~16,384 per image at the processor cap. Videos scale dramatically — short clips ~750–1,500 tokens; medium clips ~10,000–20,000; long clips can exceed 50,000. Bound video size with `--mm-processor-kwargs '{"video": {"max_frames": 64, "fps": 1}}'` if accepting untrusted clips.

`--limit-mm-per-prompt '{"image": 2, "video": 1, "audio": 0}'` is **load-bearing**: omitting the flag defaults to 999 per modality, which **crashes the boot** on this nightly + SM 12.0 (post-profiling TensorRT-LLM `throwRuntimeError`). Safe range is `image ∈ {0, 1, 2, 3}`. Audio stays 0: Qwen3.6 is text+vision only.

---

## 6. vLLM issues — complete enumeration and per-issue disposition

Each issue below is classified as **A**. Runtime bug, **B**. Model OOD failure, **C**. Infrastructure bug, or **D**. Client-interop bug. The five surviving §6 entries each correspond to one §7 patch in the same numerical slot.

### 6.1 Ingest silently drops `reasoning_content` [Class A — vLLM internal inconsistency]

`vllm/entrypoints/chat_utils.py:1519` reads `message.get("reasoning")` and silently drops `reasoning_content`. The chat template at `chat_template.jinja:91-92` reads `message.reasoning_content` to render historical `<think>` blocks under `preserve_thinking=true`. **vLLM is feeding its own template a field its own ingest discards.** Without resolution, every multi-turn agent loop loses prior reasoning on replay; the model — RL-trained to expect prior-turn reasoning — re-derives context from scratch and tool-arg correctness degrades after 2-3 turns. Our two production clients (Qwen Code CLI, Qwen-Agent) both write `reasoning_content`.

**Affects us**: yes. **Resolution**: §7.1.

### 6.2 Issue #39771 — qwen3_coder crashes on truncated `<parameter=` tag [Class A]

`vllm/tool_parsers/qwen3coder_tool_parser.py:236` uses unsafe `str.index(">")`. When the model is truncated mid-`<parameter=NAME` (before `>`), `.index()` raises `ValueError`. The exception is caught at lines 320-324 and the parser returns `tools_called=False, tool_calls=[]` — collapsing **every well-formed sibling tool call in the same response**. Sibling code at line 227 already uses the safe `.find()/-1` pattern; line 236 is an internal inconsistency upstream PR #39772 acknowledges.

**Affects us**: yes, whenever a response is truncated by `max_tokens` or by client disconnect mid-generation. **Resolution**: §7.2; complemented by `max_tokens=16384` keeping truncation rare.

### 6.3 Issue #37121 — hybrid-KV scheduler/log under-reports concurrent capacity [Class D — observability bug]

vLLM's V1 paged KV cache manager forms one `KVCacheGroupSpec` per same-shape layer set. For Qwen3.6-27B that's 4 groups (1 full × 16 + 3 GDN × 16). The byte allocator at `vllm/v1/core/kv_cache_utils.py:1148-1169` allocates a single shared pool sized correctly. The bug is purely in *reporting*: `_report_kv_cache_config:1305-1346` and `get_max_concurrency_for_kv_cache_config:802-820` divide by `len(kv_cache_groups)` (4 for our model), so the displayed `GPU KV cache size: X tokens` and `Maximum concurrency` are ~4× understated. For the 27B-AWQ build: boot log says `GPU KV cache size: ~28K tokens`; with patch 3 installed → `~111K tokens`. Operators sizing `--max-model-len` against the displayed number under-utilize their hardware.

**Upstream status (2026-04-25)**: issue #37121 open since 2026-03-15. PR #40384 (narrow scheduler-reporting fix, our backport source) and competing PR #40694 both open. PR #37429 (broader byte-level redesign) blocked on RFC.

**Resolution**: §7.3. Cosmetic for the deployment — byte allocation and admission are correct without it.

### 6.4 Egress emits non-standard `reasoning` field name [Class D]

vLLM emits the non-standard field name `reasoning` on the wire (since commit `c5113f60f2` deliberately removed `reasoning_content`). Qwen-Agent's OAI client at `Qwen-Agent/qwen_agent/llm/oai.py:111-112,126-127,169` strict-checks `reasoning_content` with **no fallback**; without the alias, every multi-turn agent loop loses prior reasoning on egress and degrades after 2-3 turns. Pydantic v2 compiled core schemas embed nested schemas by snapshot at build time, so a leaves-only rebuild still leaks `reasoning` through wrappers — every class on the dump chain must rebuild under `serialize_by_alias=True` for the leaf alias to reach the wire.

**Affects us**: yes. **Resolution**: §7.4 (egress). §7.1 closes the matching ingest half.

### 6.5 Issue #39056 — `<tool_call>` inside `<think>` is model OOD, not a parser bug [Class B]

Qwen3.6 occasionally emits `<tool_call>...</tool_call>` markup inside `<think>...</think>` (single-digit percent under agentic workloads). Upstream's `Qwen3ReasoningParser.extract_reasoning` correctly partitions on `</think>` first (`qwen3_reasoning_parser.py:142-144`) and routes mid-think markup to `reasoning`. **The parser is correct to its contract; the model is misbehaving.** Qwen3.6's chat template never renders historical tool_calls inside `<think>`, the Qwen3-Coder-Next training penalizes the pattern, and Alibaba's own evaluation in `Qwen-Agent/benchmark/deepplanning` strips everything up to `</think>` before parsing.

**Affects us**: yes, intermittently. **Resolution**: §7.5 (detect, don't rescue). The agent's retry policy decides what to do; the patch surfaces a structured WARNING so an operator can monitor the rate.

### 6.6 Issue #39584 — parallel tool calls crash Responses API [Class A — sidestepped]

`vllm/entrypoints/openai/responses/serving.py:1377` has a hardcoded `assert len(delta_message.tool_calls) == 1`. **Affects us**: no — we use Chat Completions.

---

## 7. The patches in this repo

Five monkey-patches plus one container-entrypoint launcher plus one sitecustomize loader. Every patch is **server-side**, loaded into the vLLM Python process by `launch_with_patches.py` (in PID 1) and re-loaded by `sitecustomize.py` in spawned EngineCore subprocesses (load-bearing for patch 3; see §7.S). There is no client-side code in this repo.

Each patch addresses a specific defect named in §6 and only that defect. Each strictly validates its target's structure via landmarks, refuses to apply on any landmark mismatch with a typed exception that names the exact landmark that failed, stamps `__qwen36_patch__` on every target, and verifies via both `getattr` and `inspect.getattr_static` that its install took effect. The patch file itself is the source of truth; this section is a contract index.

| # | File | Defect addressed |
|---|---|---|
| 1 | [`monkey_patch_reasoning_field_ingest.py`](monkey_patch_reasoning_field_ingest.py) | §6.1 — accept `reasoning_content` on replayed assistant messages |
| 2 | [`monkey_patch_qwen3_coder.py`](monkey_patch_qwen3_coder.py) | §6.2 / #39771 — `_parse_xml_function_call` crash on truncated `<parameter=` |
| 3 | [`monkey_patch_hybrid_kv_allocator.py`](monkey_patch_hybrid_kv_allocator.py) | §6.3 / #37121 (PR #40384 backport) — boot-log under-reporting |
| 4 | [`monkey_patch_reasoning_field_egress.py`](monkey_patch_reasoning_field_egress.py) | §6.4 — rename `reasoning` → `reasoning_content` on response serialization |
| 5 | [`monkey_patch_tool_call_in_think_detector.py`](monkey_patch_tool_call_in_think_detector.py) | §6.5 / #39056 — detect `<tool_call>` emitted inside `<think>`, structured WARNING |
| L | [`launch_with_patches.py`](launch_with_patches.py) | Container entrypoint that imports the 5 patches in order, runs per-patch verification, then hands off to `vllm.entrypoints.cli.main` via `runpy.run_module(alter_sys=True)` |
| S | [`sitecustomize.py`](sitecustomize.py) | CPython auto-imports this from `PYTHONPATH=/opt/patches` at every interpreter startup — including the spawned EngineCore subprocess — so patch 3's targets are live in EngineCore's own `sys.modules` |

### 7.1 Patch 1 — `monkey_patch_reasoning_field_ingest.py`

Wraps `vllm.entrypoints.chat_utils._parse_chat_message_content`. When `role == "assistant"`, `reasoning is None`, and `reasoning_content` is a non-None string, synthesises a shallow copy with `reasoning` populated from `reasoning_content`. Both fields present with **different** values raises `ReasoningFieldAmbiguityError` (HTTP 400). Identical values pass through. Non-string `reasoning_content` (dict, list) refuses rather than being stringified. **Removal trigger**: vLLM widens ingest to accept `reasoning_content`.

### 7.2 Patch 2 — `monkey_patch_qwen3_coder.py`

Replaces `Qwen3CoderToolParser._parse_xml_function_call` to use `str.find(">")` and, on a malformed `<parameter=` tag, return `None` for the whole tool call rather than raising `ValueError` (which the upstream `try/except Exception` collapses into "drop all N tool calls in this response"). Sibling well-formed tool calls in the same response are preserved. The MRO walk for inherited `self.tools` / `self.tool_call_parameter_regex` attribute landmarks is load-bearing per the prior audit (the inherited `self.tools` from `ToolParser.__init__` would refuse incorrectly under a non-walking check). **Removal trigger**: PR #39772 merges.

### 7.3 Patch 3 — `monkey_patch_hybrid_kv_allocator.py`

Replaces `get_max_concurrency_for_kv_cache_config` and `_report_kv_cache_config` in `vllm.v1.core.kv_cache_utils`. Both reporting sites divide by `len(kv_cache_groups)` (4 for our model: 1 attn + 3 GDN), making displayed token capacity ~4× understated. Patched to filter `kv_cache_groups` to token-capacity-contributing specs only (`AttentionSpec` always; `MambaSpec` only when `cache_config.mamba_cache_mode == "all"`). Boot-log `GPU KV cache size` goes from `~28K tokens` to `~111K tokens`. Backport semantics, not literal port — the pinned commit's `MambaSpec.max_memory_usage_bytes` signature is `(self, vllm_config)`, not the master signature PR #40384 targets. **Removal trigger**: PR #40384 or PR #40694 merges. **CRITICAL**: remove BEFORE pulling an image with PR #37429 (the broader byte-level redesign) — it changes the tensor layout and the patch's reporting view would no longer be coherent.

### 7.4 Patch 4 — `monkey_patch_reasoning_field_egress.py`

Installs Pydantic v2 `serialization_alias = "reasoning_content"` on the `reasoning` field of `ChatMessage` and `DeltaMessage`, flips `model_config["serialize_by_alias"] = True` on **all six** classes that vLLM dumps on the wire (the two leaves plus `ChatCompletionResponseChoice`, `ChatCompletionResponseStreamChoice`, `ChatCompletionResponse`, `ChatCompletionStreamResponse`), drops the cached `__pydantic_core_schema__` / `__pydantic_validator__` / `__pydantic_serializer__`, and calls `model_rebuild(force=True)`. A leaves-only patch is provably insufficient — Pydantic v2 compiled core schemas embed nested schemas by snapshot at build time. Phase 3 verification constructs real `ChatCompletionResponse` and `ChatCompletionStreamResponse` instances, dumps each via `model_dump_json()` and `model_dump_json(exclude_unset=True)`, and asserts wire bytes contain `"reasoning_content":` and not `"reasoning":`. The internal Python attribute stays `.reasoning`. **Removal trigger**: vLLM ships native `reasoning_content` on Chat Completions.

### 7.5 Patch 5 — `monkey_patch_tool_call_in_think_detector.py`

Wraps `Qwen3ReasoningParser.extract_reasoning` (non-streaming **only**) and emits a single structured WARNING (`model_emit_warning kind=tool_call_in_reasoning reasoning_len=N marker_count=M`) whenever the upstream-returned reasoning half contains literal `<tool_call>` markup. The wrapped return value is the upstream tuple **unchanged** — the agent's retry policy decides what to do. Detect, don't rescue: the parser is correct to its contract, the model is misbehaving, and a state-machine bandage across streaming deltas is high-complexity for a stochastic model-OOD failure mode. Streaming path is intentionally unwrapped — the rate is a model-side property, not per-modality, and a single non-streaming wrapper suffices. **Removal trigger**: Qwen3.6 retraining eliminates the OOD emission.

### 7.L Launcher — `launch_with_patches.py`

Container entrypoint, replacing `["vllm", "serve"]` with `["python", "/opt/patches/launch.py", "serve", ...]`. Imports every registered patch in `_PATCH_MODULES` order, runs the per-patch `_PATCH_VERIFICATION` verifier for each (re-imports the relevant vLLM target FROM SCRATCH and asserts the install took effect), then hands off to vLLM's CLI via `runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)`. Required because `PYTHONSTARTUP` does not fire under non-interactive entrypoints.

Verifiers split into two classes. Patches 2 and 3 carry **behavioural** verifiers — they instantiate the patched class with a synthetic input designed to expose the bug and assert the post-patch return value. Patches 1, 4, 5 carry tag-only verifiers (with `getattr` and `inspect.getattr_static` agreement); their patch-internal Phase verifications (the egress patch's wire-dump check; the ingest patch's static-lookup check) carry the load-bearing functional verification, so duplicating it here would only double the surface area.

Three pre-flight checks run BEFORE the per-patch import loop: sitecustomize-present (refuse if Debian's stub got loaded instead of ours), registry drift (refuse if `sitecustomize._PATCH_MODULES != launch_with_patches._PATCH_MODULES`), and a subprocess install probe (`subprocess.run([sys.executable, "-c", PROBE])` to confirm a freshly-spawned interpreter sees patched targets). All three are load-bearing for patch 3 — without them, the spawned EngineCore silently runs unpatched code while the launcher reports success. **Load order** matters: `reasoning_field_egress` (patch 4) must come before any patch that constructs `DeltaMessage` at request time, since the rebuild changes Pydantic's compiled schema.

### 7.S sitecustomize loader — `sitecustomize.py`

vLLM v1 spawns EngineCore as a `multiprocessing` child process via `spawn` (CUDA forbids `fork` after init). The spawned interpreter does not inherit `sys.modules`. Of the five patches, **only patch 3** (`monkey_patch_hybrid_kv_allocator`) targets EngineCore-resident code; patches 1, 2, 4, 5 target API-server-resident code and become live in PID 1 directly. Without `sitecustomize`, patch 3 is silently dead in EngineCore while the launcher's PID-1 verifier reports success. The pass/fail discriminator is the boot-log filename annotation: `[kv_cache_utils.py:NNN]` (unpatched, ~28K tokens) vs `[monkey_patch_hybrid_kv_allocator.py:NNN]` (patched, ~111K tokens).

CPython's `site.py` auto-imports `sitecustomize` from `sys.path` at every interpreter startup, including spawned children. With `PYTHONPATH=/opt/patches`, `site.py` finds our file, which imports each patch in launcher order. Each patch's strict landmark check runs in EngineCore too; any refusal aborts startup loudly. The same flow runs in PID 1 — sitecustomize installs the patches, `launch.py` then hits cache via `importlib.import_module(...)`. Each patch's module-level code fires once.

**Load-bearing for**: patch 3. **Defense-in-depth for**: patches 1, 2, 4, 5. **Removal trigger**: when patch 3 is removed; recommendation is to keep for defense-in-depth.

---

## 8. Deployment commands

### 8.1 Fetch and pin the Docker image digest

```bash
docker pull vllm/vllm-openai@sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba

docker inspect vllm/vllm-openai@sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba \
  --format '{{.Id}} {{.Architecture}}'
```

### 8.2 Launch vLLM

```bash
docker run --rm -d --name qwen36 --gpus all \
  --network host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/sitecustomize.py:/opt/patches/sitecustomize.py:ro" \
  -v "$PWD/monkey_patch_reasoning_field_ingest.py:/opt/patches/monkey_patch_reasoning_field_ingest.py:ro" \
  -v "$PWD/monkey_patch_qwen3_coder.py:/opt/patches/monkey_patch_qwen3_coder.py:ro" \
  -v "$PWD/monkey_patch_hybrid_kv_allocator.py:/opt/patches/monkey_patch_hybrid_kv_allocator.py:ro" \
  -v "$PWD/monkey_patch_reasoning_field_egress.py:/opt/patches/monkey_patch_reasoning_field_egress.py:ro" \
  -v "$PWD/monkey_patch_tool_call_in_think_detector.py:/opt/patches/monkey_patch_tool_call_in_think_detector.py:ro" \
  -v "$PWD/launch_with_patches.py:/opt/patches/launch.py:ro" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e VLLM_USE_V1=1 \
  -e PYTHONPATH=/opt/patches \
  --entrypoint python3 \
  vllm/vllm-openai@sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba \
  /opt/patches/launch.py serve \
  --model QuantTrio/Qwen3.6-27B-AWQ \
  --revision 9b507bdc9afafb87b7898700cc2a591aa6639461 \
  --served-model-name Qwen3.6-27B-AWQ \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  --limit-mm-per-prompt '{"image": 2, "video": 1, "audio": 0}'
```

Each flag's rationale lives in §5. Load-bearing items: `--limit-mm-per-prompt` (default crashes boot, §5.8); `--max-model-len 65536` (byte budget, §5.2); `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` (tool calling); `--reasoning-parser qwen3` (server-side `<think>` extraction); `--default-chat-template-kwargs '{"preserve_thinking": true}'` (§5.7); `--host 127.0.0.1` + `--network host` (loopback-only — do not change to `0.0.0.0` on a publicly-routable host without an authenticating reverse proxy; vLLM has no built-in auth); `sitecustomize.py` bind-mount (load-bearing for patch 3, see §7.S).

### 8.3 Smoke tests

```bash
# Liveness
curl -fs http://127.0.0.1:8000/health  # expects HTTP 200, empty body

# Chat with thinking + tool schema
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-AWQ",
    "messages": [{"role": "user", "content": "What is 127 * 349?"}],
    "tools": [{"type": "function", "function": {
      "name": "calculator", "description": "Evaluate a math expression.",
      "parameters": {"type": "object", "properties": {"expr": {"type": "string"}},
                     "required": ["expr"]}
    }}],
    "temperature": 0.6, "max_tokens": 4096,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}}
  }' | jq .
```

Expected (after §7.4 egress patch): `choices[0].message.reasoning_content` populated; `choices[0].message.tool_calls[0].function.name == "calculator"` with valid JSON `arguments`; `finish_reason == "tool_calls"`.

**Operator visibility for `<tool_call>`-in-`<think>` model-OOD emission**: grep server logs for `model_emit_warning kind=tool_call_in_reasoning` (the §7.5 detector's structured WARNING).

---

## 9. Known unknowns

| Item | Evidence status |
|---|---|
| `<tool_call>`-in-`<think>` rate, thinking-mode loop rate under our AWQ config | `<tool_call>`-in-`<think>` community-reported single-digit-percent; patch §7.5 surfaces it as structured WARNING (`model_emit_warning kind=tool_call_in_reasoning`). **Loop rate** measured 17.4% on Qwen3.5-35B-A3B unquantized; not re-measured under AWQ + BF16-linear-attn + BF16-KV. |
| Image count boot failure threshold (§5.8) | `image ∈ {0, 1, 2, 3}` boots; `image ∈ {10, 100, 999}` (and the default-999 when flag omitted) crash with TensorRT-LLM `throwRuntimeError`. We did not bisect 4–9; treat 3 as the ceiling. |

---

## 10. What this deployment explicitly does not use

| Option | Reason |
|---|---|
| llama.cpp | No Qwen3 tool/reasoning parser; vision uses BILINEAR vs HF BICUBIC; agentic prompt-cache regression #21383 open. Full citations §5.4. |
| SGLang | Triton dtype mismatch in `causal_conv1d_triton.py:510` blocked our 27B-AWQ boot; vLLM bypasses via `mamba_mixer2 + gdn_attention_core`. |
| Other quants (NVFP4, GPTQ, MXFP4) | Triaged in §3.1; don't fit on 32 GiB, lack recipes, strip vision, or target other runtimes. |
| MTP speculative decoding | Head auto-skipped at load; not validated on this stack (§5.3). |
| Parallel tool calls via Responses API | #39584 crashes streaming path; Chat Completions sidesteps. |
| `--enable-chunked-prefill` flag | Implicit — default on, auto-forced by `--enable-prefix-caching`. |
| TurboQuant low-bit KV | vLLM PR #39931 open; realistic merge mid-May 2026. |

---

## 11. Validation status

210-prompt corpus served 2026-04-25 (Hebrew + ancient-Egyptian + emoji, 0% true garbling). 2026-04-27 production-flag run (container `qwen36` on `127.0.0.1:8001`, §8.2 flags) closed B4 / B6 / B7 — artifacts under `/tmp/qwen36_research/validation_2026-04-27/`.

| ID | Path | Status | Coverage |
|---|---|---|---|
| **B1** | Tool-bearing requests | **Partial** | 5-turn agentic round-trip of `tool_calls[]`; parallel tools, deeply nested params, `anyOf` not yet a dedicated corpus. |
| **B2** | `<tool_call>`-in-`<think>` | **Detector-validated** | §7.5 emits structured WARNING; retry policy belongs to the agent. |
| **B3** | Multi-turn `reasoning_content` round-trip | **Validated** | PRESERVE/STRIPPED/CORRUPTED arms distinguishable; 5-turn round-trip. |
| **B4** | Vision input | **Validated** | 2 real images correctly described (`disc5_imgs/img1.png`, `img2.png`); negative control without image hallucinates unrelated content. |
| **B5** | Streaming correctness | **Validated** | `reasoning_content` round-trips across deltas. |
| **B6** | Concurrency stress | **Validated** | 4 concurrent at 30K input tokens each fully admitted (peak Running=3, peak KV usage 76.6%, no OOM); 4×10K thinking-on shows 1.50× speedup over serial (peak Running=4). |
| **B7** | Long-context retrieval | **Validated (single-needle)** | Needle recalled at 32K @ depth 10/50/90% and 60K @ depth 50%; negative control did not invent. RULER multi-needle / multi-distractor not exercised. |

What remains: B1 schema-variation corpus and B7 multi-needle RULER. Neither is gated by the patches; both depend on workload-specific fixtures.

---

## 12. Update cadence

Re-evaluate the pinned versions when:

1. **vLLM tags a `v0.19.2` final** or later — migrate to a semver-tagged image.
2. **A newer nightly passes the boot smoke test.**
3. **vLLM widens ingest to accept `reasoning_content`** — remove patch 1.
4. **PR #39772 merges** — remove patch 2.
5. **PR #40384 or #40694 merges** — remove patch 3.
6. **PR #37429 merges** — **CRITICAL**: remove patch 3 BEFORE pulling that image (tensor layout changes; patch 3's reporting view would no longer be coherent).
7. **vLLM ships OpenAI-standard `reasoning_content` natively** — remove patch 4.
8. **Qwen3.6 retraining eliminates the OOD mid-think `<tool_call>` emission** — remove patch 5.
9. **vLLM issue #38182 (MTP + prefix cache) closes** — reconsider enabling MTP.

Each is tracked; none urgent.

---

## 13. File structure of this project

```
.
├── README.md                                              # this document
├── launch_with_patches.py                                 # §7.L — container entrypoint; imports the 5 patches then runpys vLLM
├── sitecustomize.py                                       # §7.S — auto-loads patches in EngineCore (and PID 1) at interpreter startup
├── monkey_patch_reasoning_field_ingest.py                 # §7.1 — accept reasoning_content on inbound assistant messages
├── monkey_patch_qwen3_coder.py                            # §7.2 — parser crash fix on truncated <parameter=
├── monkey_patch_hybrid_kv_allocator.py                    # §7.3 — hybrid-KV scheduler-budget fix (PR #40384 backport)
├── monkey_patch_reasoning_field_egress.py                 # §7.4 — Pydantic serialization rename reasoning → reasoning_content
├── monkey_patch_tool_call_in_think_detector.py            # §7.5 — detect <tool_call> emitted inside <think>; structured WARNING
└── tests/
    └── test_patches_against_master.py                     # static + structural-mirror suite (runs without torch/CUDA)
```
