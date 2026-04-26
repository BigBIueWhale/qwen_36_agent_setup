# Qwen3.6 on a single RTX 5090 — production-grade agentic deployment

**Last updated: 2026-04-25.** Currently deploying `QuantTrio/Qwen3.6-27B-AWQ` (dense vision-language, AWQ INT4).
**Runtime: vLLM** (llama.cpp rejected for reasons documented in §5.4).
**Target workload: multi-tool agentic pipelines** with vision, reasoning, and preserved-thinking across turns.
**Host: single NVIDIA RTX 5090, 32 GB VRAM, Blackwell SM 12.0, Linux, CUDA 13.0, nvidia-container-toolkit v1.19.0.**

This README documents a specific, pinned, reproducible deployment. Every choice below is deliberate.

---

## ⚠️ 2026-04-25 — Deployment target changed

Previous target was `RedHatAI/Qwen3.6-35B-A3B-NVFP4`. New target: **`QuantTrio/Qwen3.6-27B-AWQ`**. The 27B dense model is more capable per forward pass than a 35B-A3B MoE that activates only ~3B per token, fits on the card cleanly, and the AWQ recipe preserves the load-bearing layers (vision, `linear_attn.in_proj_a/b`, `lm_head`, embeddings, layer 0, MTP) at BF16 while quantizing MLPs and the rest. NVFP4 builds for 27B-dense were triaged and rejected — see §3.1.

---

### The short version — load-bearing pins

| Slot | Pin |
|---|---|
| Model | `QuantTrio/Qwen3.6-27B-AWQ` at revision `9b507bdc9afafb87b7898700cc2a591aa6639461` |
| Quantization format | AWQ INT4 (gemm, group_size=128, zero_point=true), data-free calibration. Vision encoder, `linear_attn.in_proj_a/b`, all `self_attn.{q,k,v}_proj`, layer 0, embeddings, `lm_head`, MTP, norms/conv1d kept BF16 |
| Runtime | vLLM Docker image `vllm/vllm-openai@sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c` (build `0.19.2rc1.dev21+g893611813`, commit `8936118134d0547fa1cc78adab2d03edd6d3dc48`, container CUDA 12.9, PyTorch `2.11.0+cu129`, FlashInfer `0.6.7`) |
| KV cache dtype | BF16 (FP8 KV scales are not shipped with this checkpoint; rationale §5.1) |
| `--max-model-len` | **65,536** |
| Disk size | 20.36 GiB |
| VRAM at boot (measured) | 19.78 GiB resident (MTP head auto-skipped via `qwen3_5.py:701-706`'s `skip_prefixes=["mtp."]`) |
| KV pool available | 6.89 GiB at gmu=0.92, all-modalities-on. Boot log says `GPU KV cache size: 28,224 tokens` (under-reported by §6.7's bug; with patch 2 installed → ~113K) |
| Per-token attention KV at BF16 | `4 KV heads × 256 head_dim × 2 (K+V) × 16 attn layers × 2 bytes = 65,536 bytes/token` (16 GiB at the model's native 262K context) |
| `preserve_thinking` | Set as server-wide default via `--default-chat-template-kwargs '{"preserve_thinking": true}'` |
| MTP speculative decoding | OFF (head present in the AWQ checkpoint as BF16 but auto-skipped at load; §5.3) |
| Patches in this repo | 7 strict, fail-loud Python monkey-patches (§7), all server-side, loaded before `vllm serve` starts. Zero client-side code |

---

## 1. What this project is

A production deployment of **`QuantTrio/Qwen3.6-27B-AWQ`** (a 27-billion-parameter dense vision-language model from Alibaba, AWQ INT4 quantized) served via **vLLM** behind an OpenAI-compatible HTTP API at `http://localhost:8000/v1/chat/completions`, intended for agentic coding via the Qwen Code CLI on a single RTX 5090.

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

**Why this quant over published NVFP4 alternatives**: every Qwen3.6-27B NVFP4 quant we triaged either (a) won't fit on 32 GiB (huginnfork / kaitchup / igf-oeaw / lkk688 are all ≥26 GiB on disk because they preserve full `linear_attn` BF16 + everything else BF16 + only quantize MLPs), (b) has no documented recipe and no community validation (Benasd / reinforce20001 uploaded 2026-04-26 with no README), (c) strips vision (sakamakismile-Text), or (d) targets a different runtime (mlx, gguf, sglang). QuantTrio AWQ is the only option that simultaneously fits on the card, preserves the load-bearing layers, works with vLLM's `qwen3_coder` tool parser, and has been empirically validated on the target hardware (210-prompt corpus, 0% true garbling). Full triage at `/tmp/qwen36_research/qwen36_27b_nvfp4_full_triage_2026-04-26.md`.

### 3.2 Runtime

| Field | Value |
|---|---|
| Runtime | vLLM |
| Docker image (published) | `vllm/vllm-openai:nightly-8936118134d0547fa1cc78adab2d03edd6d3dc48` |
| Image digest (amd64) | `sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c` |
| Underlying vLLM commit | `8936118134d0547fa1cc78adab2d03edd6d3dc48` (`0.19.2rc1.dev21+g893611813`) |
| CUDA toolkit inside image | 12.9.86 |
| Image PyTorch | `2.11.0+cu129` |
| Image FlashInfer | `0.6.7` |
| `transformers` pinned inside image | 5.5.4 (this nightly does not exhibit the §6.6 import regression) |

This is **yesterday's** nightly relative to the original deployment date. The most recent nightly on 2026-04-21 (`nightly-b47840019e61a3983c8144066a99c843d177947d`) ships a broken `transformers==5.5.4` import that blocks `vllm serve` at boot — see §6.6. Image digest is volatile; nightly tags may be re-pushed in place, so we pin by digest.

### 3.3 Client

This repo does not ship or pin a client and does not require any client-side code. The deployment exposes an OpenAI-compatible HTTP API at `/v1/chat/completions` that any off-the-shelf OpenAI-compatible client connects to directly.

**Verified-interoperable clients** (connect unmodified; no shims, no post-processors):

- **OpenAI Python SDK** (`openai.OpenAI(base_url=…).chat.completions.create(…)`).
- **Qwen Code CLI** (TypeScript, Alibaba — `qwen-code/packages/core/`).
- **Qwen-Agent** (Python, Alibaba — `Qwen-Agent/qwen_agent/llm/oai.py`).
- Any other OpenAI Chat Completions client that reads `choices[i].message.reasoning_content` / `choices[i].delta.reasoning_content` and writes `message.reasoning_content` on outgoing assistant turns.

The three wire-level interop hazards that historically made third-party clients silently lose data against vLLM — §6.1, §6.4, §6.5 — are closed server-side by the §7 patches.

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

`--max-model-len 65536`. The byte budget on a 32 GiB card after weights + activations + vision profiling reservation leaves ~6.89 GiB for KV at gmu=0.92. At per-token attention KV of 65,536 bytes/token for 27B (4 KV heads × 256 head_dim × 2 (K+V) × 16 attn layers × 2 bytes), 65,536 tokens is comfortable headroom. Boot log reports `GPU KV cache size: 28,224 tokens` (under-reported by ~4× due to the §6.7 hybrid-KV bug; with patch 2 installed → ~113K concurrent). The model's native 262K context would require 16.0 GiB of attention-only KV — does not fit on this card without dropping precision.

`--gpu-memory-utilization 0.92` is the empirical knee: lower wastes pool, higher risks Triton-warmup OOM.

### 5.3 MTP speculative decoding: OFF

The MTP head ships in the QuantTrio AWQ checkpoint as BF16 (~0.68 GiB) but is auto-skipped at load via `qwen3_5.py:701-706`'s `skip_prefixes=["mtp."]`. Not exercised on the 27B-AWQ deployment. To enable: append `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` to the launch line. **Not validated on this stack** — vLLM issues #36872 and #38182 still apply in principle (MTP gibberish on quantized weights, prefix-cache hit rate regression).

### 5.4 Runtime: vLLM, not llama.cpp

Re-verified against `ggml-org/llama.cpp` master `0d0764df` (2026-04-22 10:52 PDT):

| Dimension | vLLM | llama.cpp (verified 2026-04-22) |
|---|---|---|
| Dedicated Qwen3 tool parser | Yes (`qwen3_coder`, 683 lines) | **No.** `common/chat.cpp` contains no Qwen3-era handler; Qwen3.x routes through generic `peg-native` parser, which has live OPEN failure modes #20260, #20837, #21771, #22240. No merged fix PR for any of these. |
| Dedicated reasoning parser | Yes (`qwen3`) | Generic `--reasoning-format deepseek`; no Qwen3-specific handling. The tool-call-in-think rescue our §7.5 patch implements has no upstream analog. |
| MTP head | Loaded (disabled by §5.3) | **Dropped at GGUF conversion** (`convert_hf_to_gguf.py:4781-4782`: *"ignore MTP layers for now"*). PR #20700 unmerged WIP. |
| Vision preprocessing | BICUBIC (matches HF `Qwen3VLImageProcessor`) | **BILINEAR** hardcoded at `tools/mtmd/clip.cpp:1357` for the QWEN3VL projector type. No upstream PR open. |
| Agentic prompt-cache stability | Stable on our stack | Live regression llama.cpp #21383 (OPEN, 2026-04-03): Qwen3.5-27B CUDA illegal-memory-access under `--cache-ram` + agentic tool-call loop. |
| CUDA / Ollama compatibility | Pinned CUDA 12.9 in image, runs on host 13.0 | Unsloth Qwen3.6 docs: *"Do NOT use CUDA 13.2 as you may get gibberish outputs"*; *"Currently no Qwen3.6 GGUF works in Ollama due to separate mmproj vision files."* |
| Qwen official endorsement | Launch command in HF model card | Mentioned as "supported" without a pinned command |

Bottom line: every historical wire-level advantage llama.cpp held for this model is closed by a patch in this repo. The tool-parsing, MTP-drop, vision-preprocessing, and agentic-prompt-cache gaps above are live on master with no merged fix.

### 5.5 API endpoint: `/v1/chat/completions`, not Responses API

vLLM issue #39584 asserts `len(tool_calls)==1` in the Responses API streaming path, which crashes on legitimate parallel tool calls. Chat Completions is universally supported across every Python and TypeScript OpenAI-SDK client; using it costs nothing and sidesteps the bug.

### 5.6 Sampling parameters

Per the Qwen3.6 model card's "Best Practices" block:

For thinking-mode agentic use:
```
temperature: 1.0
top_p: 0.95
top_k: 20
min_p: 0.0
presence_penalty: 1.5
repetition_penalty: 1.0
max_tokens: 16384
```

For precise coding / low-variance tool argument generation: `temperature: 0.6`, otherwise as above with `presence_penalty: 0.0`.

`presence_penalty=1.5` is Alibaba's shipped mitigation for the Qwen3-family thinking-mode loop pathology (Qwen's own LiveCodeBench surfaced 17.4% of 1,400 outputs truncated with no `</think>`, scaling to 27.5% on hard problems — measured with the precise-coding preset). It *"may occasionally result in language mixing and a slight decrease in model performance"* per Qwen's docs. A community alternative is `min_p=0.2` with `temperature=1.0` (per `janreges3` on HF `Qwen/Qwen3.5-35B-A3B` discussion #39); if you switch to `min_p=0.2` you can drop `presence_penalty` back to 0.

`max_tokens=16384` gives generous headroom against the truncation crash mode patched in §7.1. Qwen3.6 does NOT support soft `/think` or `/nothink` switches — thinking mode is controlled exclusively via `chat_template_kwargs`.

### 5.7 `preserve_thinking=true` as a server-side default

`--default-chat-template-kwargs '{"preserve_thinking": true}'` sets `preserve_thinking=true` as a server-wide default. The Qwen3.6 chat template only emits `<think>` blocks from historical assistant turns when `preserve_thinking=true`; otherwise only the most recent assistant turn retains its `<think>`. The model was RL-trained expecting reasoning to persist across turns; stripping it degrades tool-argument correctness after 2–3 turns (`badlogic/pi-mono#3325`).

`enable_thinking` is *not* set server-side — clients pass `"chat_template_kwargs": {"enable_thinking": true}` in the request body when they want thinking on.

The separate `reasoning` vs `reasoning_content` wire-format mismatch (§6.4) is closed by the §7.3 and §7.4 patches.

### 5.8 Multimodal cost accounting

- **Vision encoder weights** (~0.83 GiB) are always resident in VRAM, whether or not any request contains an image. Subsumed into the boot weights footprint.
- **Vision encoder profiling reservation** (~1.56 GiB) is paid at boot if `--limit-mm-per-prompt` admits any image or video modality. Binary switch: `image: 1` and `image: 3` pay the same tax. Setting `{image: 0, video: 0, audio: 0}` reclaims it.
- **Per-request transient activations** are small (+2–10 MiB measured for 896×896 to 1792×1792 images); absorbed by safety headroom.
- **Image and video tokens occupy the KV pool**: ~256 tokens for 512×512, ~787 for 896×896, ~3,136 for 1792×1792, up to ~16,384 per image at the processor cap. Videos scale dramatically — short clips ~750–1,500 tokens; medium clips ~10,000–20,000; long clips can exceed 50,000. Bound video size with `--mm-processor-kwargs '{"video": {"max_frames": 64, "fps": 1}}'` if accepting untrusted clips.

`--limit-mm-per-prompt '{"image": 2, "video": 1, "audio": 0}'` is **load-bearing**: omitting the flag defaults to 999 per modality, which **crashes the boot** on this nightly + SM 12.0 (post-profiling TensorRT-LLM `throwRuntimeError`). Safe range is `image ∈ {0, 1, 2, 3}`. Audio stays 0: Qwen3.6 is text+vision only.

---

## 6. vLLM issues — complete enumeration and per-issue disposition

Each issue below is classified as **A**. Runtime bug, **B**. Model OOD failure, **C**. Infrastructure bug, or **D**. Client-interop bug.

### 6.1 Issue #39056 — `<tool_call>` inside `<think>` swallowed by reasoning parser [Class B]

When the model occasionally emits `<tool_call>...</tool_call>` inside `<think>...</think>` (out-of-distribution emission), vLLM's reasoning parser swallows the markup into `reasoning_content` before the tool parser sees it. The response arrives with `tool_calls=[]` and the markup embedded in `reasoning_content` as plain text.

The parser is correct-to-contract: Qwen3.6's chat template never renders historical tool_calls inside `<think>`, the Qwen3-Coder-Next training penalizes the pattern, and Alibaba's own evaluation in `Qwen-Agent/benchmark/deepplanning` strips everything up to `</think>` before parsing. Mid-think emission is a model failure mode.

**Affects us**: yes, intermittently. **Resolution**: §7.5.

### 6.2 Issue #39584 — parallel tool calls crash Responses API [Class A]

`vllm/entrypoints/openai/responses/serving.py:1377` has a hardcoded `assert len(delta_message.tool_calls) == 1`. Legitimate parallel tool calls trip the assertion.

**Affects us**: no — we use Chat Completions. **Resolution**: none required.

### 6.3 Issue #39771 — qwen3_coder crashes on truncated `<parameter=` tag [Class A]

`vllm/tool_parsers/qwen3coder_tool_parser.py:236` uses unsafe `str.index(">")`. When the model is truncated mid-`<parameter=NAME` (before `>`), `.index()` raises `ValueError`. The exception is caught at line 320-324 and the parser returns `tools_called=False, content=raw_text` — a silent failure. Line 227, handling function names, already uses the safe `.find()`/`-1` pattern; line 236 is an internal inconsistency.

**Affects us**: potentially, whenever a response is truncated by `max_tokens` or by client disconnect mid-generation. **Resolution**: §7.1; complemented by `max_tokens=16384` keeping truncation rare.

### 6.4 `reasoning` vs `reasoning_content` response field name [Class D]

vLLM uses the non-standard field name `reasoning` on **both** sides of the wire:

- **Egress**: responses populate `choices[i].message.reasoning` (non-streaming) and `choices[i].delta.reasoning` (streaming). The OpenAI-standard field name is `reasoning_content`.
- **Ingest**: on replay of prior assistant turns, vLLM reads `message.reasoning` only (`vllm/entrypoints/chat_utils.py:1519`) and ignores `message.reasoning_content`.

Any client following the OpenAI standard silently loses reasoning in **both directions** every turn — defeats `preserve_thinking` (§5.7).

**Affects us**: yes (Qwen-Agent and Qwen Code CLI both write `reasoning_content`). **Resolution**: §7.3 (egress) and §7.4 (ingest).

### 6.5 vLLM silent tool-parser failure modes [Class A — observability gap]

The `qwen3_coder` tool parser has 8 non-streaming and 6 streaming code paths where it returns `tools_called=False` with the raw `<tool_call>…` markup as `content`, or returns `tools_called=True` with invalid / truncated JSON arguments. All produce HTTP 200 with no error surfaced. vLLM emits no server-side metrics. Community evidence puts baseline frequency at 1–5% of tool-calling responses, rising to 10–20% under long context, reasoning, or speculative decoding.

**Affects us**: yes — silent failures in agent loops manifest as mysterious behavioral drift. **Resolution**: §7.6 (non-streaming) and §7.7 (streaming) turn the silent-failure class into a Prometheus counter `vllm_qwen3_coder_silent_tool_call_failures_total{failure_kind, model}`.

### 6.6 `transformers==5.5.4` broken `GenerationConfig` import in latest nightly [Class C]

The most recent published nightly Docker image on 2026-04-21, `vllm/vllm-openai:nightly-b47840019e61a3983c8144066a99c843d177947d` (digest `sha256:d39d4b0f…`), ships `transformers==5.5.4`. The statement `from transformers import GenerationConfig, PretrainedConfig` at `vllm/transformers_utils/config.py:18` raises `ImportError` at CLI boot, because `transformers._LazyModule` has not initialized the top-level `GenerationConfig` export by the time vLLM's CLI loader imports it. Downgrading `transformers` in-container to `4.57.6` lets the APIServer start, but the spawned `EngineCore` subprocess re-hits the same ImportError because of stale `sys.modules` state.

**Affects us**: avoided by pinning yesterday's nightly. **Resolution**: pin the older nightly digest. Re-evaluate when a new nightly tests clean with `docker run --rm vllm/vllm-openai@<digest> vllm --help`.

### 6.7 Issue #37121 — hybrid-KV scheduler/log under-reports concurrent capacity [Class D — observability bug]

vLLM's V1 paged KV cache manager forms one `KVCacheGroupSpec` per same-shape layer set. For Qwen3.6-27B that's 4 groups (1 full × 16 + 3 GDN × 16). The byte allocator at `vllm/v1/core/kv_cache_utils.py:1148-1169` allocates a single shared pool sized correctly. The bug is purely in *reporting*: `_report_kv_cache_config:1320-1324` and `get_max_concurrency_for_kv_cache_config:802-820` divide by `len(kv_cache_groups)` (4 for our model), so the displayed `GPU KV cache size: X tokens` and `Maximum concurrency` are ~4× understated. `scheduler.max_num_kv_tokens` is also affected but only consumed inside `if vllm_config.parallel_config.enable_return_routed_experts:` — off by default — so admission behavior is unchanged.

For the 27B-AWQ build: boot log says `GPU KV cache size: ~28K tokens`; with patch 2 installed → `~113K tokens`.

**Upstream status (2026-04-25)**: issue #37121 open since 2026-03-15. PR #40384 (narrow scheduler-reporting fix, our backport source) and competing PR #40694 both open. PR #37429 (broader byte-level redesign) blocked on RFC.

**Resolution**: backported as `monkey_patch_hybrid_kv_allocator.py` (§7.2). Cosmetic for the deployment — byte allocation and admission are correct without it.

---

## 7. The patches in this repo

Seven monkey-patches plus one container-entrypoint launcher. Every patch is **server-side**, loaded into the vLLM Python process by `launch_with_patches.py` before `vllm serve` hands over to `uvicorn`. There is no client-side code in this repo.

Each patch addresses a specific defect named in §6 and only that defect. Each strictly validates its target's structure via landmarks before touching anything, and refuses to apply on any landmark mismatch with a typed exception that names the exact landmark that failed. The patch file itself is the source of truth; this section is a contract index.

| # | File | Defect addressed |
|---|---|---|
| 1 | [`monkey_patch_qwen3_coder.py`](monkey_patch_qwen3_coder.py) | §6.3 / #39771 — `_parse_xml_function_call` crash on truncated `<parameter=` |
| 2 | [`monkey_patch_hybrid_kv_allocator.py`](monkey_patch_hybrid_kv_allocator.py) | §6.7 / #37121 (PR #40384 backport) — boot-log under-reporting |
| 3 | [`monkey_patch_reasoning_field_egress.py`](monkey_patch_reasoning_field_egress.py) | §6.4 egress — rename `reasoning` → `reasoning_content` on response serialization |
| 4 | [`monkey_patch_reasoning_field_ingest.py`](monkey_patch_reasoning_field_ingest.py) | §6.4 ingest — accept `reasoning_content` on replayed assistant messages |
| 5 | [`monkey_patch_tool_call_in_think_rescue.py`](monkey_patch_tool_call_in_think_rescue.py) | §6.1 / #39056 — rescue `<tool_call>` blocks emitted inside `<think>` |
| 6 | [`monkey_patch_extract_tool_calls_metrics.py`](monkey_patch_extract_tool_calls_metrics.py) | §6.5 non-streaming observability — Prometheus + WARNING on `markup_leak` |
| 7 | [`monkey_patch_extract_tool_calls_streaming_metrics.py`](monkey_patch_extract_tool_calls_streaming_metrics.py) | §6.5 streaming observability — per-delta `markup_leak_streaming` |
| L | [`launch_with_patches.py`](launch_with_patches.py) | Container entrypoint that imports patches 1–7 in order, runs per-patch verification, then hands off to `vllm.entrypoints.cli.main` via `runpy.run_module(alter_sys=True)` |

### 7.1 Patch 1 — `monkey_patch_qwen3_coder.py`

Replaces `Qwen3CoderToolParser._parse_xml_function_call` to use `str.find(">")` and, on a malformed `<parameter=` tag, return `None` for the whole tool call rather than raising `ValueError` (which the upstream `try/except Exception` collapses into "drop all N tool calls in this response"). Sibling well-formed tool calls in the same response are preserved. The dropped call's markup leaks to `content`, where §7.6 / §7.7 surface it as `markup_leak`. **Removal trigger**: PR #39772 merges.

### 7.2 Patch 2 — `monkey_patch_hybrid_kv_allocator.py`

Replaces `get_max_concurrency_for_kv_cache_config` and `_report_kv_cache_config` in `vllm.v1.core.kv_cache_utils`. Both reporting sites divide by `len(kv_cache_groups)` (4 for our model: 1 attn + 3 GDN), making displayed token capacity ~4× understated. Patched to filter `kv_cache_groups` to token-capacity-contributing specs only (`AttentionSpec` always; `MambaSpec` only when `cache_config.mamba_cache_mode == "all"`). Boot log `GPU KV cache size` goes from `~28K tokens` to `~113K tokens` on the 27B-AWQ build. Backport semantics, not literal port — the pinned commit's `MambaSpec.max_memory_usage_bytes` signature is `(self, vllm_config)`, not the master signature PR #40384 targets. **Removal trigger**: PR #40384 or PR #40694 merges.

### 7.3 Patch 3 — `monkey_patch_reasoning_field_egress.py`

Installs Pydantic v2 `serialization_alias = "reasoning_content"` on the `reasoning` field of `ChatMessage` and `DeltaMessage`, flips `model_config["serialize_by_alias"] = True`, deletes the cached `__pydantic_core_schema__` / `__pydantic_validator__` / `__pydantic_serializer__`, and calls `model_rebuild(force=True)`. Every response-serialization path emits `reasoning_content` on the wire. The internal Python attribute stays `.reasoning` — `serving.py:1036-1037` reads it as an attribute. **Removal trigger**: vLLM ships native `reasoning_content` on Chat Completions.

### 7.4 Patch 4 — `monkey_patch_reasoning_field_ingest.py`

Wraps `vllm.entrypoints.chat_utils._parse_chat_message_content`. When `role == "assistant"`, `reasoning is None`, and `reasoning_content` is a non-None string, synthesises a shallow copy with `reasoning` populated from `reasoning_content`. Both fields present with **different** values raises `ReasoningFieldAmbiguityError` (HTTP 400). Identical values pass through. Non-string `reasoning_content` (dict, list) refuses rather than being stringified. **Removal trigger**: vLLM widens ingest to accept `reasoning_content`.

### 7.5 Patch 5 — `monkey_patch_tool_call_in_think_rescue.py`

Wraps `Qwen3ReasoningParser.extract_reasoning` (non-streaming) and `.extract_reasoning_streaming` (streaming). Moves any `<tool_call>...</tool_call>` markup the model emits INSIDE `<think>...</think>` out of reasoning and into content, so the downstream `Qwen3CoderToolParser` can see it.

The streaming path is architecturally load-bearing: `parse_delta` gates the tool-parser handoff on `self.is_reasoning_end(delta_token_ids)` (requires `</think>` in the current delta's tokens). Mid-`<think>`, that predicate is False, so a `DeltaMessage(content="<tool_call>…</tool_call>")` returned from the reasoning parser goes directly to the wire without ever reaching `extract_tool_calls_streaming`. The patch solves it with **deferred flush**: completed rescue blocks accumulate in a per-instance pending list; on the handoff delta the concatenation is prepended to upstream's `content` so the tool parser parses each complete block in a single pass.

**Removal trigger**: vLLM adopts a reasoning parser that re-routes tool-call markup itself.

### 7.6 Patch 6 — `monkey_patch_extract_tool_calls_metrics.py`

Wraps `Qwen3CoderToolParser.extract_tool_calls` (non-streaming). Fires server-side observability iff the result has the silent-failure shape (`tools_called is False` AND `model_output` contains `<tool_call>` / `<function=` / `<parameter=`); returns the upstream result unchanged. Emits Prometheus counter `vllm_qwen3_coder_silent_tool_call_failures_total{failure_kind="markup_leak", model=...}` plus a structured WARNING. Observation-only — upstream call is OUTSIDE the instrumentation `try/except`. **Removal trigger**: vLLM ships first-class metrics for tool-parser silent failures.

### 7.7 Patch 7 — `monkey_patch_extract_tool_calls_streaming_metrics.py`

Wraps `Qwen3CoderToolParser.extract_tool_calls_streaming`. Per-delta detection of `markup_leak_streaming` (non-empty `content` containing a complete marker AND empty `tool_calls`). Shares the counter with §7.6 via `failure_kind` label; on collision (already-registered) looks up the existing collector via `prometheus_client.REGISTRY._names_to_collectors` and reuses it. Partial markers do not fire — cross-delta accumulation is not implemented (acceptable observability noise, not a correctness bug). **Removal trigger**: same as §7.6.

### 7.L Launcher — `launch_with_patches.py`

Container entrypoint, replacing `["vllm", "serve"]` with `["python", "/opt/patches/launch.py", "serve", ...]`. Imports every registered patch in `_PATCH_MODULES` order, runs the per-patch `_PATCH_VERIFICATION` verifier for each (re-imports the relevant vLLM target FROM SCRATCH and asserts the install took effect), then hands off to vLLM's CLI via `runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)`. Required because `PYTHONSTARTUP` does not fire under non-interactive entrypoints.

The Pydantic-schema patch (egress) adds a behavioural verification — constructs an instance and asserts `model_dump()['reasoning_content']` equals a probe value while `'reasoning'` is absent — because a tag-only check would miss a "tag stamped but serializer rebuild silently failed" regression.

**Load order** matters: `reasoning_field_egress` must come before `tool_call_in_think_rescue` because the egress patch rebuilds the `DeltaMessage` Pydantic schema and the rescue patch constructs `DeltaMessage` instances at request time.

**Known limitation — plugin-snapshot gap**: if vLLM's tool-parser registry takes a snapshot of a class before the patch ran, the verifier would pass and the served parser would still be unpatched. Mitigation: live smoke test (POST requests exercising each patched path).

---

## 8. Deployment commands

### 8.1 Fetch and pin the Docker image digest

```bash
docker pull vllm/vllm-openai@sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c

docker inspect vllm/vllm-openai@sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c \
  --format '{{.Id}} {{.Architecture}}'
```

### 8.2 Launch vLLM

```bash
docker run --rm -d --name qwen36 --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/monkey_patch_qwen3_coder.py:/opt/patches/monkey_patch_qwen3_coder.py:ro" \
  -v "$PWD/monkey_patch_hybrid_kv_allocator.py:/opt/patches/monkey_patch_hybrid_kv_allocator.py:ro" \
  -v "$PWD/monkey_patch_reasoning_field_egress.py:/opt/patches/monkey_patch_reasoning_field_egress.py:ro" \
  -v "$PWD/monkey_patch_reasoning_field_ingest.py:/opt/patches/monkey_patch_reasoning_field_ingest.py:ro" \
  -v "$PWD/monkey_patch_tool_call_in_think_rescue.py:/opt/patches/monkey_patch_tool_call_in_think_rescue.py:ro" \
  -v "$PWD/monkey_patch_extract_tool_calls_metrics.py:/opt/patches/monkey_patch_extract_tool_calls_metrics.py:ro" \
  -v "$PWD/monkey_patch_extract_tool_calls_streaming_metrics.py:/opt/patches/monkey_patch_extract_tool_calls_streaming_metrics.py:ro" \
  -v "$PWD/launch_with_patches.py:/opt/patches/launch.py:ro" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e VLLM_USE_V1=1 \
  -e PYTHONPATH=/opt/patches \
  --entrypoint python \
  vllm/vllm-openai@sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c \
  /opt/patches/launch.py serve \
  --model QuantTrio/Qwen3.6-27B-AWQ \
  --revision 9b507bdc9afafb87b7898700cc2a591aa6639461 \
  --served-model-name Qwen3.6-27B-AWQ \
  --host 0.0.0.0 --port 8000 \
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

Each flag's rationale lives in §5. Load-bearing items:
- `--limit-mm-per-prompt` — default crashes boot (§5.8).
- `--max-model-len 65536` — set by the byte budget (§5.2).
- `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` — tool calling.
- `--reasoning-parser qwen3` — server-side `<think>` extraction.
- `--default-chat-template-kwargs '{"preserve_thinking": true}'` — preserves thinking across turns (§5.7).

### 8.3 Smoke tests

**Liveness**:
```bash
curl -fs http://localhost:8000/health
# expects: HTTP 200, empty body
```

**Chat with thinking and tool schema**:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-AWQ",
    "messages": [
      {"role": "user", "content": "What is 127 * 349?"}
    ],
    "tools": [
      {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "parameters": {
          "type": "object",
          "properties": {"expr": {"type": "string"}},
          "required": ["expr"]
        }
      }}
    ],
    "temperature": 0.6,
    "max_tokens": 4096,
    "extra_body": {
      "chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}
    }
  }' | jq .
```

Expected (after the §7.3 egress patch is installed): `choices[0].message.reasoning_content` contains the `<think>` content; `choices[0].message.tool_calls[0].function.name == "calculator"` with valid JSON `arguments`; `finish_reason == "tool_calls"`.

**Operator visibility for silent tool-parser failures**: scrape `vllm_qwen3_coder_silent_tool_call_failures_total{failure_kind, model}` from the server's Prometheus endpoint. Non-zero rate indicates a residual parser silent-failure rate; investigate at the model-prompt level.

---

## 9. Known unknowns

| Item | Evidence status |
|---|---|
| Long-context retrieval quality past ~32K | Only 16 of 64 layers are full attention; long-range retrieval rides on a thin substrate. No RULER / needle-in-haystack numbers published for Qwen3.6-27B at any precision. Validate on your workload before committing. |
| `<tool_call>`-inside-`<think>` frequency, and thinking-mode loop rate under our specific AWQ config | `<tool_call>`-inside-`<think>` is community-reported at single-digit-percent. Patch §7.5 handles it; watch the `vllm_qwen3_coder_silent_tool_call_failures_total` series for residual rate. **Thinking-mode loop rate** is measured at 17.4% on Qwen3.5-35B-A3B unquantized (issue #88 on `QwenLM/Qwen3.6`); we have not measured it under our AWQ + BF16-linear-attn + BF16-KV config against a BF16-weights baseline. |
| Hybrid KV scheduler-reporting bug (§6.7) | Boot log under-reports concurrent-KV capacity by ~4×. Patch 2 corrects the log; admission behavior is identical with or without it. Tracked at vLLM #37121. |
| Image count boot failure threshold (§5.8) | `image ∈ {0, 1, 2, 3}` boots; `image ∈ {10, 100, 999}` (and the default-999 when flag omitted) crash with TensorRT-LLM `throwRuntimeError`. We did not bisect 4–9; treat 3 as the ceiling. |

---

## 10. What this deployment explicitly does not use

| Option | Status | Reason not chosen |
|---|---|---|
| llama.cpp | Not chosen | No dedicated Qwen3 tool / reasoning parser at master `0d0764df` (verified 2026-04-22); generic `peg-native` has four live OPEN failure modes. Vision resize is BILINEAR vs HF BICUBIC. Agentic prompt-cache regression on 27B-class models with `--cache-ram` is open (#21383). Full citations in §5.4. |
| SGLang | Considered, not chosen | `qwen3_coder` tool parser exists but is independent from vLLM's, and our §7 patches target vLLM's parser surface specifically. Validated SGLang boot for 27B-AWQ was blocked by a Triton dtype mismatch in `causal_conv1d_triton.py:510` (`Mismatched type for col0 between then block (<['256'], bf16>) and else block (<['256'], fp16>)`); vLLM bypasses this via `mamba_mixer2 + gdn_attention_core`. |
| Other community quants (NVFP4, GPTQ, MXFP4) | Rejected | Triaged in §3.1; either don't fit on 32 GiB, lack documented recipes, strip vision, or target other runtimes. |
| MTP speculative decoding | Disabled by default | MTP head present in checkpoint but auto-skipped at load. Append `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` to enable; not validated on this stack (§5.3). |
| Parallel tool calls via Responses API | Not used | vLLM issue #39584 crashes the Responses-API streaming path. Chat Completions sidesteps the bug. |
| `--enable-chunked-prefill` as an explicit flag | Implicit | Default on; auto-forced by `--enable-prefix-caching` on this hybrid model. |
| TurboQuant low-bit KV | Not available *yet* | Hybrid follow-up vLLM PR #39931 (open, ready-for-review). Realistic merge mid-May 2026. |

---

## 11. Remaining work for the 27B-AWQ deployment

The 27B-AWQ deployment booted and served a 210-prompt corpus on 2026-04-25 (Hebrew + ancient-Egyptian + emoji, 0% true garbling), but two patch-level fixes and seven validation paths are still open.

### 11.1 Patches to implement before declaring "production"

**A1 — Fix `monkey_patch_reasoning_field_egress.py` nested-serialization bug.** The current patch installs `serialization_alias = "reasoning_content"` on `ChatMessage.reasoning` and `DeltaMessage.reasoning`, and flips `model_config["serialize_by_alias"] = True` on those classes. This works when either class is serialized standalone. **It does not work when the class is serialized nested inside the parent response wrapper** — Pydantic v2's parent serializer (`ChatCompletionResponseChoice`, `ChatCompletionStreamResponse`) does not honor the inner class's `serialize_by_alias` setting. Result: the wire still emits `"reasoning":` instead of `"reasoning_content":` for the nested cases, which is the path real clients hit.

Fix scope: add `serialize_by_alias = True` to `ChatCompletionResponseChoice` and `ChatCompletionStreamResponse` (and any other wrapper that contains `ChatMessage` or `DeltaMessage`); audit the pinned commit for any other wrapper classes; extend the patch's verification phase (currently only constructs a standalone `ChatMessage` and asserts `model_dump()['reasoning_content']` is set) to construct a full nested response and assert the same property end-to-end. Estimated time: ~30–60 min including verifier extension. The bug is currently masked by Qwen Code CLI's `reasoning_content ?? reasoning` fallback, but any strict OpenAI-spec client (including official Python SDK in some configurations) sees the wrong field name.

**A2 — Decide on `monkey_patch_qwen3_coder.py` (currently disabled).** The launcher's test variant `launch_no_p1.py` disables this patch because the upstream `Qwen3CoderToolParser._parse_xml_function_call` signature drifted between the version the patch targeted and the version in our pinned image (`vllm/vllm-openai@sha256:baaf5fc76b…`). The patch's import-time landmark check refuses on signature mismatch. Three options:

1. **Verify upstream silently fixed #39771 in our pinned image.** Read `vllm/tool_parsers/qwen3coder_tool_parser.py:_parse_xml_function_call` at the pinned commit. If the unsafe `str.index(">")` was replaced with `str.find(">")`/sentinel handling (PR #39772 semantics), the patch is unnecessary and should be deleted from the launcher's `_PATCH_MODULES`.
2. **Rewrite the patch for the new signature.** If the bug is still present at our pinned image but the surrounding API shape changed, update the landmark set and method signature in the patch, re-test against the new shape.
3. **Leave disabled** with a documented rationale — accept the residual silent-failure rate when generation truncates mid-`<parameter=NAME` (worst case: all tool calls in the affected response are silently dropped). Acceptable only if `max_tokens=16384` truly prevents truncation in practice on this model — which we have not measured.

Option 1 is the cheapest first step. Estimated time: ~15 min to read the source and decide.

### 11.2 Validation gaps for the 27B-AWQ deployment

The 2026-04-25 validation booted the server and ran 210 non-streaming text-only prompts. The following paths are not yet exercised on this model + this image and should be covered before declaring the deployment production-ready:

| ID | Path | What to test | Why it matters |
|---|---|---|---|
| **B1** | Tool-bearing requests | ~50-prompt corpus varying tool schemas: single tool, parallel tools, nested params, array-of-object, `anyOf` | Patches 1, 5, 6, 7 all touch the tool-call code path; none have been exercised on this model with real tool schemas |
| **B2** | Tool-call-in-think rescue | Prompts that reliably elicit `<tool_call>` inside `<think>` (system prompt that asks the model to "think and call a tool" without explicit ordering); confirm patch 5 deferred-flush state machine fires and the tool call lands in `tool_calls[]` not `reasoning_content` | Patch 5 is architecturally load-bearing for §6.1 and unproven on this specific model |
| **B3** | Multi-turn `reasoning_content` round-trip | Send a request, take the assistant's `reasoning_content` from the response, include it in the next request's assistant-turn message, send again; verify `preserve_thinking` actually preserves prior reasoning across turns | This is the **whole point** of patches 3+4 plus `--default-chat-template-kwargs '{"preserve_thinking": true}'`; without round-trip validation we don't know if the contract is met |
| **B4** | Vision input | Send images of the resolutions in §5.8 (512×512 through 2048×2048) and a multi-image prompt; verify HTTP 200, sensible response, no encoder-allocation regression | Model is VL but corpus was text-only; the BICUBIC resize, the 1.56 GiB profiling reservation, and the `--limit-mm-per-prompt` ceiling all need real-image confirmation |
| **B5** | Streaming correctness | Repeat B1 + B2 + B3 with `stream: true`; verify tool calls land correctly across deltas and reasoning splits cleanly | Qwen Code CLI uses streaming; patches 5 and 7 have streaming-specific code paths unproven on this model |
| **B6** | Concurrency stress | Two concurrent 60K-token requests; verify both complete without eviction and total VRAM stays below ceiling | Validates the actual concurrent capacity at the chosen `--max-model-len 65536` and confirms no patch-2-induced over-admission OOM |
| **B7** | Long-context retrieval quality | Needle-in-haystack at 32K and 64K tokens; verify retrieval accuracy at the boundary of the byte-budget wall | The 27B model has only 16 of 64 layers carrying full attention KV; long-range retrieval rides on a thin substrate |

A reasonable order: B1 → B5 (extend to streaming) → B3 (round-trip) → B2 (rescue) → B4 (vision) → B6 (concurrency) → B7 (retrieval). B1 builds on the existing `/tmp/qwen36_research/vllm_22087_validation/` scaffolding (corpus generation, response collection, garbling detection); B7 is the one that wants a separate eval harness.

### 11.3 What "production-ready" means for this deployment

Production-ready = (A1 + A2 resolved) AND (B1 + B2 + B3 + B5 passing). B4 is required only if the agentic workload sends images. B6 and B7 are nice-to-have hardening but not blocking for a single-user Qwen Code CLI workflow.

---

## 12. Update cadence

Re-evaluate the pinned versions in this README when any of the following happens:

1. **vLLM tags a `v0.19.2` final** or later — migrate from the nightly image to a semver-tagged image.
2. **A newer nightly image passes the `docker run --rm <image> vllm --help` smoke test** (i.e., does not exhibit the §6.6 transformers import regression). Upgrade the pinned digest.
3. **Upstream PR #39772 merges** — remove `monkey_patch_qwen3_coder.py` (patch 1) and its bind-mount; remove the launcher's entry and verifier. The patch's source-landmark check refuses against a fixed function and tells the operator exactly this.
4. **vLLM issue #38182 (MTP + prefix cache) closes with a verified fix** — reconsider enabling MTP speculative decoding on this stack.
5. **vLLM PR #40384 (narrow hybrid-KV scheduler-reporting fix) merges** — or competing PR #40694 — remove `monkey_patch_hybrid_kv_allocator.py` (patch 2) and its bind-mount; remove the launcher entries.
6. **vLLM PR #37429 (broader hybrid-KV byte-level redesign) merges** — would give Mamba/DeltaNet its own dedicated `MambaPool`. **Critical sequencing**: remove patch 2 BEFORE pulling an image that contains #37429 — the broader fix changes the underlying tensor layout, and patch 2's reporting view would no longer be coherent.
7. **vLLM ships OpenAI-standard `reasoning_content` on ChatMessage / DeltaMessage natively** — remove `monkey_patch_reasoning_field_egress.py` (patch 3) and its bind-mount. Concurrent: if upstream also widens ingest acceptance to include `reasoning_content`, remove `monkey_patch_reasoning_field_ingest.py` (patch 4) in the same pass.
8. **vLLM adopts a reasoning parser that re-routes `<tool_call>` markup itself** — remove `monkey_patch_tool_call_in_think_rescue.py` (patch 5). No upstream PR is tracking this today.
9. **vLLM ships first-class metrics for tool-parser silent failures** — remove `monkey_patch_extract_tool_calls_metrics.py` (patch 6) AND `monkey_patch_extract_tool_calls_streaming_metrics.py` (patch 7).

Each of these is tracked and none are urgent.

---

## 13. File structure of this project

```
.
├── README.md                                                      # this document
├── launch_with_patches.py                                         # §7.L — container entrypoint; imports patches 1–7 then runpys vLLM
├── monkey_patch_qwen3_coder.py                                    # §7.1 — parser crash fix on truncated <parameter=
├── monkey_patch_hybrid_kv_allocator.py                            # §7.2 — hybrid-KV scheduler-budget fix (PR #40384 backport)
├── monkey_patch_reasoning_field_egress.py                         # §7.3 — Pydantic serialization rename reasoning → reasoning_content
├── monkey_patch_reasoning_field_ingest.py                         # §7.4 — accept reasoning_content on inbound assistant messages
├── monkey_patch_tool_call_in_think_rescue.py                      # §7.5 — rescue <tool_call> emitted inside <think> (streaming + non-streaming)
├── monkey_patch_extract_tool_calls_metrics.py                     # §7.6 — non-streaming markup-leak observability
└── monkey_patch_extract_tool_calls_streaming_metrics.py           # §7.7 — streaming markup-leak observability
```

Eight files, all Python, all server-side. Seven runtime monkey-patches plus one container entrypoint that loads them in the required order. **No `client/` subtree; no client-side code.** Off-the-shelf OpenAI-compatible clients (Qwen Code CLI, Qwen-Agent, OpenAI Python SDK) connect to the patched server unmodified. The docker run command (§8.2) and smoke tests (§8.3) live inline.
