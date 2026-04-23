# Qwen3.6-35B-A3B on a single RTX 5090 — production-grade agentic deployment

**Last updated: 2026-04-21** (model released 2026-04-16, five days before this write-up).
**Runtime: vLLM** (llama.cpp rejected for reasons documented in `#runtime-choice`).
**Target workload: multi-tool agentic pipelines** with vision, reasoning, and preserved-thinking across turns.
**Host: single NVIDIA RTX 5090, 32 GB VRAM, Blackwell SM 12.0, Linux, CUDA 13.0, nvidia-container-toolkit v1.19.0.**
**Deployment status: booted and validated on the target hardware on 2026-04-21.** Measured peak VRAM, measured concurrent-KV capacity, and live chat-completion sanity-check are documented in `#memory-math` and `#smoke-tests` — not estimates.

This README documents a specific, pinned, reproducible deployment. Every choice below is deliberate — nothing is default, nothing is incidental. If you fork this setup for different hardware or workloads, read the "why" before changing the "what".

### The short version — load-bearing pins (every number below measured on the target hardware)

| Slot | Pin |
|---|---|
| Model | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` at revision **`e850c696e6d75f965367e816c16bc7dacd955ffa`** |
| Quantization format | **NVFP4 = NVIDIA Floating Point 4** (W4A4: 4-bit FP weights, block-scaled with FP8 E4M3 per 16-element group; 4-bit FP activations at compute time). Vision encoder is kept at BF16 (unquantized). MTP head ships in the checkpoint as a separate shard but is never loaded. |
| Runtime | vLLM Docker image `vllm/vllm-openai@sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c` (tag `nightly-8936118134d0547fa1cc78adab2d03edd6d3dc48`, container CUDA **12.9**, PyTorch **`2.11.0+cu129`**, FlashInfer **`0.6.7`** — runs on our host CUDA 13.0 driver via forward-compat) |
| MoE backend | `flashinfer_cutlass` (native SM 12.0 NVIDIA-FP4 tensor-core GEMM via `cutlass_scaled_fp4_mm_sm120a`; gated by `ENABLE_NVFP4_SM120`, landed in vLLM PR #33417) |
| KV cache dtype | BF16 (explicitly not FP8 — reasoning-quality rationale in §5.1) |
| `--max-model-len` | **131,072** |
| Concurrent-KV pool (measured at boot, 2026-04-22) | **63,360 tokens / 4.88 GiB** at BF16 KV, MTP off, with `--limit-mm-per-prompt '{"image": 2, "video": 1, "audio": 0}'`. Dropping all multimodal to zero (`{"image": 0, "video": 0, "audio": 0}`) recovers the vision-encoder profiling reservation and raises the pool to **83,424 tokens / 6.44 GiB** (+32%). If MTP is enabled the pool drops to **39,072 tokens / 3.30 GiB** — a 38% reduction. |
| Why the pool is ~4× smaller than the attention math alone would imply | The model has 40 decoder layers of which only 10 (the full-attention ones) consume per-token KV. vLLM's hybrid-KV allocator currently reserves pool space for **all 40 layers** at the padded attention page size, so 75% of the pool is held for layers that don't need it. This is a known, open upstream bug: [vLLM issue #37121](https://github.com/vllm-project/vllm/issues/37121), candidate fix in PR #37429 (blocked pending RFC-level review). SGLang does not have this bug and would give ~3–4× larger concurrent-KV on the same weights. We stay on vLLM for the dedicated `qwen3_coder` tool parser and `flashinfer_cutlass` NVFP4 MoE path; see §6.15. |
| Idle VRAM after boot (measured) | **28,494 MiB / 32,607 MiB = 27.83 GiB used** at boot, ~30,386 MiB after the KV block pool becomes committed by the first request |
| Peak VRAM under a 119,907-token single-request prompt (measured) | **30,304 MiB / 32,607 MiB = 29.59 GiB used, 2.25 GiB absolute safety margin.** Cross-validated 2026-04-22 with a fresh boot sweep: all prompts within `--max-model-len` peak near 30,396 MiB regardless of text size or image presence — the KV pool is **pre-allocated at boot**, not grown per request. Image-bearing prompts add only +2–10 MiB of transient vision-encoder activations. See §5.8 for the full multimodal cost accounting. |
| MTP speculative decoding | **OFF.** Measured: MTP-off 153 tok/s vs MTP-on 166 tok/s steady-state — a ~8% speedup for a 38% concurrent-KV cut. Stability preferred over a marginal decode gain. See §5.3. |
| `preserve_thinking` | Set as server-wide default via `--default-chat-template-kwargs '{"preserve_thinking": true}'`; no per-request injection needed |
| Patches in this repo | 7 strict, fail-loud Python monkey-patches (§7), all server-side, loaded into the vLLM Python process before `vllm serve` starts serving traffic. Zero client-side code. Qwen Code CLI, Qwen-Agent, the OpenAI Python SDK, and any other OpenAI-compatible client connect unmodified. None require a vLLM fork. Each patch is opinionated, type-annotated, landmark-validated, and refuses to apply on any landmark mismatch. |
| Alternate quant (AWQ fallback) | `QuantTrio/Qwen3.6-35B-A3B-AWQ` is supported — exact flag swap in §10. Rejected as primary because its INT4 Marlin kernel path on SM 12.0 is emulated rather than using native Blackwell FP4 tensor cores, and because it ships without a published calibration dataset or quality-recovery benchmark. |

---

## 1. What this project is

A production deployment of **Qwen/Qwen3.6-35B-A3B** (a 35-billion-parameter Mixture-of-Experts model from Alibaba, with ~3 billion parameters active per token) served via **vLLM** behind an OpenAI-compatible HTTP API at `http://localhost:8000/v1/chat/completions`.

The deployment is opinionated toward five simultaneous correctness requirements:

1. **Tool calling that actually works in multi-turn agent loops** (no silent failures that derail the loop).
2. **Preserved thinking across tool turns** — the Qwen3.6 training contract requires historical `<think>` blocks to remain visible to the model on subsequent turns.
3. **Vision input at full preprocessing fidelity** — no resize-algorithm drift, and the vision encoder is kept at BF16 (not quantized alongside the text weights).
4. **131,072-token `--max-model-len` at BF16 KV cache precision** (16-bit, not 8-bit — see §5.1 for why FP8 KV is rejected). Concurrent-KV capacity at this config is 63,360 tokens; a single request can stretch to the full 131K via block-on-demand allocation (verified with a 119,907-token request).
5. **Stability over a marginal throughput gain** — MTP speculative decoding is disabled. We measured MTP-on at 166 tok/s vs MTP-off at 153 tok/s (~8% speedup) and it would cost 38% of concurrent-KV capacity; not worth it for this workload. See §5.3 for the numbers.

All five are non-negotiable. Every software pin, every launch flag, every server-side monkey-patch below exists to uphold them simultaneously.

---

## 2. Hardware

| Component | Spec | Why it matters |
|---|---|---|
| GPU | NVIDIA RTX 5090, 32 GB VRAM | The largest consumer Blackwell card. 32 GB is the binding constraint: it dictates that we quantize text+activations to 4-bit NVFP4, and that we cap `--max-model-len` at 131,072 (not the model's native 262,144). |
| GPU compute capability | SM 12.0 (Blackwell consumer) | Native FP4 tensor cores. CUTLASS NVFP4 MoE kernels (`cutlass_scaled_fp4_mm_sm120a`) are compiled in the vLLM image we pin, routed via the FlashInfer dispatch path. See `#runtime-image` for the verified pin. |
| Host OS | Linux (not WSL2) | On WSL2, dxgkrnl does not expose Blackwell's native FP8 tensor cores, which would invalidate half the FP8-related guidance. Linux native is required for the quoted performance figures. |
| Host CUDA | 13.0 (runtime) | Our host driver-runtime pair. The image's PyTorch was linked against CUDA 12.9 — it runs on CUDA 13.0 via NVIDIA's forward-compat layer. Avoid host setups that stray to an unvetted CUDA minor version; pin to 13.0 unless you have evidence of a newer one working. |
| Host driver | NVIDIA Linux driver ≥ 580.65.06 | Minimum for CUDA 13.0 forward-compatibility. Recommended: 580.95.05 (current R580 LTS, 2026-04). |
| nvidia-container-toolkit | 1.19.0 (released **2025-03-12**, the current stable line) | Provides `--gpus all` GPU passthrough to Docker. Older versions may lack Blackwell device enumeration fixes. |

### 2.1 Why not larger GPUs or multi-GPU

We designed for a single 32 GB card deliberately. Qwen's official BF16 weights are ~70 GB and their FP8 build is ~35 GB — both require at least a 48 GB card or multi-GPU tensor parallelism. A single 5090 with 4-bit NVFP4 is the smallest realistic desktop-class setup that preserves the vision encoder at BF16, and staying on one GPU avoids an entire class of NCCL / expert-parallelism bugs that Blackwell vLLM has open against it.

Some public setups for this model (e.g., the configuration shared in `RedHatAI/Qwen3.6-35B-A3B-NVFP4` discussion #6) run across two GPUs with `--tensor-parallel-size 2 --enable-expert-parallel` and stretch to the native 262,144 context via aggressive FP8 KV. That is not our path. We want a single-GPU deployment with BF16 KV for reasoning fidelity, and we accept 131,072 as the context ceiling in exchange.

---

## 3. Software pins (exact)

All versions are pinned for reproducibility. Floating tags (`latest`, `main`, `nightly`) are not used.

### 3.1 Model

| Field | Value |
|---|---|
| HuggingFace repo | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` |
| Revision (commit SHA) | **`e850c696e6d75f965367e816c16bc7dacd955ffa`** — pinned. Fetched via `curl -s https://huggingface.co/api/models/RedHatAI/Qwen3.6-35B-A3B-NVFP4 \| jq -r .sha` on 2026-04-21. Initial commit `67eb1625` was 2026-04-17; last update 2026-04-20. The RedHatAI card marks the release "preliminary (and subject to change)" — pinning by SHA protects us from silent upstream edits. |
| Published | 2026-04-17, four days before this deployment (one day after the base model's public release) |
| Files in the repo | `model.safetensors` (20.91 GiB, quantized text) + `model_mtp.safetensors` (1.57 GiB, MTP head — **never loaded**, see §5.3) + `model_visual.safetensors` (853 MiB, BF16 vision encoder) + configs. **Total on-disk: 23.34 GiB.** |
| Actual VRAM footprint of weights after vLLM load | **21.88 GiB measured** (all shards minus the auto-skipped MTP head, confirmed at live boot on 2026-04-21) |
| Text + activation quantization | **NVFP4 W4A4** — 4-bit FP4 (E2M1) weights with per-group FP8 (E4M3) scale at group_size=16, plus per-tensor global scale. Activations are also FP4 at compute time. |
| Vision encoder dtype | **BF16 (unquantized)** — verified by reading the safetensors headers of `model_visual.safetensors`: all 333 tensors are BF16. |
| What is kept at higher precision (the `ignore` list) | 342 entries total. Composed of: (a) **110 explicit visual-block paths** (every `model.visual.blocks.N.{attn,mlp,…}`), (b) **231 explicit language-model paths** covering every layer's `linear_attn.*`, `mlp.gate`, and `mlp.shared_expert_gate`, (c) `lm_head` (exact string), (d) one regex: `re:^mtp.*`. The `recipe.yaml` that drove the quantization specifies the shorter regex set (`re:visual.*`, `re:.*mlp.gate$`, `re:.*linear_attn.*`, `re:.*shared_expert_gate$`, `re:.*embed_tokens$`, `re:.*lm_head`, `re:^mtp.*`), but `llm-compressor` materialized those regex matches into explicit paths in the saved `config.json`. There is **no** first-layer preservation regex — layer 0's MoE experts are quantized along with the rest. Keeping `linear_attn.*` in BF16 is likely load-bearing for thinking-mode loop resistance, not just a generic "preserve sensitive layers" pattern — `cpatonn` on HF `cyankiwi/Qwen3.5-27B-AWQ-4bit` discussion #2 reports *"Linear attention params are heavily prone to quantization errors and I always leave linear attention parameters at BF16"*, and the `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4` variant (linear-attn in BF16, rest INT4) was confirmed *"significantly better on the infinite loop issue"*. Our NVFP4 build inherits this protection by virtue of the ignore list; a quant that omitted `re:.*linear_attn.*` would likely loop substantially more on the same weights. |
| Calibration dataset | 256 samples from `HuggingFaceH4/ultrachat_200k`, seqlen 4096 (full calibration script published in the model card) |
| Published quality (vs BF16) | GSM8K-Platinum 96.28% vs BF16 95.62% → **100.69% recovery**. This is one of very few quantized Qwen3.6 releases with published quality numbers against the BF16 reference. |

**Why NVFP4 over AWQ on RTX 5090 (Blackwell SM 12.0)**: RTX 5090 has native FP4 tensor cores. NVFP4 W4A4 uses them directly via CUTLASS (`csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu:60` → `cutlass_scaled_fp4_mm_sm120a`, gated by the `ENABLE_NVFP4_SM120` build flag that is on in our pinned image). AWQ W4A16 on the same card is emulated through the INT4 Marlin dequant path (base `sm_80` PTX with SM 12.0 FP8 cubins for a subset of hot ops). NVFP4 is therefore:
- **Faster**: native 4-bit Tensor-Core GEMMs, roughly 2× the TFLOPS of INT4 Marlin at W4A4.
- **Smaller activation footprint**: W4A4 keeps activations at 4 bits during compute; AWQ (W4A16) dequants to BF16 activations.
- **Higher-quality calibration available**: ultrachat-256 calibration with a published quality-recovery number. AWQ variants we evaluated are either data-free (QuantTrio) or do not publish recovery numbers.

**Why this NVFP4 build and not another quant** — rejected alternatives, each with reasons:

| Candidate | Size | Why rejected |
|---|---:|---|
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` | 25.49 GB | **Runner-up.** Functional, vision-preserved, loads via `awq_marlin`. Data-free quantization, no calibration dataset, no published quality-recovery number. Kernel path is INT4 Marlin emulation rather than native SM 12.0 FP4. Kept as documented fallback in case NVFP4 hits a loader bug we haven't anticipated. |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | 27.7 GB | `compressed-tensors` packed-int4 at group_size=32; larger on disk; hits the vLLM `compressed-tensors + AWQ MoE + actorder=null` bug class (issue #35303 lineage). The `llmcompressor==0.13.1.a20260219` tag in its metadata is a pre-release. |
| `Intel/Qwen3.6-35B-A3B-int4-AutoRound` | 20.93 GB | Smallest candidate, but `ignore: []` in `quantization_config` means `linear_attn`, `mlp.gate`, `mtp`, and `visual` were all eligible for quantization. The vision tensors happen to land as BF16 due to HF `dtype=auto` rescuing them at load time — fragile, easily broken by a vLLM loader change. README recommends `--max-model-len 2048` which hints at instability. |
| `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` | 22.78 GB | README lists kept-in-BF16 modules (attention, router, shared_expert, mtp, visual) but `ignore` field in `config.json` is empty — an internal inconsistency. 97.42% calibration success with 2.58% RTN fallback is mediocre. |
| `rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bpp` | 22.92 GB | Mixed-precision recipe (NVFP4 + MXFP8 + BF16). Calibrated on only 8 multimodal samples — too few for reliable vision preservation. 67 of 111 vision Linear layers stayed BF16 by luck of calibration, not by design. Novel and unproven. |
| `caiovicentino/Qwen3.6-35B-A3B-HLWQ-CT-INT4` | 19.45 GB | **No vision tensors in the repo.** Text-only. Disqualified. |
| `Qwen/Qwen3.6-35B-A3B-FP8` (official) | 37.49 GB | Does not fit on 32 GB with any useful context. |
| `Qwen/Qwen3.6-35B-A3B` (official BF16) | ~70 GB | Does not fit on 32 GB. |

**Caveats on the chosen quant**:
- RedHatAI themselves tag the release "preliminary version (and subject to change)". **Pin by revision SHA** (fetch at deploy time) so upstream-edits don't silently change what you're serving.
- One earlier `ignore`-list bug was reported and fixed in head before 2026-04-20; no other open correctness bugs on the repo's discussion tab as of today.
- Native SM 12.0 NVFP4 kernel was first landed in vLLM PR #33417 (Jan 2026); it has been in mainline for ~3 months. Less battle-tested on consumer Blackwell than AWQ Marlin on the same hardware, but the CUTLASS path is the same code that RedHat and NVIDIA run in production on SM 10.0 (data-center Blackwell).

### 3.2 Runtime

| Field | Value |
|---|---|
| Runtime | vLLM |
| Docker image (published) | **`vllm/vllm-openai:nightly-8936118134d0547fa1cc78adab2d03edd6d3dc48`** |
| Image digest (amd64) | **`sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c`** |
| Underlying vLLM commit | `8936118134d0547fa1cc78adab2d03edd6d3dc48` (approximately two weeks of commits ahead of `v0.19.2rc0`'s `aeee7ef9`; includes every Qwen3.5-MoE NVFP4-MoE and `flashinfer_cutlass` moe-backend fix we need) |
| CUDA toolkit inside image | **12.9.86** (the container's `nvcc --version` and `/usr/local/cuda-12.9`) |
| Host CUDA runtime required | ≥ 13.0 via forward-compat. PyTorch inside the container links against CUDA 12.9, which runs fine on our CUDA 13.0 host driver — do NOT assume the image brings CUDA 13. |
| Image PyTorch | **`2.11.0+cu129`** (not a bare 2.11.0 — the `+cu129` suffix is the CUDA version it was linked against) |
| Image Triton | 3.6.0 (shipped inside the torch 2.11 wheel) |
| Image FlashInfer | **`0.6.7`** (provides `cutlass_scaled_fp4_mm_sm120a` and `FLASHINFER_CUTLASS` MoE dispatch) |
| `TORCH_CUDA_ARCH_LIST` env var | `7.0 7.5 8.0 8.9 9.0 10.0 12.0` — set as an env var inside the image so downstream source builds target SM 12.0. (`torch.cuda.get_arch_list()` for the shipped torch returns `[]` — its own prebuilt kernels carry per-op arch tags rather than a monolithic arch list.) |
| `transformers` pinned inside image | 5.5.4 (see §6.14 — this particular nightly does **not** exhibit the `GenerationConfig` import regression) |

**Why this specific nightly**: the obvious pick — the most recent nightly on 2026-04-21, `nightly-b47840019e61a3983c8144066a99c843d177947d` (`sha256:d39d4b0f5220fa64557e3bf1addb49fea8e097b2d876c3a38da58fbd5fc8a003`) — **ships a broken `transformers==5.5.4` import** that blocks `vllm serve` at boot (`from transformers import GenerationConfig, PretrainedConfig` raises `ImportError` because `_LazyModule` has not initialized when `vllm.transformers_utils.config` is imported by the CLI path). We reproduced this on the target hardware and switched to **yesterday's nightly**, which carries the same Qwen3.5-MoE + NVFP4 + `flashinfer_cutlass` code path but without the import regression. Full detail in §6.14.

**Why not `v0.19.2rc0` or stable tags**: `v0.19.2rc0` is a Git tag only — no published Docker image. `v0.19.1` (the last `-cu130` stable image, digest `sha256:cd7bf919…`) predates Qwen3.6's 2026-04-16 release by one day and is missing hundreds of commits of NVFP4 + Qwen3.5-MoE fixes. `vllm/vllm-openai:latest` is a floating tag that is unsafe for reproducibility.

**Image digest is volatile**. Docker Hub nightly tags may be re-pushed in place. We pin by digest so that `docker pull vllm/vllm-openai@sha256:9bba4628…` is byte-identical across time. If you re-deploy later, re-fetch the current nightly list and pick one that (a) has the Qwen3.5-MoE NVFP4 paths, (b) boots without the transformers import bug. The `docker buildx imagetools inspect` command in §8.1 prints the digest for any tag.

### 3.3 Client

This repo does not ship or pin a client and does not require any client-side code. The deployment exposes an OpenAI-compatible HTTP API at `/v1/chat/completions` that any off-the-shelf OpenAI-compatible client connects to directly.

**Verified-interoperable clients** (connect unmodified; no shims, no post-processors, no wrappers):

- **OpenAI Python SDK** (`openai.OpenAI(base_url=…).chat.completions.create(…)`) — direct.
- **Qwen Code CLI** (TypeScript, Alibaba — `qwen-code/packages/core/`) — direct.
- **Qwen-Agent** (Python, Alibaba — `Qwen-Agent/qwen_agent/llm/oai.py`) — direct.
- Any other OpenAI Chat Completions client that reads `choices[i].message.reasoning_content` / `choices[i].delta.reasoning_content` and writes `message.reasoning_content` on outgoing assistant turns.

The three wire-level interop hazards that historically made third-party clients silently lose data against vLLM — §6.1 (tool calls emitted inside `<think>`), §6.12 (non-standard `reasoning` field name on both ingest and egress), §6.13 (silent tool-parser failure modes) — are **closed server-side** by the §7 patches. No client-side compensation is required and none is shipped. Anything calling itself an OpenAI-compatible Chat Completions client should interoperate correctly.

---

## 4. Model architecture background (non-obvious context)

Qwen3.6-35B-A3B is not a plain transformer. Understanding its shape is prerequisite to understanding the deployment config.

- **Total parameters: ~35 B. Active per forward pass: ~3 B** (8 of 256 routed MoE experts + 1 shared expert per token).
- **40 decoder layers**, arranged as a 3:1 hybrid:
  - 30 layers are **Gated DeltaNet** (linear attention with a fixed-size recurrent state — does not grow with context length).
  - 10 layers are **full (softmax) attention** — the only layers whose KV cache grows with context. Positions: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 (every 4th layer per `full_attention_interval=4`).
- **Attention config of full layers**: `num_key_value_heads=2`, `head_dim=256`. Grouped-query attention with a very low KV-head count, which keeps KV cache small per token.
- **Rotary positional embedding**: M-RoPE with `mrope_section=[11,11,10]` and `partial_rotary_factor=0.25`. Used for both text and vision tokens. No YaRN scaling needed — the model's native 262K context is used directly.
- **MTP (Multi-Token Prediction) head**: a 1-layer MTP module (19 tensor keys under the `mtp.*` prefix) that Qwen trained during Qwen3.6 pretraining and ships inside the official BF16 release (scattered across `model-00025-of-00026.safetensors` and `model-00026-of-00026.safetensors` in `Qwen/Qwen3.6-35B-A3B`). Usable in vLLM for speculative decoding in principle. **We disable it** — see §5.3.
  - **Note on RedHatAI's `model_mtp.safetensors`**: `RedHatAI/Qwen3.6-35B-A3B-NVFP4` separates the exact same 19 MTP tensors (verified byte-identical by set equality of names and matching BF16 dtype) into a dedicated `model_mtp.safetensors` shard. RedHatAI did not train or create this module — they extracted Alibaba's weights into their own file for cleanliness and kept them unquantized via their `recipe.yaml`'s `re:^mtp.*` ignore pattern.
- **Vision encoder**: Qwen3-VL tower, 27 layers, patch size 16, spatial_merge 2, temporal_patch 2. Bundled into the main checkpoint (no separate mmproj file when loading via vLLM).

The hybrid attention pattern is why the KV-cache memory math diverges sharply from a pure-transformer model of similar size: only 10 of 40 layers contribute to KV growth with context, so context-length headroom is roughly 4× better than a dense 35B model at the same KV dtype.

---

## 5. Decisions that shape the deployment

Each subsection below states the decision, the tradeoff, and the evidence.

### 5.1 KV cache: BF16, not FP8

**Decision**: `--kv-cache-dtype auto` (which resolves to BF16 because the model dtype is BF16). We do **not** use `--kv-cache-dtype fp8`.

**Rationale**:
- FP8 KV cache halves KV memory (from ~16 KiB/token to ~8 KiB/token) but requires per-tensor or per-head scaling factors to preserve numerical range. **Qwen3.6-35B-A3B ships without calibrated FP8 KV scaling factors** — vLLM falls back to scale=1.0, which is the documented cause of mild long-context quality drift.
- SGLang's own documentation flags this: *"these FP8 checkpoints do not include pre-calibrated KV cache scaling factors; SGLang defaults to scale 1.0, which may cause noticeable accuracy degradation on reasoning-heavy tasks."*
- For a reasoning model in an agentic pipeline — where every `<think>` block feeds into a downstream tool-call accuracy — uncalibrated FP8 KV is not acceptable.
- The rotation-based low-bit KV quantization methods that would preserve quality (TurboQuant, RotorQuant) **do not support Qwen3.6's DeltaNet hybrid architecture in any mainstream runtime today**. vLLM PR #38479 ("TurboQuant: 2-bit KV cache compression with 4x capacity") was **merged 2026-04-15**, but the documentation states explicitly: *"Supports full-attention and uniform sliding-window transformer models. Hybrid architectures (mamba+attention, interleaved SWA) are planned for a follow-up PR."* Attempting to enable TurboQuant on Qwen3.5/3.6-35B-A3B emits: *"TurboQuant KV cache is not supported for hybrid (attention + Mamba) models. Boundary layer protection requires uniform attention layers."* (The superseded earlier PR #38280 was closed.)

**What we sacrifice**: half the KV memory vs FP8 (and three-quarters vs a hypothetical TurboQuant-2bit). This caps single-request context at 131,072 and concurrent-KV at ~63K on a 32 GB card.

**Revisit when**: the follow-up PR that adds hybrid-architecture support to TurboQuant lands and is validated against Qwen3.5/3.6-A3B. Until then, BF16 KV is the only honest choice.

### 5.2 Max context length: 131,072 tokens (128K), not 262K

**Decision**: `--max-model-len 131072`.

**Memory math at 128K, BF16 KV, NVFP4 weights — measured on target hardware (single RTX 5090, 32,607 MiB total per nvidia-smi, cross-validated 2026-04-22)**:

| Component | Size (measured) |
|---|---|
| Loaded NVFP4 weights (W4A4) + BF16 vision encoder — MTP head **auto-skipped** (see §5.3) | **21.88 GiB** measured at boot (model.safetensors 20.91 GiB + model_visual.safetensors 0.83 GiB + load overhead) |
| CUDA graphs + activation workspace + FlashInfer CUTLASS NVFP4 workspace | ~1.7 GiB |
| Vision encoder profiling reservation (because `--limit-mm-per-prompt` admits image or video modality) | ~1.56 GiB — measured as the delta between `{image: 2, video: 1, audio: 0}` (pool = 4.88 GiB) and `{image: 0, video: 0, audio: 0}` (pool = 6.44 GiB). This is reclaimable **only** by disabling vision entirely; any non-zero image or video limit triggers the full reservation. |
| BF16 KV cache pool allocated by vLLM at `--gpu-memory-utilization 0.92` | **4.88 GiB** (logged as `Available KV cache memory: 4.88 GiB` at boot) → **63,360 tokens** reported by vLLM as `GPU KV cache size`, translated to **`Maximum concurrency for 131,072 tokens per request: 1.85x`**. See "hybrid KV inflation" note below — the real attention-layer math would imply ~255K tokens; the 4× gap is an upstream vLLM bug. |
| DeltaNet recurrent state (30 linear-attn layers, fixed size, independent of context) | ~0.06 GiB natively — but in practice folded inefficiently into the KV pool accounting (see §6.15) |
| Safety headroom reserved by gmu=0.92 (unused at peak) | ~2.25 GiB |
| **Idle VRAM after boot (before first request)** | **27.83 GiB used / 31.84 GiB total** (28,494 / 32,607 MiB per `nvidia-smi`) |
| **Idle VRAM after first request (pool becomes committed)** | ~29.68 GiB used (30,386 MiB) — once the block pool is touched, this is the new baseline for the life of the server |
| **Peak VRAM under a real huge-prompt request** | **29.59 GiB used / 31.84 GiB total** (30,304 / 32,607 MiB at 119,907 text-only tokens; re-measured 2026-04-22 at 125,642 text-only tokens = 30,396 MiB and 123,368 tokens-with-image = 30,396 MiB. Peak is essentially constant.) |

The 29.59–29.68 GiB peak is the empirical ceiling at this configuration. **Absolute safety margin against GPU-total OOM: ~2.2 GiB.** The peak does not increase with even-larger single-request token counts because vLLM's paged KV allocator **pre-allocates its entire pool at boot**. Larger requests draw blocks from that fixed pool, so they use more *of the pool* (eviction-free up to `--max-model-len`) without expanding total VRAM.

**The hybrid KV inflation — why 63,360 tokens and not ~255,000:**

By the model's attention config (2 KV heads × 256 head_dim × 10 full-attention layers × 2 bytes × K+V = 20,480 bytes/token at BF16), 4.88 GiB of pool should hold approximately 255,000 concurrent tokens. vLLM instead reports 63,360 — roughly 4× less. The factor is explained exactly by the ratio of total decoder layers to full-attention layers: `4.88 GiB / (40 layers × 2048 bytes/token/layer) = 63,963` tokens ≈ 63,360 reported. In other words, the paged allocator reserves per-token KV block space for **all 40 decoder layers**, not just the 10 that actually carry attention KV — so 30 DeltaNet layers (which natively need only a small fixed recurrent state) are consuming 75% of the pool's capacity for nothing.

This is a known upstream bug: see §6.15 for the full treatment, the tracking issue, and why we accept it for now instead of switching runtimes.

**Concurrent-KV capacity is the real constraint, not `--max-model-len`.** The 4.88 GiB KV pool at BF16 (20,480 bytes/token × 10 full-attention layers) holds **63,360 tokens of concurrent context at once**. This means:

- A single request can stretch to the full 131,072 `--max-model-len` — vLLM will allocate the blocks for it on the fly from the unused portion of the pool.
- **Two long requests active at once will start evicting each other's blocks** once their combined context crosses 63,360 tokens. For our single-user agent workload this is irrelevant; for a multi-user service it is the binding constraint.

**Why `--gpu-memory-utilization 0.92` and not 0.90 or 0.95**: at 0.92 we measured 1.5 GiB of formal slack above peak, which is comfortable Triton-autotune headroom on cold boot. Dropping to 0.90 would shave ~640 MiB off the KV pool (≈ 32K concurrent tokens lost). Pushing to 0.95 would reclaim ~960 MiB (≈ 48K concurrent tokens) but is within one stray Triton warmup spike of OOM. 0.92 is the empirical knee.

**Why not 262K (the model's native context)**: BF16 KV at 262,144 tokens = 5.00 GiB cache — larger than the entire pool we were just allocated. We would have to drop `--gpu-memory-utilization` or switch to FP8 KV to make it fit, and the former fails to boot while the latter contradicts `#5.1`. Stretching a single-request context past 131K is also not practical without changing the knee.

**Why not a round 100K, 120K, 150K**: 131,072 is 2^17, aligns cleanly with vLLM's paged KV block size (default 16 tokens per block, so 131,072 ÷ 16 = 8,192 blocks), and matches what adjacent public configurations have tested. Intermediate values work but are not more informative.

**If you push max_model_len lower to broaden concurrent capacity**: `--max-model-len 65536` gives 63,360 concurrent tokens of near-pure headroom (more than enough for any single request at that length and leaves pool for truly parallel agents). Keep everything else the same.

### 5.3 MTP speculative decoding: disabled — we measured the tradeoff and the win is marginal

**Decision**: do **not** pass `--speculative-config`. MTP on Qwen3.6-35B-A3B-NVFP4 on this RTX 5090 **works correctly** (contrary to my initial read of the literature), but it is a poor tradeoff for our specific workload.

**What we measured on target hardware, 2026-04-21**:

| Metric | MTP OFF | MTP ON (`{"method":"mtp","num_speculative_tokens":1}`) |
|---|---:|---:|
| Steady-state decode throughput (mean of two 1.2K–1.5K-token runs, post-warmup) | **153 tok/s** (152.2 + 154.8) | **166 tok/s** (169.6 + 162.8) |
| MTP acceptance rate (vLLM's own SpecDecoding metrics) | n/a | **87–96%** (mean-acceptance-length 1.87–1.96 with 1 speculative token; theoretical ceiling at 2.00) |
| Available KV cache pool at boot | **4.88 GiB → 63,360 tokens concurrent** | **3.30 GiB → 39,072 tokens concurrent** |
| Idle VRAM after boot | **28,494 MiB** (27.83 GiB) | **28,888 MiB** (28.22 GiB) |
| Output quality on a 1500-token essay prompt | coherent | coherent (no gibberish, no repetition loops) |

**Real speedup: ~8%. Real cost: 38% of concurrent-KV capacity (24K fewer tokens across all active sequences).**

Earlier public reports predicted worse for MTP:
- vLLM issue #36872 (open): MTP + FP8/AWQ on Qwen3.5/3.6-35B-A3B → output gibberish, 0% acceptance. **We did not reproduce on NVFP4.**
- vLLM issue #38182 (open): MTP reduces prefix-cache hit rate by 21 pp.
- vLLM issue #36331 (closed): reported 0% acceptance on Qwen 3.5 **122B** NVFP4; poster noted the 35B worked — matches our measurement.
- Discussion #6 on `RedHatAI/Qwen3.6-35B-A3B-NVFP4` — user `livepeer-ren`: *"i cant make it work with latest vllm docker image i had to remove --speculative-config ... to fit it"* — we hit the same VRAM-pressure issue; with the correct pinned image the config DOES fit.
- NVIDIA Developer Forum (DGX Spark, SM 12.1): Qwen3.6-35B-A3B with MTP num_speculative_tokens=2 dropped 7000 t/s → 4000 t/s. Note they used num_speculative_tokens=2; with 1 token we see a net win of ~8%.

**Why we still recommend MTP off**:

1. **Concurrent-KV is precious on 32 GB.** Cutting the pool from 63,360 to 39,072 tokens significantly hurts agentic workloads that (a) interleave multiple short tool-result turns in the same conversation or (b) stretch single requests above 40K tokens toward our 131K max. (Single requests still succeed above 39K via block-on-demand allocation — we verified 66K and 120K single-request prompts succeed — but concurrent multi-request scenarios will see eviction earlier.)
2. **MTP head adds 1.57 GiB of weight load** plus FlashInfer JIT-cache emits new "GPU lacks shared memory resources" warnings at boot (non-fatal, but indicate MTP-specific kernels fell back to slow paths on SM 12.0).
3. **Prefix-caching interaction is untested on our stack**: the open issue #38182 predicts a 21 pp cache-hit-rate drop with MTP — we did not stress-test this but it would hurt agentic loops with static system prompts.
4. **8% speedup is not worth stability risk**. Your declared posture is correctness over throughput.

**Alternative config for readers who want MTP**: append `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` to the launch line. Measured working on this hardware. Do NOT use `num_speculative_tokens=2` — that's where the DGX Spark net-slowdown report lives.

**Revisit when**: either (a) vLLM implements an MTP variant that doesn't claim 1.57 GiB of weight VRAM upfront, or (b) #38182 closes with a verified fix for prefix-caching hit rate preservation under MTP.

### 5.4 Runtime: vLLM, not llama.cpp

**Decision**: vLLM.

**Rationale**, re-verified against `ggml-org/llama.cpp` master `0d0764df` (2026-04-22 10:52 PDT):

| Dimension | vLLM | llama.cpp (verified 2026-04-22) |
|---|---|---|
| Dedicated Qwen3 tool parser | Yes (`qwen3_coder`, 683 lines) | **No.** `common/chat.cpp` contains no Qwen3-era handler (grep for `qwen3*` returns zero hits); `docs/function-calling.md` stops at Qwen 2.5. Feature request llama.cpp #15012 was closed **stale** 2025-11-18 with no replacement PR. Qwen3.x routes through the generic `peg-native` parser, which has these live OPEN failure modes: **#20260** (fails when the model emits prose between `</think>` and `<tool_call>` — Qwen3.x reliably does this), **#20837** (XML tool call inside `<think>` halts generation), **#21771** (`p.json()` fails on `array<object>` parameter values, emits partial tool_calls), **#22240** (`anyOf` tool schemas rejected — impacts Claude-Code-style built-ins). No merged fix PR for any of these. |
| Dedicated reasoning parser | Yes (`qwen3`) | Generic `--reasoning-format deepseek`; no Qwen3-specific handling. The tool-call-in-think rescue our §7.5 patch implements has no upstream analog. |
| MTP head | Loaded (disabled by our §5.3 decision; tensors remain addressable) | **Dropped at GGUF conversion.** `convert_hf_to_gguf.py:4781-4782` on master today: `if name.startswith("mtp"): return  # ignore MTP layers for now`. PR #20700 (speculative MTP for dense Qwen3.5) remains **UNMERGED WIP**, author's comment "do not expect near-term merge". |
| Vision preprocessing | BICUBIC (matches HuggingFace `Qwen3VLImageProcessor`) | **BILINEAR.** `tools/mtmd/clip.cpp:1357` hardcodes `RESIZE_ALGO_BILINEAR` in the `PROJECTOR_TYPE_QWEN2VL`/`QWEN25VL`/`QWEN3VL` case block. HF's processor is BICUBIC. No upstream PR open to close the gap. Separately, llama.cpp #18181 (OPEN) — `--fit` params ignore vision-encoder VRAM. |
| Agentic prompt-cache stability on RTX-class CUDA | Stable on our stack | **Live regression**: llama.cpp #21383 (OPEN, filed 2026-04-03) — Qwen3.5-27B CUDA illegal-memory-access under `--cache-ram` + agentic tool-call loop; bisect window `149b2493c..f49e91787` (2026-03-19 ok, 2026-04-03 broken), no fix PR. This is exactly the `--cache-ram 4096` config Simon Willison's 2026-04-22 Qwen3.6-27B recipe uses. |
| CUDA / Ollama compatibility | Pinned CUDA 12.9 in image, runs on host 13.0 via forward-compat | Unsloth's Qwen3.6 docs state verbatim: *"Do NOT use CUDA 13.2 as you may get gibberish outputs. NVIDIA is working on a fix."* And: *"Currently no Qwen3.6 GGUF works in Ollama due to separate mmproj vision files."* |
| Qwen official endorsement | Launch command in HF model card | Mentioned as "supported" without a pinned command |

**Closed or fixed upstream since the operator last surveyed** — do not cite as llama.cpp risks anymore:
- Qwen3-Next architecture: merged via llama.cpp PR #16095 (2025-11-28).
- Gated DeltaNet op (CPU+CUDA via #19504, 2026-03-07; chunked fused GDN path via #20340, 2026-03-11).
- `tool_calls.arguments` emitted as JSON object not string (#20198): fixed by PR #20213 (2026-03-08); current `common/chat.cpp:322-326` stringifies via `args.is_string() ? args : args.dump()`.
- Qwen3.6-35B crash on `-sm tensor` state save at ~40k ctx (#22058): fixed by PR #22063 (2026-04-18).
- Server speculative checkpointing for recurrent/hybrid modules: merged via PR #19493 (2026-04-19).

**What llama.cpp used to win on (now closed by our server-side patches)**:
- Native OpenAI-standard `reasoning_content` field on the wire → closed by §7.3 and §7.4.
- Loud rather than silent tool-parser failures → operator-visible Prometheus counter `vllm_qwen3_coder_silent_tool_call_failures_total` via §7.6 and §7.7.

**Bottom line**: every historical wire-level advantage llama.cpp held for this model is closed by a patch in this repo. The tool-parsing, MTP-drop, vision-preprocessing, and agentic-prompt-cache gaps above are live on master at 2026-04-22 with no merged fix. Simon Willison's 2026-04-22 Qwen3.6-27B recipe does not exercise tool calls and runs `--no-mmproj`; it is not evidence that this workload works on llama.cpp.

**Retired from this section because prior claims were not defensible**: previous versions cited "OCR quality hit of 2–10 points on OCRBench" for BILINEAR and "113 vs ~3 commits on the vision file" — neither was backed by a citation I could retrieve. The BILINEAR mismatch is still real (code-verified above); the point-delta figure is not.

**Also retired: three issues that were earlier flagged but turn out not to be llama.cpp bugs** (closure-reason verified by reading the GitHub comment threads):
- llama.cpp #22255 ("Qwen3.6-27B `preserve_thinking` ignored") — closed by the reporter himself 2026-04-22 once he understood he needed to re-inject `reasoning_content` from each response into the next request's messages. Caller contract, not an llama.cpp bug.
- llama.cpp #20614 ("`<tool_call>` in `<think>` cache corruption") — reporter and commenters concluded the cache wasn't the problem; the real symptom maps to #20260 (already cited above).
- llama.cpp #21118 ("`</think>` not emitted when calling tools") — reporter's own final comment: "this issue turned out to be not related to llama.cpp at all"; root cause was the VS Code Copilot extension not round-tripping `reasoning_content`.

### 5.5 API endpoint: `/v1/chat/completions`, not Responses API

**Decision**: serve and consume only the Chat Completions endpoint.

**Rationale**: vLLM issue #39584 asserts `len(tool_calls)==1` in the Responses API streaming path, which crashes on legitimate parallel tool calls. Chat Completions is the universally-supported endpoint across every Python and TypeScript OpenAI-SDK client; using it costs nothing and sidesteps the bug entirely.

### 5.6 Sampling parameters

Per the Qwen3.6-35B-A3B model card's "Best Practices" block, for thinking-mode agentic use:

```
temperature: 1.0
top_p: 0.95
top_k: 20
min_p: 0.0
presence_penalty: 1.5
repetition_penalty: 1.0
max_tokens: 16384   (our recommendation, see below)
```

For precise coding / low-variance tool argument generation:

```
temperature: 0.6
top_p: 0.95
top_k: 20
min_p: 0.0
presence_penalty: 0.0
```

**Caveat on `presence_penalty=1.5`** — it is a workaround, not a clean setting. The Qwen3-family has a documented thinking-mode loop pathology that Qwen's own LiveCodeBench evaluation of Qwen3.5-35B-A3B (issue #88 on `QwenLM/Qwen3.6`) surfaced at 17.4% of 1,400 outputs truncated with no `</think>`, scaling to 27.5% on hard problems — measured using the "precise coding" preset above (i.e. `presence_penalty=0.0`). So the coding preset is *not* loop-safe either; it simply loops on different token distributions. `presence_penalty=1.5` is Alibaba's shipped mitigation and Qwen's own docs acknowledge it *"may occasionally result in language mixing and a slight decrease in model performance."* A separate community lever, `min_p=0.2` with `temperature=1.0`, is reported by user `janreges3` on HF `Qwen/Qwen3.5-35B-A3B` discussion #39 (*"vLLM - Looping prevention"*) as having *"the greatest positive impact"* on loop suppression without the language-mixing side effect — worth trying in agentic workloads where loop stability dominates over the marginal quality cost of a higher `min_p` floor. If you switch to `min_p=0.2` you can drop `presence_penalty` back to 0. The root cause appears to be RL post-training mode collapse on long thinking traces (Qwen3 tech report itself references entropy control and filtering "outputs that contain substantial repetition"); no inference-time knob is a true fix, only weight-level anti-repetition DPO would be, and no community has shipped that for Qwen3.6 as of this writing.

**Why `max_tokens: 16384` (and not smaller)**: the vLLM `qwen3_coder` tool parser has a documented crash mode (issue #39771) when generation is truncated mid-`<parameter=` tag. We patch the parser (see §7.1), but generous `max_tokens` headroom prevents truncation from firing in the first place. On a 5090 at 128K context with BF16 KV, 16K `max_tokens` costs ~0.3 GiB of reserved KV slots — comfortably within headroom. Note that the 17.4% LiveCodeBench truncation rate cited above is effectively a prior on how often a hard-coding request will *not* converge within this `max_tokens` envelope regardless of how generous the envelope is; the failure mode is loop-attractor behavior, not token-budget shortfall.

**Qwen3.6 does NOT support soft `/think` or `/nothink` switches** like some earlier Qwen revisions. Thinking mode is controlled exclusively via `chat_template_kwargs` — see below.

### 5.7 `preserve_thinking=true` as a server-side default (not per-request)

**Decision**: pass `--default-chat-template-kwargs '{"preserve_thinking": true}'` on the vLLM launch line. This sets `preserve_thinking=true` as a server-wide default applied to every incoming request unless the request explicitly overrides it with its own `chat_template_kwargs`.

**Why this matters**: the Qwen3.6 chat template only emits `<think>` blocks from historical assistant turns when `preserve_thinking=true`; otherwise, only the most recent assistant turn retains its `<think>`, and earlier ones are stripped. The model was RL-trained expecting reasoning to persist across turns, and stripping it causes tool-argument correctness to degrade after 2–3 turns (documented in `badlogic/pi-mono#3325`).

**Why the server-side default rather than per-request injection**: most OpenAI-SDK client code does not surface `chat_template_kwargs` as a first-class request field, and writing `extra_body={"chat_template_kwargs": {"preserve_thinking": true}}` at every call site is fragile. The `--default-chat-template-kwargs` flag (spotted in a production config shared in discussion #6 on `RedHatAI/Qwen3.6-35B-A3B-NVFP4`) sets `preserve_thinking=true` once at server start. Server-side defaults are merged with per-request `chat_template_kwargs`; request-specified fields override server defaults, so nothing is lost for clients that do set it.

**`enable_thinking`** is *not* set server-side — we leave it to the client. A request that wants thinking on passes `"chat_template_kwargs": {"enable_thinking": true}` in its body (`extra_body=...` via the OpenAI Python SDK).

The separate `reasoning` vs `reasoning_content` wire-format mismatch is a distinct interop bug from the `preserve_thinking` default (see §6.12); it is closed server-side by the §7.3 and §7.4 patches, so clients do not need a rename shim.

### 5.8 Multimodal cost accounting — what the vision encoder and `--limit-mm-per-prompt` actually cost you

This is the longest subsection in §5 because the interaction between the vision encoder, the profiler, the `--limit-mm-per-prompt` flag, and the KV pool is where several non-obvious costs accumulate. Everything below is either measured on the target hardware on 2026-04-22 with a 9-boot empirical sweep, or read directly from the vLLM source code on the pinned commit.

**Four costs, each distinct, each paid at a different time.** Confusing these is the source of most multimodal-related deployment surprises.

#### 5.8.1 Cost #1 — vision encoder weights (always resident, not reclaimable, not transient)

The vision encoder is a 27-layer Qwen3-VL transformer tower bundled inside the model checkpoint as `model_visual.safetensors` (853 MiB on disk, 333 BF16 tensors, ~447M parameters). On startup, vLLM loads these weights into VRAM as part of the overall model load. **They stay resident in VRAM for the entire life of the server, whether or not any request ever contains an image.** You do not pay a runtime cost for text-only prompts because the encoder is never *invoked*, but the weights themselves are permanently committed. This ~0.83 GiB is already subsumed into the 21.88 GiB weights footprint the boot log reports; it is not a separate line item.

**Implication**: you cannot "save VRAM by never sending images." The vision encoder is either loaded or it isn't. There is no request-time toggle and no partial-unload option in vLLM today.

#### 5.8.2 Cost #2 — vision encoder *profiling reservation* (always paid if any image/video modality is admitted at all)

When vLLM boots a multimodal model, its GPU profiler builds a dummy worst-case request and runs a forward pass through the encoder and the LLM to measure peak activation memory. It then reserves enough headroom to handle that worst case at runtime. For the Qwen3.6-35B-A3B-NVFP4 checkpoint, this reservation is **~1.56 GiB**, which is deducted from the pool that would otherwise become available KV cache.

Critically, the profiler's dummy input uses **exactly one image at maximum feature size**, regardless of what integer you pass to `--limit-mm-per-prompt`. The boot log always says:

> `Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum feature size.`

We verified this by booting five configurations back-to-back (image counts 0, 2, 3, 10, 100, 999, and no-flag-at-all) and reading `Available KV cache memory` from each:

| `--limit-mm-per-prompt` | Boot result | `Available KV cache memory` | `GPU KV cache size` |
|---|---|---:|---:|
| `{image: 2, video: 1, audio: 0}` (our config) | ✅ ready | 4.88 GiB | **63,360 tokens** |
| `{image: 2, video: 0, audio: 0}` | ✅ ready | 4.88 GiB | 63,360 tokens |
| `{image: 3, video: 0, audio: 0}` | ✅ ready | 4.88 GiB | 63,360 tokens |
| `{image: 10, video: 0, audio: 0}` | ❌ boot fails (TensorRT-LLM runtime error post-profiling) | 4.88 GiB (logged before crash) | 63,360 tokens (logged) |
| `{image: 100, video: 0, audio: 0}` | ❌ boot fails (same) | 4.88 GiB | 63,360 tokens |
| `{image: 999, video: 0, audio: 0}` | ❌ boot fails (same) | 4.88 GiB | 63,360 tokens |
| **`--limit-mm-per-prompt` omitted** (defaults to 999 per modality) | ❌ boot fails (same) | 4.88 GiB | 63,360 tokens |
| `{image: 0, video: 0, audio: 0}` — full multimodal disable | ✅ ready | **6.44 GiB** | **83,424 tokens (+32%)** |

Three consequences flow from this table, none of them obvious:

1. **Admitting image at all costs a flat ~1.56 GiB / ~20,064 tokens of KV pool.** It is a binary switch, not a per-unit cost. `image: 1` and `image: 3` pay exactly the same profiling tax.
2. **Raising `image` above 3 crashes the server boot entirely** on this nightly + SM 12.0 combination. The KV math still reports correctly — the crash happens in a later init phase (likely CUDA graph capture for the multimodal branch) with a TensorRT-LLM `throwRuntimeError`. We did not bisect the exact ceiling between 4 and 9. **Treat `image: 3` as the maximum safe value.**
3. **Omitting the flag entirely is dangerous.** The default is 999 per modality, which hits the same boot crash. `--limit-mm-per-prompt '{"image": 2, "video": 1, "audio": 0}'` is not conservative hygiene — it is a **load-bearing** parameter without which the server does not start. Do not delete it.

The 1.56 GiB reservation is triggered by having `image > 0` OR `video > 0`. Setting one or both to zero skips the vision-encoder branch of the profiler; setting all three to zero unlocks the full 6.44 GiB pool. The `audio: 0` in the current config already reclaims the audio-profiling reservation; the model is vision-only, so this costs nothing.

**If you never intend to send images or videos**, set `{"image": 0, "video": 0, "audio": 0}` and reclaim +20,064 concurrent-KV tokens. This is currently the largest single lever available short of switching runtimes.

#### 5.8.3 Cost #3 — per-request transient activations (small, bounded, absorbed by the safety margin)

When a request actually contains an image, the vision encoder runs during prefill. The forward pass allocates transient activations (per-layer hidden states, attention scores). These are freed layer-by-layer and fully released before decode begins. Measured on the target hardware:

| Image | Image tokens produced | Transient VRAM spike above baseline |
|---|---:|---:|
| 896×896 PNG | 787 | **+2 MiB** |
| 1792×1792 PNG | ~3,136 | **+8–10 MiB** |
| Two 1024×1024 PNGs in one prompt | ~2,048 (combined) | **+10 MiB** |

These spikes are absorbed trivially by the ~2.25 GiB safety headroom. The previously-quoted "hundreds of MiB for large images, potentially >1 GiB for multi-image" was speculative; the actual measured cost is much smaller than that. The transient-activation cost is not the practical concern for multimodal VRAM budgeting.

#### 5.8.4 Cost #4 — image and video tokens consuming the KV pool at request time

Once the vision encoder produces its output embeddings, they become **tokens in the KV cache**, alongside any text tokens. An image token and a text token are indistinguishable at the KV level — both take the same per-token KV bytes and both draw from the 63,360-concurrent-token pool.

Approximate image-token counts for Qwen3-VL on this checkpoint (patch size 16, spatial_merge 2):

| Image resolution | Image tokens (after patching/merging) |
|---|---:|
| 512×512 | ~256 |
| 896×896 (our `make_image.py` default) | ~787 (measured) |
| 1024×1024 | ~1,024 |
| 1568×1568 | ~2,400 |
| 1792×1792 | ~3,136 (measured) |
| 2048×2048 | ~4,096 |
| Processor cap (`longest_edge=16,777,216` pixels) | up to ~16,384 per image |

Videos are dramatically more expensive. The checkpoint's `processor_config.json` sets `fps=2, min_frames=4, max_frames=768`. A single video input:

- **Short clip (a few seconds, default resolution)**: ~750–1,500 video tokens.
- **Medium clip (~1 minute at default resolution)**: ~10,000–20,000 video tokens.
- **Long clip (several minutes, max frames, moderate per-frame resolution)**: **50,000–75,000 video tokens** — a single video can eat your entire 63,360-token concurrent pool.
- **Pathological (max frames × max per-frame resolution)**: millions of tokens; the processor or max-model-len will reject.

**Practical conclusion**: `video: 1` is not a "cheap" modality. One video can consume the entire KV pool. The real blast-radius cap for video is not `--limit-mm-per-prompt`; it is `--mm-processor-kwargs '{"max_frames": N, "fps": M}'` or equivalent request-time `mm_processor_kwargs`, which bounds the per-video frame count. If you plan to accept videos, pass something like `--mm-processor-kwargs '{"video": {"max_frames": 64, "fps": 1}}'` alongside `video: 1` — otherwise an untrusted client can submit a long clip and starve the pool.

#### 5.8.5 Cost summary table (what you actually pay for common configurations)

| Configuration | Boot? | KV pool | KV tokens | Reclaim vs baseline |
|---|---|---:|---:|---:|
| **Baseline: `{image: 2, video: 1, audio: 0}`** (our current) | ✅ | 4.88 GiB | 63,360 | — |
| `{image: 3, video: 1, audio: 0}` | ✅ | 4.88 GiB | 63,360 | 0 (free increase to 3 images) |
| `{image: ≥10, ...}` | ❌ boot fail | — | — | — |
| `--limit-mm-per-prompt` omitted | ❌ boot fail | — | — | — |
| `{image: 0, video: 0, audio: 0}` | ✅ | 6.44 GiB | 83,424 | **+20,064 tokens (+32%)** — valid only if you never use vision |
| `{image: 2, video: 1, audio: 0}` + MTP enabled | ✅ | 3.30 GiB | 39,072 | −24,288 tokens (−38%) |

#### 5.8.6 Why the pool is so small — and yes, it's largely vLLM's fault

Even our best-case 83,424 concurrent tokens on a 32 GB GPU serving a 35B model with only 10 KV-carrying layers is below what the attention math actually supports. By the layer count alone, 4.88 GiB should hold ~255,000 BF16 KV tokens, and 6.44 GiB should hold ~335,000. vLLM reports roughly 1/4 of that, because its hybrid KV cache allocator reserves per-token block space for **all 40 decoder layers** rather than just the 10 that actually carry attention — a known upstream bug documented as vLLM issue #37121 with candidate fix PR #37429. The 30 Gated DeltaNet layers in this model natively need only a small fixed recurrent state (~0.06 GiB total across all sequences) but are consuming ~75% of the KV pool because the allocator treats hybrid models uniformly.

Put bluntly: your effective concurrent context length is small because vLLM's hybrid KV allocator is, for this architecture, broken in a way that burns 75% of the pool on nothing. SGLang's memory-pool implementation does not have this bug — on the same hardware and the same weights, SGLang would report roughly 3–4× the concurrent-KV capacity, purely by accounting for hybrid layers correctly. We stay on vLLM anyway because the `qwen3_coder` tool parser, the `flashinfer_cutlass` NVFP4 MoE path, and the broader integration surface are worth more than the pool size for our specific agent workload. But this is a real, measurable cost we are paying, not a perceived one. The full treatment, including upstream links and the code-level citation, is in §6.15.

---

## 6. vLLM issues — complete enumeration and per-issue disposition

Every open vLLM issue encountered during stack research is listed below, classified into one of four categories, with an explicit statement of whether it affects our specific setup and what we do about it.

**Classification legend**:
- **A. Runtime bug** — a real bug in vLLM that should be fixed upstream.
- **B. Model OOD failure** — the model occasionally emits output outside its training contract; vLLM is correct-to-contract but brittle.
- **C. Infrastructure bug** — kernel, build, or environment problem, not related to training contract.
- **D. Client-interop bug** — vLLM's HTTP wire format disagrees with an accepted standard.

### 6.1 Issue #39056 — `<tool_call>` inside `<think>` swallowed by reasoning parser [Class B]

**What happens**: when the model occasionally emits a `<tool_call>...</tool_call>` block inside a `<think>...</think>` reasoning block (an out-of-distribution emission), vLLM's reasoning parser swallows the entire `<tool_call>` markup into `reasoning_content` before the tool parser sees it. The response arrives with `tool_calls=[]` and the swallowed markup embedded in `reasoning_content` as plain text.

**Whether it's a parser bug**: no. Qwen3.6's training contract — as evidenced by (a) the chat template's rendering invariant that historical tool_calls always appear **after** `</think>`, (b) the Qwen3-Coder-Next technical report's turn-level tool-format penalty during RL, (c) Alibaba's own evaluation code in `Qwen-Agent/benchmark/deepplanning` stripping everything up to `</think>` before parsing — all indicate the model was trained to emit tool calls only after reasoning closes. Mid-think emission is a model failure mode, not trained behavior. The parser is correct-to-contract.

**Whether it's a Jinja template bug**: no. The template never renders historical tool_calls inside `<think>` — it is structurally incapable of doing so. The template cannot cause this emission pattern.

**Does it affect us**: yes, intermittently. Community reports suggest single-digit-percent frequency on tool-calling responses.

**Resolution — server-side, no client code**: §7.5 (`monkey_patch_tool_call_in_think_rescue.py`) wraps both `Qwen3ReasoningParser.extract_reasoning` and `.extract_reasoning_streaming`. Non-streaming: regex-splice the block from reasoning to content. Streaming: deferred-flush state machine that buffers completed blocks until the `</think>` handoff delta, then prepends the concatenation to upstream's `content` so `parse_delta:614-618` routes a complete block to the tool parser. Pair with a system-prompt guardrail ("close `</think>` before emitting any `<tool_call>`") to reduce residual frequency.

**Do not**: submit an upstream PR to make vLLM's parser tolerate mid-think tool calls. That would desynchronize the parser from the trained distribution, and both the Qwen team and Alibaba's internal evaluator treat mid-think tool calls as malformed.

### 6.2 Issue #39584 — parallel tool calls crash Responses API [Class A]

**What happens**: vLLM's Responses API streaming path has a hardcoded `assert len(delta_message.tool_calls) == 1` in `vllm/entrypoints/openai/responses/serving.py:1377`. Legitimate parallel tool calls (which Qwen3.6's chat template explicitly supports, iterating `message.tool_calls` and emitting multiple `<tool_call>...</tool_call>` blocks) trip the assertion and crash the response.

**Does it affect us**: no. We use `/v1/chat/completions`, not the Responses API. The OpenAI Python SDK's `client.chat.completions.create(...)` call (the one our client glue uses) targets that endpoint, not the Responses path.

**Resolution**: none required. If we ever migrate to the Responses API, wait for upstream PR #39600 or #39586 to merge.

### 6.3 Issue #39598 — MTP drops closing `}` in qwen3_coder tool stream [Class A]

**What happens**: with MTP speculative decoding enabled, MTP occasionally packs `{` + args + `</function>` into a single streaming delta. The `qwen3_coder` streaming tool parser has an early-return that skips the closing-brace flush in that case, producing tool_calls with invalid JSON arguments.

**Does it affect us**: no. MTP is disabled (`#decision-mtp-disabled`).

**Resolution**: none required. If MTP ever becomes viable, wait for the in-flight streaming-parser PR.

### 6.4 Issue #39771 — qwen3_coder crashes on truncated `<parameter=` tag [Class A]

**What happens**: `vllm/tool_parsers/qwen3coder_tool_parser.py:236` uses unsafe `str.index(">")` while extracting a parameter name. When the model is truncated mid-`<parameter=NAME` (before `>`), `.index()` raises `ValueError: substring not found`. The exception is caught at line 320-324 and the parser returns `tools_called=False, content=raw_text` — a silent failure. Line 227, handling function names, already uses the safe `.find()`/`-1` pattern; line 236 is an internal inconsistency.

**Does it affect us**: potentially, whenever a response is truncated by `max_tokens` or by client disconnect mid-generation.

**Resolution**: §7.1 (`monkey_patch_qwen3_coder.py`). At vLLM container startup we replace `Qwen3CoderToolParser._parse_xml_function_call` with a strict version that mirrors PR #39772's `.find()` semantics, but goes further: on a malformed `<parameter=` it returns `None` for the entire tool call (drops, never partially salvages). The patch validates eight import-time landmarks (class lineage, method signature, source containing the buggy sentinel, `__init__` source containing the regex landmark, sibling-method shape, helper-function shape, post-install patch tag, static-lookup tag) and raises `MonkeyPatchRefusedError` on any mismatch. No fork required. `max_tokens=16384` complements the patch by making truncation rare in the first place.

### 6.5 Issue #36872 — MTP produces output gibberish on quantized Qwen3.5/3.6-35B-A3B [Class C]

**What happens**: MTP speculative decoding on FP8 or AWQ quantized Qwen3.5/3.6-35B-A3B produces garbled output. Acceptance rate collapses request-to-request (observed 61.3% → 0.9% → 0%). Root cause is a kernel-level mismatch between the MTP speculator and quantized MoE verification paths.

**Does it affect us**: no. MTP is disabled.

**Resolution**: none required.

### 6.6 Issue #38182 — MTP + prefix caching drops cache hit rate by 21 pp [Class C]

**What happens**: MTP's extra hidden layer changes the effective block payload in vLLM's paged KV cache, causing hash desync and a measured 21-percentage-point drop in prefix-cache hit rate.

**Does it affect us**: no. MTP is disabled.

**Resolution**: none required.

### 6.7 Issue #36865 — SM 12.0 source builds register unsupported FlashMLA/FA kernels [Class C]

**What happens**: when building vLLM from source on SM 12.0, some FlashMLA / FlashAttention kernels are registered despite being unsupported on consumer Blackwell, causing runtime errors.

**Does it affect us**: no. We use the pre-built Docker image, not a source build. The image's kernel set is pre-validated for SM 12.0.

**Resolution**: none required. If you ever build from source for this hardware, use `--attention-backend FLASHINFER` explicitly.

### 6.8 Issue #35303 — `CompressedTensorsWNA16MarlinMoEMethod` crash on `actorder=null` AWQ MoE [Class C]

**What happens**: the `compressed-tensors` quantization path unconditionally registers `g_idx` for AWQ MoE, crashing on checkpoints that have `actorder=null`.

**Does it affect us**: no. Our chosen model (`RedHatAI/Qwen3.6-35B-A3B-NVFP4`) uses `compressed-tensors` but with NVFP4 format, loaded via `CompressedTensorsW4A4NVFP4Method` — a different code path from the buggy `CompressedTensorsWNA16MarlinMoEMethod`. The bug is specific to the WNA16 (weight-N-bit-activation-16) Marlin MoE method and its `g_idx`/`actorder` handling, which NVFP4 does not use.

**Resolution**: none required (by virtue of quant-method choice).

### 6.9 Native NVFP4 MoE on SM 12.0 — historically missing, now fixed upstream

**What historically happened**: NVFP4 (native Blackwell 4-bit floating point) MoE kernels were not wired for SM 12.0 (consumer Blackwell) when the RTX 5090 launched. vLLM would crash with *"FlashInfer-CUTLASS MoE kernel does not support current device sm_120"*. The tracking feature request was issue **#31085** (reported capability gap); the concrete bug report that drove the fix was issue **#33416**.

**Status**: **fixed by PR #33417** (merged 2026-01-31). The PR adds `is_device_capability_family(120)` checks across `flashinfer_cutlass_moe.py`, `flashinfer_cutedsl_moe.py`, `flashinfer_trtllm_moe.py`, `flashinfer_fp4_moe.py`. Follow-up fixes (PR #37725 for NaN on desktop Blackwell, PR #38423 for RTX 5090 specific) landed through Q1 2026.

**Does it affect us**: our entire NVFP4 stack depends on these fixes being present in the runtime. They are in the pinned nightly. Verified at boot on 2026-04-21: vLLM dispatches NVFP4 MoE through `FlashInferCutlassNvFp4LinearKernel` and the `FLASHINFER_CUTLASS` MoE backend, NOT falling back to Marlin.

**Resolution**: none required.

### 6.10 Issue #37714 — SM 12.0 + CUDA 13 pip install failures [Class C]

**What happens**: building vLLM from source via pip on SM 12.0 + CUDA 13 has multiple known failure modes.

**Does it affect us**: no. We use the pre-built Docker image.

**Resolution**: none required.

### 6.11 Issue #37242 — WSL2 RTX 5090 configuration notes [Class C — community documentation, not a bug report]

**What it is**: Community-filed issue (title *"[Community] RTX 5090 (Blackwell sm_120) + WSL2 2.7.0: CUDA graphs work — benchmarks + full config"*). It documents a **working** WSL2 configuration, with the caveat that dxgkrnl on WSL2 does not expose native FP8 tensor cores — so FP8 paths on WSL2 fall back to slower kernels. Not a bug per se; a user note on performance delta.

**Does it affect us**: no — we are on Linux native, and we use AWQ/NVFP4, not FP8 weights.

**Resolution**: none required.

### 6.12 `reasoning` vs `reasoning_content` response field name [Class D — interop bug, not a numbered issue]

**What happens**: vLLM uses the non-standard field name `reasoning` on **both** sides of the wire:

- **Egress**: responses populate `choices[i].message.reasoning` (non-streaming) and `choices[i].delta.reasoning` (streaming). The OpenAI-standard field name is `reasoning_content`.
- **Ingest**: on replay of prior assistant turns, vLLM reads `message.reasoning` only (`vllm/entrypoints/chat_utils.py:1519` in the pinned commit) and ignores `message.reasoning_content`.

Any client glue that follows the OpenAI standard (reading `reasoning_content` on responses, writing `reasoning_content` on outgoing messages) silently loses reasoning in **both directions** every turn. That defeats `preserve_thinking` (§5.7) and degrades tool-argument accuracy after two or three turns.

This is not a numbered vLLM issue and is unlikely to become one — vLLM and the OpenAI ecosystem have been disagreeing on the field name for several releases. Fix the wire-level mismatch at the client boundary.

**Resolution — both sides closed server-side, no client code**: egress handled by `monkey_patch_reasoning_field_egress.py` (§7.3); ingest handled by `monkey_patch_reasoning_field_ingest.py` (§7.4). Qwen-Agent's `hasattr(delta, 'reasoning_content')` check (`oai.py:112`) now evaluates True, Qwen Code CLI's `reasoning_content ?? reasoning` fallback (`converter.ts:941-942`) finds the value in its primary slot, and plain OpenAI SDK clients work unmodified.

### 6.13 vLLM silent tool-parser failure modes [Class A — observability gap]

**What happens**: the `qwen3_coder` tool parser has 8 non-streaming and 6 streaming code paths where it returns `tools_called=False` with the raw `<tool_call>…` markup as `content`, or returns `tools_called=True` with invalid / truncated JSON arguments. All paths produce HTTP 200 with no error surfaced. vLLM emits no server-side metrics for parser failure. Community evidence puts baseline frequency at 1–5% of tool-calling responses, rising to 10–20% under long context, reasoning, or speculative decoding.

**Does it affect us**: yes. Silent failures in agent loops manifest as mysterious behavioral drift — the agent reads raw `<tool_call>` markup in `content` as plain text, interprets it as "the model chose not to call a tool", and proceeds off-track.

**Resolution — server-side observability only; no client-side validator**: `monkey_patch_extract_tool_calls_metrics.py` (§7.6, non-streaming) and `monkey_patch_extract_tool_calls_streaming_metrics.py` (§7.7, per-delta streaming) turn the silent-failure class into an operator-visible Prometheus counter `vllm_qwen3_coder_silent_tool_call_failures_total{failure_kind, model}` plus a structured WARNING log. Both are observation-only — upstream return values forwarded unchanged.

Other failure classes (truncation mid-tool-call, invalid JSON in `arguments`, schema-required-field misses, duplicate keys) are model-behavior issues that do not belong to the server-compatibility contract. Any client that wants per-call diagnostic depth can add its own validation layer; this repo does not ship one.

### 6.14 `transformers==5.5.4` broken `GenerationConfig` top-level import in 2026-04-21's newest nightly [Class C]

**What happens**: the most recent published nightly Docker image on 2026-04-21, `vllm/vllm-openai:nightly-b47840019e61a3983c8144066a99c843d177947d` (digest `sha256:d39d4b0f…`), ships `transformers==5.5.4`. The statement `from transformers import GenerationConfig, PretrainedConfig` at `vllm/transformers_utils/config.py:18` raises `ImportError` at CLI boot, because `transformers._LazyModule` has not initialized the top-level `GenerationConfig` export by the time vLLM's CLI loader imports it. A direct `python -c "import transformers; transformers.GenerationConfig"` works; the failure is specific to the import-order path that `vllm serve` follows. Downgrading `transformers` in-container to `4.57.6` lets the APIServer start, but the spawned `EngineCore` subprocess re-hits the same ImportError because of stale `sys.modules` state, which blocks the server from actually serving requests.

**Does it affect us**: it *would* if we used the newest-available nightly. We avoid it by pinning **yesterday's** nightly — `nightly-8936118134d0547fa1cc78adab2d03edd6d3dc48`, digest `sha256:9bba4628…` — which carries the same functional surface (Qwen3.5-MoE + NVFP4 + FlashInfer CUTLASS) but does not exhibit the import regression. See §3.2.

**Resolution**: pin the older nightly digest. Re-evaluate (and upgrade) once a new nightly appears that tests clean with `docker run --rm vllm/vllm-openai@<digest> vllm --help`. The expected fix upstream is either a transformers patch release or a vLLM patch that imports `GenerationConfig` defensively.

### 6.15 Issue #37121 — hybrid KV cache allocator over-reserves for non-attention layers (Qwen3-Next / Qwen3.5 / Qwen3.6 family) [Class A — runtime inefficiency, the single largest cost in our deployment]

**What happens**: vLLM's V1 paged KV cache manager uses a **unified page-size allocator**. For hybrid-attention models — where some layers carry real per-token KV state (full softmax attention) and others carry only a small fixed recurrent state (Mamba, GDN, Gated DeltaNet) — the allocator pads the linear-attention layers' small native page size up to the attention layers' padded page size, then reserves `num_blocks × padded_page_size` per layer for **every layer**, not just the attention ones. For Qwen3.6-35B-A3B (40 decoder layers, 10 full-attention + 30 Gated DeltaNet) this causes the KV pool to be over-reserved by a factor of roughly **40 / 10 = 4×**. The concrete code lives at:

- `vllm/platforms/interface.py:615-635` — forces `cache_config.mamba_page_size_padded = attn_page_size` and logs *"Padding mamba page size by X% to ensure that mamba page size and attention page size are exactly equal."*
- `vllm/v1/core/kv_cache_utils.py:1148-1168` (`get_kv_cache_config_from_groups`) — allocates padded-page-size × num_blocks per layer in the group; for Qwen3.6 all 40 layers end up in one group.
- `vllm/v1/core/kv_cache_utils.py:808-820` (`get_max_concurrency_for_kv_cache_config`) — uses the unified page size for the `Maximum concurrency for X tokens per request: Y×` boot log line. This is why our boot reports `1.85×` instead of the ~7.4× the attention math alone would give.
- `vllm/v1/kv_cache_interface.py:383-409` (`MambaSpec.max_memory_usage_bytes`) — correctly reports that Mamba/DeltaNet natively needs O(requests) slots, not O(tokens); but the unified-pool sizing discards this information.

**Measured impact on this deployment**:
- Reported: `Available KV cache memory: 4.88 GiB`, `GPU KV cache size: 63,360 tokens`, `Maximum concurrency: 1.85×`.
- Attention-math expected: 4.88 GiB / (10 layers × 2 KV heads × 256 head_dim × 2 bytes × 2(K+V)) = **~255,850 tokens**, concurrency ~7.4×.
- The ratio 255,850 / 63,360 ≈ 4.04 matches 40 / 10 exactly, confirming the all-40-layers-reserved diagnosis.

**Does it affect us**: yes, profoundly. Our entire concurrent-KV budget is ~4× smaller than the hardware and the model architecture would otherwise support. This is the single largest performance cost in the deployment.

**Upstream status as of 2026-04-22**:
- **Tracking issue: [#37121](https://github.com/vllm-project/vllm/issues/37121)** (filed 2026-03-15 by `swtb3`, "KV cache ~7x memory overestimation for hybrid Mamba/attention models (Qwen3.5)"). Still open; no assignee; no merged fix.
- **Broad-scope candidate fix PR: [#37429](https://github.com/vllm-project/vllm/pull/37429)** (opened 2026-03-18, "Fix KV cache sizing and allocation for hybrid Mamba/attention models"). Gives Mamba/DeltaNet its own dedicated pool. Review is paused by maintainer `@NickLucche` pending an RFC-level design discussion because it contradicts the unified-pool architecture. As of 2026-04-22 the PR has been **silent since 2026-03-19** (pre-commit CI failing, no author response to NickLucche's "overriding a lot of work that went into HMA" comment); still the only broad-scope candidate in the tree.
- **Narrow-scope candidate fix PR: [#40384](https://github.com/vllm-project/vllm/pull/40384)** (opened 2026-04-20, "[Bugfix] Exclude O(1) Mamba groups from hybrid KV cache token capacity") by `jhsmith409` (same author as the hybrid TurboQuant PR #39931). **Explicitly validated on our exact model** (`RedHatAI/Qwen3.6-35B-A3B-NVFP4`, "40 layers = 30 DeltaNet linear-attention + 10 full-attention"). Small diff: +115/-13 across `kv_cache_utils.py`, `scheduler.py`, and a new test. Key quote from the PR: *"For a typical hybrid with one attention group and N Mamba groups, that's off by a factor of `(1 + N) / 1` — 2x understatement for the common case, 4x for Nemotron-H-style 1 attn + 3 mamba groups."* Mergeable=true, no maintainer approval yet (8 reviewers requested); as of 2026-04-22 external verification by `Sandermage` on A5000 (946K-token capacity, ~5.2× concurrency) has landed in the thread, and a related follow-up issue [#40417](https://github.com/vllm-project/vllm/issues/40417) was filed noting the same divisor bug also affects `SlidingWindowSpec` / `ChunkedLocalAttentionSpec` on Gemma 3 / Phi / Mistral hybrids (not Qwen3.6 — ours is full-attention + GDN only, so #40417 does not affect us).
- **Scope caveat**: #40384 fixes the `max_num_kv_tokens` / per-token capacity counting and scheduler budget only. It does NOT reshape the underlying block allocation or page-size unification. So post-#40384 the reported token capacity will jump (the scheduler will see and act on more usable capacity), but the deeper memory-allocation inefficiency — padding all 40 layers' blocks up to the attention page size — remains until #37429 or its successor lands. Therefore: **#40384 is a real and immediate improvement (probably a 2–4× jump in reported concurrent tokens on our deployment), but is not the full fix.**
- **Why we do NOT attempt to backport #37429 locally**: scoping research traces the unified-pool assumption through nine load-bearing sites in the pinned tree — `platforms/interface.py:615-635` (root cause), `v1/kv_cache_interface.py:383-409` (`MambaSpec.page_size_bytes`), `v1/core/kv_cache_utils.py:857-863, 914-951, 959-1078, 1081-1157` (uniform-page-size asserts and the byte-reservation site), `v1/core/kv_cache_coordinator.py:368-451` (`HybridKVCacheCoordinator` single-`BlockPool` assumption), `v1/core/kv_cache_manager.py:143`, and `v1/worker/gpu_model_runner.py:6476-6506` (tensor materialisation from the config). A correct port requires a new `MambaPool` class + a `HybridAllocator` wrapping two `BlockPool`s + split tensor construction + second-block-table plumbing in the coordinator + loosened uniform-page-size asserts + `KVCacheConfig` carrying two `num_blocks`/tensor lists. Estimated scope: **+650 to +1,100 LOC, –150 LOC, across 7–9 files** (medium confidence); SGLang's equivalent pool code is ~670 LOC (`MambaPool` + `HybridReqToTokenPool` + `HybridLinearKVPool`) as an upper-bound reference. This is architectural, not monkey-patch scope: it mutates dataclass fields and physical tensor layout, which no landmark-validated import-time patch can safely cover.
- **Prior RFC acknowledgment**: closed RFC [#11382](https://github.com/vllm-project/vllm/issues/11382) documented up to **79.6% memory waste** for hybrid architectures. So this has been known since late 2024.
- **SGLang does not have this bug at all.** Their memory-pool implementation (`sglang/srt/mem_cache/memory_pool.py:480-508` and `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:85-131`) allocates a separate `MambaPool` sized via `--mamba-full-memory-ratio`, so the attention KV pool is sized only against what attention layers actually need. The same model on the same hardware would give roughly 3–4× the concurrent-KV capacity on SGLang today.

**Resolution**: backported in this repo as `monkey_patch_hybrid_kv_allocator.py` (§7.2). The patch mirrors PR #40384's intent — exclude O(1) Mamba/DeltaNet groups from per-token KV-capacity counting in both `get_max_concurrency_for_kv_cache_config` and `_report_kv_cache_config`. The `GPU KV cache size: 63,360 tokens / Maximum concurrency: 1.85x` boot line should become roughly `~200K+ tokens / ~7x` with no other changes. SGLang remains a documented alternative if vLLM ever stops being viable, but with the patch applied the concurrent-KV gap closes and the trade-off no longer favors switching.

**Caveats this patch does NOT eliminate**:
1. Scheduler-budget fix only. The pool's actual byte footprint is still over-reserved (75% of bytes go to DeltaNet layers) — that's PR #37429's territory and remains un-addressed in our deployment. The bytes are sitting there unused; the patch credits the scheduler with capacity that already exists physically. No new memory is conjured.
2. Patch must be removed when either PR #40384 or PR #37429 lands. The patch's source landmarks refuse to apply against a fixed function automatically; PR #37429 specifically would change the underlying pool byte size and the patch's scheduler view would no longer be coherent with what the pool can actually hold.
3. **Live smoke test required before declaring validated** — the patch's blast radius is the request-admission path. Boot the patched server, confirm the boot log reports the higher `GPU KV cache size` value, then run a workload that admits >63K concurrent tokens and confirm no OOM. The static landmark validation cannot prove this dynamically.

**Action taken**: posted a comment on issue #37121 with our measured evidence on Qwen3.6-35B-A3B-NVFP4 + RTX 5090 + SM 12.0 as a second reproduction point (the issue was filed on a Qwen3.5 H100 setup). Watching both #37121 and PR #37429 for merge.

**Revisit when**: either PR #40384 or PR #37429 lands in a nightly — at that point delete the patch file. PR #37429's additional fix for the underlying byte-level padding is a second win on top of #40384; if it lands while our backport is still in place, remove the backport before adopting the new image (composing them risks pool-size / scheduler-view mismatch).

---

## 7. The patches in this repo

Seven monkey-patches plus one container-entrypoint launcher. Every patch is **server-side** — loaded into the vLLM Python process by `launch_with_patches.py` before `vllm serve` hands over to `uvicorn`. There is **no client-side code in this repo**. Off-the-shelf OpenAI-compatible clients (Qwen Code CLI, Qwen-Agent, the OpenAI Python SDK) connect to the patched server unmodified.

Each patch is a self-contained Python script. Each addresses a specific defect named in §6 and only that defect — no scope creep, no incidental refactors. Each strictly validates its target's structure via landmarks before touching anything, and each refuses to apply on any landmark mismatch with a typed exception that names the exact landmark that failed.

The patch file itself is the source of truth. This section is a contract index, not a re-statement. Read the file when you need the implementation; the file's module docstring states the contract authoritatively.

| # | File | Defect addressed |
|---|---|---|
| 1 | [`monkey_patch_qwen3_coder.py`](monkey_patch_qwen3_coder.py) | §6.4 / #39771 — `_parse_xml_function_call` crash on truncated `<parameter=` tag |
| 2 | [`monkey_patch_hybrid_kv_allocator.py`](monkey_patch_hybrid_kv_allocator.py) | §6.15 / #37121 (PR #40384 backport) — exclude O(1) Mamba groups from per-token KV-capacity counting; raises reported concurrent-KV from ~63K → ~200K+ tokens at our config with zero quality tradeoff |
| 3 | [`monkey_patch_reasoning_field_egress.py`](monkey_patch_reasoning_field_egress.py) | §6.12 egress — rename `reasoning` → `reasoning_content` on serialization of `ChatMessage` and `DeltaMessage`. Every response-dump path (including streaming `exclude_unset=True`) emits the OpenAI-spec key. |
| 4 | [`monkey_patch_reasoning_field_ingest.py`](monkey_patch_reasoning_field_ingest.py) | §6.12 ingest — accept `reasoning` or `reasoning_content` on replayed assistant messages. Refuses ambiguous client input (both fields present with differing values) with HTTP 400. |
| 5 | [`monkey_patch_tool_call_in_think_rescue.py`](monkey_patch_tool_call_in_think_rescue.py) | §6.1 / #39056 — rescue `<tool_call>...</tool_call>` blocks emitted inside `<think>`. Non-streaming: reasoning-to-content splice. Streaming: deferred-flush state machine that buffers rescued blocks across deltas and prepends them to the `</think>` handoff delta so the downstream tool parser sees a complete block. |
| 6 | [`monkey_patch_extract_tool_calls_metrics.py`](monkey_patch_extract_tool_calls_metrics.py) | §6.13 non-streaming observability — server-side Prometheus counter + structured WARNING on `markup_leak`. Observation-only wrapper; upstream return unchanged. |
| 7 | [`monkey_patch_extract_tool_calls_streaming_metrics.py`](monkey_patch_extract_tool_calls_streaming_metrics.py) | §6.13 streaming observability — per-delta `markup_leak_streaming` detection. Shares the counter with patch 6 via `failure_kind` label; split by operator as needed. |
| L | [`launch_with_patches.py`](launch_with_patches.py) | Container-entrypoint replacement that imports patches 1–7 in order, runs per-patch post-install verification, and hands off to `vllm.entrypoints.cli.main` via `runpy.run_module(alter_sys=True)`. Required because CPython's `PYTHONSTARTUP` does not fire under the image's non-interactive ENTRYPOINT; the launcher replaces `["vllm", "serve"]` with `["python", "/opt/patches/launch.py", "serve", ...]`. |

### 7.1 Patch 1 — `monkey_patch_qwen3_coder.py`

**What it patches**: replaces `Qwen3CoderToolParser._parse_xml_function_call` to use `str.find(">")` instead of `str.index(">")` and, on a malformed `<parameter=` tag, return `None` for the whole tool call rather than raising `ValueError` (which the upstream `try/except Exception` at `extract_tool_calls:320-324` collapses into "drop all N tool calls in this response"). Sibling well-formed tool calls in the same response are preserved.

**Drop-vs-salvage policy**: a ToolCall with silently-omitted fields is the worst possible outcome in an agent loop. Returning `None` lets the upstream filter at `extract_tool_calls:313` keep only valid siblings; the `<parameter=` markup of the dropped call leaks to `content`, where §7.6 / §7.7 surface it as `markup_leak` / `markup_leak_streaming`.

**Removal trigger**: upstream PR #39772 merges. The patch's source-landmark check refuses against a fixed function.

### 7.2 Patch 2 — `monkey_patch_hybrid_kv_allocator.py`

**What it patches**: replaces `get_max_concurrency_for_kv_cache_config` and `_report_kv_cache_config` in `vllm.v1.core.kv_cache_utils`. Both were including O(1)-per-request `MambaSpec` groups in per-token capacity counting as if they scaled with tokens — for Qwen3.6 that halves the reported `GPU KV cache size` and scheduler budget. Patched to filter `kv_cache_groups` to token-capacity-contributing specs only (`AttentionSpec` always; `MambaSpec` only when `cache_config.mamba_cache_mode == "all"`).

**Backport semantics, not literal port**: PR #40384's intent translated to the pinned commit's function shapes. The PR's literal diff targets master and uses signatures (`MambaSpec.max_memory_usage_bytes(num_running_requests, num_total_tokens)`) that do not exist at the pinned commit (real signature is `(self, vllm_config)`; the O(1) vs O(tokens) branch is selected internally via `cache_config.mamba_cache_mode`). The patch follows the source, not the upstream PR text.

**Scope**: scheduler-budget only (PR #40384's scope). PR #37429's broader byte-level over-allocation fix is explicitly out of scope — the bytes are still over-reserved; the scheduler just was not crediting them.

**Risk acknowledgment**: this is on the request-admission accounting path. An over-count here admits more concurrent tokens than the pool can hold → OOM at runtime. The landmark-validation discipline mitigates but does not eliminate this risk. **A live smoke test confirming the patched server admits requests within the post-patch reported pool size without OOM is mandatory before declaring validated.**

**Removal trigger**: upstream PR #40384 merges (source-landmark check will refuse). **Critical sequencing**: remove BEFORE pulling an image that contains PR #37429 — the broader fix changes the underlying pool byte size, and patch 2's scheduler view would no longer match.

### 7.3 Patch 3 — `monkey_patch_reasoning_field_egress.py`

**What it patches**: installs Pydantic v2 `serialization_alias = "reasoning_content"` on the `reasoning` field of `ChatMessage` and `DeltaMessage`, flips `model_config["serialize_by_alias"] = True`, deletes the cached `__pydantic_core_schema__` / `__pydantic_validator__` / `__pydantic_serializer__`, and calls `model_rebuild(force=True)`. Every response-serialization path vLLM uses (`model_dump`, `model_dump_json`, both with `exclude_unset=True` variants, and nested use) emits `reasoning_content` on the wire. The internal Python attribute stays `.reasoning` — `serving.py:1036-1037` reads it as an attribute and must keep working.

**Why the composite rather than a simpler approach**: verified empirically — none of (a) `serialization_alias` alone, (b) `serialization_alias` + `model_rebuild(force=True)`, (c) wrapping `model_dump` / `model_dump_json` in Python takes effect on every path vLLM uses. See the patch's module docstring for the four scratch-experiment trace of why. The composite is the only approach that works.

**Why class-wide `serialize_by_alias=True` is safe**: audited at the pinned commit — no other field on either target class carries a `serialization_alias`, and `api_router.py:70`, `serving.py:678/714/1193/1218`, `run_batch.py:353/424` all omit `by_alias` on their dump calls (so the class-wide flag wins). Phase 3 of the patch re-audits at install time and refuses if either precondition drifts.

**Scope caveats**:

* `run_batch.py` output JSONL will also emit `reasoning_content`. Desirable (consistent field naming across streaming, non-streaming, and batch); flagged for downstream batch-output consumers.
* The Responses API / harmony path is unaffected — it uses separate `ResponseReasoningItem` / `type="reasoning_text"` schemas, not `ChatMessage` / `DeltaMessage`.

**Removal trigger**: vLLM ships native `reasoning_content` on Chat Completions (protocol.py:63 already flags the field as "vLLM-specific"). Landmark #5 (no existing `reasoning_content` field) will refuse.

### 7.4 Patch 4 — `monkey_patch_reasoning_field_ingest.py`

**What it patches**: wraps `vllm.entrypoints.chat_utils._parse_chat_message_content`. When `role == "assistant"`, `reasoning is None`, and `reasoning_content` is a non-None string, synthesises a shallow copy with `reasoning` populated from `reasoning_content` and delegates to the original. Other paths delegate unchanged.

**Upstream bug**: `chat_utils.py:1519` reads `reasoning = message.get("reasoning")` and drops `reasoning_content` entirely. The chat template reads `message.reasoning_content` — works only when the client happens to have sent `reasoning`. Qwen-Agent (`oai.py:149, 169-173`) and Qwen Code CLI (`converter.ts:335, 514`) both write `reasoning_content`, so both were silently losing prior-turn reasoning on every round-trip. Without prior-turn reasoning the Qwen3.6 interleaved-thinking contract (§5.7) is violated and tool-argument accuracy degrades after two or three turns.

**Ambiguity handling — strict**: both fields present with **different** values raises `ReasoningFieldAmbiguityError` (`ValueError` subclass → HTTP 400). Identical values pass through. Non-string `reasoning_content` (dict, list) also refuses rather than being stringified. We do not guess; `str(dict)` would corrupt the rendered prompt in a way that compounds.

**Removal trigger**: vLLM widens ingest to accept `reasoning_content`. Landmark #4 (`reasoning = message.get("reasoning")` in source) will refuse.

### 7.5 Patch 5 — `monkey_patch_tool_call_in_think_rescue.py`

**What it patches**: wraps `Qwen3ReasoningParser.extract_reasoning` (non-streaming) and `.extract_reasoning_streaming` (streaming). Moves any `<tool_call>...</tool_call>` markup the model emits INSIDE `<think>...</think>` out of reasoning and into content, so the downstream `Qwen3CoderToolParser` can see it and extract the structured tool call. Without this patch, mid-`<think>` tool calls land in `reasoning_content` as plain text and `tool_calls` is silently empty — the §6.1 failure mode.

**Non-streaming rescue**: after `extract_reasoning` returns, a literal-marker non-greedy regex lifts every complete block from `reasoning` into `content`. Unclosed openers stay in reasoning (truncated markup is not rescuable; §7.1's territory).

**Streaming rescue — the architecturally load-bearing part**:

`parse_delta` (`abstract_parser.py:585-647`) gates the tool-parser handoff on `self.is_reasoning_end(delta_token_ids)` — which requires `</think>` in the current delta's tokens. Mid-`<think>`, that predicate is False, so a `DeltaMessage(content="<tool_call>…</tool_call>")` returned from the reasoning parser goes directly to the wire without ever reaching `extract_tool_calls_streaming`. The naïve "emit rescued content immediately" approach therefore **does not fix the silent tool-call loss** — it just moves the markup from `reasoning` to `content` on the wire. The client still sees `tool_calls=[]`.

This patch solves it with **deferred flush**: completed rescue blocks accumulate in a per-instance pending list; mid-`<think>`, nothing is emitted as content. On the handoff delta (`self.end_token_id in delta_token_ids`) the concatenation is prepended to upstream's `content`. `parse_delta:618` sets `current_text = delta_message.content = <pending_blocks> + <real_post_think_content>`, `parse_delta:625-630` hands that to `extract_tool_calls_streaming` as its first `delta_text`, and the tool parser parses each complete block in a single pass (burst-delivery is explicitly supported per `qwen3coder_tool_parser.py:499-503, 526-532`). Mid-`</think>` with an unclosed block forwards the in-flight buffer on the handoff so the tool parser's streaming machinery takes ownership for the remainder.

**Non-streaming flow and per-delta state-machine cases, straddle reconstruction, policy decisions, all ten import-time landmarks, and WHY-comments for edge cases are documented authoritatively in the patch's own module docstring and at the code sites.**

**Observability for abandoned rescues**: at stream start (`not previous_text`), `_stream_start_reset_with_warning` inspects carry-over state. Pending blocks or `_qwen36_rescue_in_block=True` emits a single WARNING naming the dropped count before clearing. Nonzero rate → investigate truncation, not the patch.

**Known limitation**: model truncates mid-`<think>` without `</think>` and without closing the in-flight `<tool_call>` → rescue blocks dropped (WARNING-logged). A system-prompt guardrail ("You MUST close `</think>` before emitting any `<tool_call>`") reduces the residual rate.

**Removal trigger**: vLLM adopts a reasoning parser that re-routes tool-call markup itself. No upstream PR is tracking this today.

### 7.6 Patch 6 — `monkey_patch_extract_tool_calls_metrics.py`

**What it patches**: wraps `Qwen3CoderToolParser.extract_tool_calls` (non-streaming). Fires server-side observability iff the result has the silent-failure shape (`tools_called is False` AND `model_output` contains `<tool_call>` / `<function=` / `<parameter=`); returns the upstream result unchanged.

**Emits**: Prometheus counter `vllm_qwen3_coder_silent_tool_call_failures_total{failure_kind="markup_leak", model=...}` plus a single structured WARNING log (`silent_tool_call_failure kind=markup_leak content_len=%d marker_count=%d model=%s`). Greppable, ELK-friendly.

**Scope discipline**: only `markup_leak`. `truncated_tool_call` needs `finish_reason` (added downstream of this method); `invalid_tool_arguments_json` needs JSON parsing of `arguments` (also downstream). Both out of scope here.

**The one place this stack catches `Exception`**: the instrumentation block inside the wrapper (documented at the site). The call to the original method is OUTSIDE the guard — real parser exceptions propagate. Observability must not take down request handling.

**Cardinality caveat**: `model` label has unbounded cardinality. Fine for a single-model deployment like this one; multi-tenant deployments should drop or hash the label.

**Removal trigger**: vLLM ships first-class metrics for tool-parser silent failures. No landmark refuses — operator must notice.

### 7.7 Patch 7 — `monkey_patch_extract_tool_calls_streaming_metrics.py`

**What it patches**: wraps `Qwen3CoderToolParser.extract_tool_calls_streaming`. Fires per-delta when the `DeltaMessage` it returns has non-empty `content` containing at least one COMPLETE marker AND empty `tool_calls`. Returns the upstream result unchanged.

**Partial-marker tolerance — deliberate**: `content="<too"` does NOT fire. Cross-delta accumulation is not implemented. Missed split-marker events are acceptable observability noise, NOT a correctness bug.

**Shares the counter with §7.6 via collision lookup**: registers under the same name, and on `ValueError` ("already registered") looks up the existing collector via `prometheus_client.REGISTRY._names_to_collectors` (probed with and without the `_total` suffix to tolerate version drift) and reuses it. `failure_kind="markup_leak_streaming"` distinguishes the series; operators query one metric family. If collision-lookup fails, degrades to log-only observability without fabricating a duplicate collector. Three `_prometheus_status` states at install: `active` / `active_shared` / `unavailable`.

**Observation-only invariant**: identical shape to §7.6 — upstream call outside the `try/except`; only instrumentation inside.

**Removal trigger**: same as §7.6.

### 7.L Launcher — `launch_with_patches.py`

**What it is**: the container entrypoint, replacing `["vllm", "serve"]` with `["python", "/opt/patches/launch.py", "serve", ...]`. Imports every registered patch in `_PATCH_MODULES` order, runs the per-patch `_PATCH_VERIFICATION` verifier for each, then hands off to vLLM's CLI via `runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)`.

**Why it exists**: `PYTHONSTARTUP` does not fire under non-interactive entrypoints (CPython honours it only in interactive mode), so the original README revision's `PYTHONSTARTUP=/opt/patches/monkey_patch.py` never loaded the patch — the server was booting unpatched regardless of the bind-mount. The launcher closes that gap.

**Per-patch verification**: after each patch imports, the launcher re-imports the relevant vLLM target FROM SCRATCH (not from the patch module's cached references) and asserts the install took effect. Method-wrapping patches (#1, #3, #4, #6, #7 of `_PATCH_MODULES`) verify the tag via both `getattr` AND `inspect.getattr_static`. The Pydantic-schema patch (`monkey_patch_reasoning_field_egress`) adds a behavioural check — constructs an instance and asserts `model_dump()['reasoning_content']` equals a probe value while `'reasoning'` is absent — because a tag-only check would miss a "tag stamped but serializer rebuild silently failed" regression. Any failure raises `PatchVerificationError` and the container exits non-zero.

**Load order** matters only where one patch's side-effects must precede another: `reasoning_field_egress` must come before `tool_call_in_think_rescue` because the egress patch rebuilds the `DeltaMessage` Pydantic schema and the rescue patch constructs `DeltaMessage` instances at request time. Current order is documented in `_PATCH_MODULES` with per-line rationale.

**Known limitation — plugin-snapshot gap**: Python's late binding ensures existing instances see patched methods, but if vLLM's tool-parser registry takes a snapshot of a class before the patch ran (plugin manifest, entry-point load, `copy.deepcopy` of method dicts during engine init), the verifier would pass and the served parser would still be unpatched. **Mitigation: live smoke test** — POST requests exercising each patched path and confirm the patched behaviour before declaring validated. Static verification does not prove the served code path uses the patched version.

---

## 8. Deployment commands

### 8.1 Fetch and pin the Docker image digest

```bash
# Pull the exact image at the pinned digest
docker pull vllm/vllm-openai@sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c

# Verify
docker inspect vllm/vllm-openai@sha256:baaf5fc76b2f203f17bd1934d9c26740b00e67a2f9b030922cf3aac880c7ba8c \
  --format '{{.Id}} {{.Architecture}}'
```

### 8.2 Launch vLLM — measured, booted, and smoke-tested on target hardware 2026-04-21

Measured boot outcome on the target RTX 5090 at this exact command:
- Model load completed in ~17 s (3 safetensors shards)
- "Available KV cache memory: 4.88 GiB" → "GPU KV cache size: 63,360 tokens"
- CUDA graphs captured (PIECEWISE 4/4, FULL 3/3)
- `INFO: Application startup complete` at T+~100 s from container start
- Idle VRAM 28,494 / 32,607 MiB (27.83 GiB used)
- First request /v1/chat/completions with a tool schema: HTTP 200, `finish_reason=tool_calls`, 608 chars of `reasoning`, valid JSON `arguments` on the tool call
- Under a 119,907-token single-request prompt: HTTP 200, peak VRAM 30,304 MiB, 2.25 GiB absolute safety margin remained
- Steady-state decode throughput post-warmup: **~153 tok/s** on a 500-word essay prompt

Exact command below — each flag linked to a numbered rationale row.

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
  --model RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --revision e850c696e6d75f965367e816c16bc7dacd955ffa \
  --served-model-name Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --moe-backend flashinfer_cutlass \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  --limit-mm-per-prompt '{"image": 2, "video": 1, "audio": 0}'
```

**Why every flag exists — each verified by live boot on the target hardware**:

| Flag | Rationale |
|---|---|
| `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864` | Required by PyTorch multiprocessing and NCCL even on single-GPU. Without `memlock=-1` and stack headroom, CUDA graph capture can fail. |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | Persists the 24 GB model download across container restarts. |
| Seven `-v $PWD/monkey_patch_*.py:/opt/patches/...:ro` mounts + `-v $PWD/launch_with_patches.py:/opt/patches/launch.py:ro` + `-e PYTHONPATH=/opt/patches` + `--entrypoint python /opt/patches/launch.py serve` | Mounts patches 1–7 (§7.1–§7.7) and the launcher (§7.L) read-only into the container, makes the patches importable from the launcher's `PYTHONPATH`, replaces the image's default `["vllm", "serve"]` ENTRYPOINT with the launcher (which imports each patch in `_PATCH_MODULES` order, post-verifies each one's tag, then `runpy`s vLLM's CLI). Earlier README revisions used `PYTHONSTARTUP` for this; that mechanism is non-functional under non-interactive `vllm serve` and is replaced. |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | Enables the `hf-transfer` Rust downloader baked into the image — roughly 3–5× faster for large shards. |
| `VLLM_USE_V1=1` | Pins the V1 engine (default at this vLLM commit, but explicit pin defends against future defaults). |
| `--model RedHatAI/Qwen3.6-35B-A3B-NVFP4 --revision e850c696e6d75f965367e816c16bc7dacd955ffa` | Pins the NVFP4 checkpoint by revision SHA (RedHatAI marks it "preliminary (and subject to change)"; SHA pin prevents silent upgrades). |
| `--max-model-len 131072` | 128K max context. See §5.2. Concurrent-KV capacity at gmu 0.92 is ~63,360 tokens — a single request can stretch to 131K, but not two concurrently. |
| `--gpu-memory-utilization 0.92` | Empirical knee. Measured 28.5 GiB peak VRAM leaves ~1.5 GiB of Triton cold-start headroom; gmu 0.95 risks warmup OOM, gmu 0.90 wastes ~640 MiB of KV pool. |
| `--max-num-seqs 4` | Caps concurrent sequences. For single-user agent loops 1–2 is typical; 4 leaves room for client-side speculative tool-call retries without OOM. |
| `--max-num-batched-tokens 8192` | No longer pinned at 2096 — the older Mamba/GDN alignment requirement (vLLM #37714) is not enforced in the pinned nightly. 8192 matches the public tasticleeze config and boots cleanly. |
| `--moe-backend flashinfer_cutlass` | Routes NVFP4 MoE through FlashInfer's CUTLASS `cutlass_scaled_fp4_mm_sm120a` kernel (native SM 12.0 FP4 Tensor Cores). Canonical flag uses hyphen; vLLM also accepts `--moe_backend` as a synonym. |
| `--reasoning-parser qwen3` | Activates server-side `<think>...</think>` extraction into the response's `reasoning` field. Required for the `<think>` block to be split from visible content. |
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Activates the dedicated 683-line Qwen3.6 XML tool parser for `<tool_call><function=NAME><parameter=KEY>VALUE`. |
| `--enable-prefix-caching` | Enables prefix-hashing. Because Qwen3.5MoE is `IsHybrid` without `SupportsMambaPrefixCaching`, this silently auto-forces `mamba_cache_mode=align` and chunked prefill (both default on). Not a problem — just be aware the config is adjusted under you. |
| `--default-chat-template-kwargs '{"preserve_thinking": true}'` | Sets `preserve_thinking=true` as the server-wide default (§5.7). Eliminates the need for every client to send this in `chat_template_kwargs`. Per-request values still override the default. |
| `--limit-mm-per-prompt '{"image": 2, "video": 1, "audio": 0}'` | **Load-bearing, not hygiene.** Omitting this flag defaults to 999 per modality which **crashes the boot** on this nightly + SM 12.0 (post-profiling TensorRT-LLM runtime error; verified in a 2026-04-22 sweep). Values `image ≥ 10` hit the same crash. Safe range is `image ∈ {0, 1, 2, 3}`. Raising image from 2 to 3 costs zero KV pool (the profiler uses exactly one dummy image regardless of N; boot log: *"profiled with 1 image items of the maximum feature size"*). Setting all three modalities to 0 reclaims the ~1.56 GiB vision-encoder profiling reservation and raises the pool from 4.88 → 6.44 GiB / 63,360 → 83,424 tokens — valid only if you never need vision. Audio stays 0: Qwen3.6 is text+vision only. See §5.8 for the full cost accounting. |

**Flags we deliberately do NOT pass — each with a reason**:

| Omitted flag | Why not |
|---|---|
| `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` | MTP disabled by choice after measurement (§5.3). 166 tok/s vs 153 tok/s is only ~8% faster and costs 38% of the KV pool. If you change your mind, append this flag — it works. |
| `--quantization` | Auto-detected from `config.json`'s `quantization_config.quant_method: "compressed-tensors"` + `format: "nvfp4-pack-quantized"`. Passing it explicitly is redundant. |
| `--dtype` | Default resolves to BF16 from `config.json:dtype`. NVFP4 W4A4 is orthogonal to this: quantized tensors stay FP4 at compute, non-quantized (vision, norms, lm_head) stay BF16. |
| `--kv-cache-dtype` | Default `auto` resolves to BF16 — our intended KV dtype (§5.1). Passing `fp8` would contradict the KV decision. |
| `--trust-remote-code` | `Qwen3_5MoeForConditionalGeneration` is a native in-tree vLLM model class. No custom Python from the HF repo is executed. |
| `--attention-backend` | Default Triton/FlashInfer dispatch is correct on SM 12.0. `FLASH_ATTN_V3` has no Blackwell release. |
| `--tensor-parallel-size` / `--enable-expert-parallel` | Single-GPU deployment. TP and EP require multi-GPU. |

### 8.3 The patch files referenced above

The eight files mounted by §8.2 are documented in §7:

* §7.1 — `monkey_patch_qwen3_coder.py` (parser crash fix)
* §7.2 — `monkey_patch_hybrid_kv_allocator.py` (KV scheduler-budget fix)
* §7.3 — `monkey_patch_reasoning_field_egress.py` (wire-format rename, serialization side)
* §7.4 — `monkey_patch_reasoning_field_ingest.py` (wire-format rename, request side)
* §7.5 — `monkey_patch_tool_call_in_think_rescue.py` (mid-`<think>` tool-call rescue, both streaming and non-streaming)
* §7.6 — `monkey_patch_extract_tool_calls_metrics.py` (non-streaming silent-failure observability)
* §7.7 — `monkey_patch_extract_tool_calls_streaming_metrics.py` (streaming silent-failure observability)
* §7.L — `launch_with_patches.py` (entrypoint that imports the seven patches before `vllm serve`)

Remove individual patches as their upstream fixes land — each patch's import-time landmark check refuses to apply against an already-fixed function and tells the operator exactly what to do.

### 8.4 Smoke tests

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
    "model": "Qwen3.6-35B-A3B",
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

Expected shape in response (after the §7.3 egress patch is installed — the field name on the wire is now OpenAI-standard):
- `choices[0].message.reasoning_content` contains the `<think>` content.
- `choices[0].message.tool_calls[0].function.name == "calculator"` and `arguments` is a JSON string parseable to `{"expr": "127 * 349"}` or similar.
- `choices[0].finish_reason == "tool_calls"`.

Note that the request body above explicitly sets `chat_template_kwargs` per-request for clarity. In practice, the server-side default (§5.7, `--default-chat-template-kwargs '{"preserve_thinking": true}'`) already applies `preserve_thinking=true`; only `enable_thinking: true` needs to come from the request.

**Operator visibility for silent tool-parser failures**: scrape `vllm_qwen3_coder_silent_tool_call_failures_total{failure_kind, model}` from the server's Prometheus endpoint. Split the time series by `failure_kind` label to distinguish non-streaming (`markup_leak`) from per-delta streaming leaks (`markup_leak_streaming`). Non-zero rate here indicates a residual parser silent-failure rate the model exhibited despite the `monkey_patch_qwen3_coder.py` + `monkey_patch_tool_call_in_think_rescue.py` mitigations; investigate at the model-prompt level rather than the patch level.

---

## 9. Known unknowns

Documented items we did not conclusively resolve and which may affect production quality. Flagging explicitly so they are not forgotten.

| Item | Evidence status |
|---|---|
| No published agentic benchmarks (SWE-Bench Verified / Aider Polyglot / OpenHands) for **Qwen3.6-35B-A3B-NVFP4** specifically | Model is 5 days old, NVFP4 quant 4 days old. RedHatAI publishes GSM8K-Platinum 96.28% vs 95.62% BF16 → **100.69% recovery** (so NVFP4 is at BF16 parity on a single reasoning benchmark). The **73.4 SWE-Bench Verified** number is reported by Qwen in the `Qwen/Qwen3.6-35B-A3B-FP8` model card, which also claims metrics are "nearly identical" to BF16. NVFP4 agentic delta vs FP8/BF16 is unmeasured — run your own eval harness. |
| RedHatAI "preliminary (and subject to change)" status on the quant | Pinned by revision SHA `e850c696e6d75f965367e816c16bc7dacd955ffa`. Safe until we deliberately upgrade. Re-check for new commits before any redeployment. |
| FlashInfer autotuner non-fatal warnings on SM 12.0 | At boot, FlashInfer logs `[Autotuner]: Skipping tactic ... Failed to initialize cutlass TMA WS grouped gemm` for some `M128`/`M256` grouped-GEMM shapes lacking SM 12.0 kernels. FlashInfer falls back silently to working tactics and performance is as measured (~153 tok/s decode). If you see a perf regression after an image bump, try `--moe-backend flashinfer_cutedsl` (FP4-only, SM 12.0-native) as the contingency. With MTP on, additional `GPU lacks the shared memory resources to run fused_moe kernel` warnings appear — non-fatal, same fallback mechanism. |
| Long-context retrieval quality past ~64K | Only 10 of 40 layers are full attention, so long-range retrieval rides on a thin substrate. No RULER / needle-in-haystack numbers published for Qwen3.6 at any precision. We verified a 119,907-token single request is *mechanically* accepted and returns HTTP 200, but we did not verify answer quality at that length. Validate on your workload before committing to 128K retrieval. |
| `<tool_call>`-inside-`<think>` frequency rate, and thinking-mode loop rate under our specific NVFP4 config | `<tool_call>`-inside-`<think>` is community-reported at single-digit-percent; no precise measurement. Our rescue patch (§7.5) handles it via the deferred-flush state machine on the streaming path and a simple reasoning-to-content splice on the non-streaming path, but frequency could change with future Qwen RL tuning. Watch the server's `vllm_qwen3_coder_silent_tool_call_failures_total` Prometheus series for any residual silent-failure rate after the rescue applies. **Thinking-mode loop rate** (the distinct, broader Qwen3-family pathology where the policy enters a repetition attractor until the token budget runs out) is measured at 17.4% of outputs / 27.5% on hard problems in Qwen's own LiveCodeBench run against Qwen3.5-35B-A3B (issue #88 on `QwenLM/Qwen3.6`) — but that measurement was against unquantized weights. We have not measured the rate under our NVFP4 + BF16-linear-attn + BF16-KV config against a BF16-weights baseline, so we do not know whether NVFP4 amplifies, suppresses, or leaves the rate unchanged. A ~100-prompt hard-reasoning eval comparing the two would be the first such measurement published and is a ~1-afternoon piece of work. |
| Concurrent-KV capacity at 131K max_model_len | Measured 63,360 tokens concurrent at gmu 0.92, MTP off, with vision enabled. A single long request up to 131K works (verified at 119,907 and 125,642 tokens, HTTP 200). Two concurrent long requests will start evicting when their combined tokens cross ~63K. Not a problem for single-user agents; document explicitly because the advertised 131K is not "131K × N concurrent". The pool is ~4× smaller than the attention math would allow, because of the hybrid-KV allocator bug in §6.15 — a real, measurable upstream cost, not a perceived one. |
| Hybrid KV allocator inflation (§6.15) | **Largest single upstream cost in this deployment.** 75% of our KV pool is held for the 30 DeltaNet layers that don't actually need per-token KV. Tracked at vLLM #37121, fix proposed in PR #37429 but blocked on design review. No workaround in vLLM today; SGLang lacks this bug but trades away other integrations. We post measurements to #37121 and watch for the merge. |
| MTP correctness under our specific stack vs broader reports | We measured MTP on this hardware + quant + max_model_len + prefix caching combination as functional (87-96% acceptance, coherent output). vLLM issue #36872 (open) predicts gibberish on FP8/AWQ — we did not reproduce on NVFP4. Issue #38182 predicts prefix-cache hit-rate regression — we did not stress-test this specifically. If the user enables MTP, monitor these two failure modes before declaring it production-ready. |
| Image count boot failure threshold (§5.8.2) | Measured boots at `image ∈ {0, 2, 3}` succeed; `image ∈ {10, 100, 999}` and the default-999-when-flag-omitted all fail with a TensorRT-LLM `throwRuntimeError` during CUDA graph capture (post-profiling). We did not bisect 4–9; treat 3 as the ceiling for now. |

---

## 10. What this deployment explicitly does not use

Documenting the "not chosen" list because public repos that show only the chosen path leave readers wondering if alternatives were considered.

| Option | Status | Reason not chosen |
|---|---|---|
| llama.cpp | Not chosen | No dedicated Qwen3 tool or reasoning parser in `common/chat.cpp` at master (verified 2026-04-22 against commit `0d0764df`); generic `peg-native` has four live OPEN failure modes on Qwen3.x (llama.cpp #20260, #20837, #21771, #22240). MTP silently dropped at conversion (`convert_hf_to_gguf.py:4781-4782`). Vision resize is BILINEAR (`tools/mtmd/clip.cpp:1357`); HF reference is BICUBIC. Agentic prompt-cache regression on 27B-class models with `--cache-ram` is open (#21383). Full citations and closed-upstream-since list in §5.4. |
| SGLang | **Strongly considered, not chosen** | **Has a measurable advantage for this model**: SGLang's memory-pool implementation allocates a separate `MambaPool` for DeltaNet state, so it does NOT suffer from the hybrid KV over-reservation described in §6.15. On the same hardware and weights, SGLang would report roughly 3–4× the concurrent-KV capacity at BF16 KV. We still chose vLLM because (a) SGLang has no dedicated `qwen3_coder` tool parser — tool-call correctness is our top priority; (b) no `flashinfer_cutlass` NVFP4 MoE dispatch path with the exact kernels we validated; (c) the §7 runtime patch (`monkey_patch_qwen3_coder.py`) is targeted at vLLM's parser surface and would need re-validation against SGLang's. For a single-user agent workload the tool-parser argument dominates; for a multi-user long-context workload where concurrent-KV is binding, SGLang would be the right choice. Re-evaluate if vLLM PR #37429 stalls past Q2 2026, or if concurrent-KV becomes the binding constraint for our workload. |
| BF16 weights (full precision) | Not feasible | ~70 GB does not fit on 32 GB. |
| FP8 weights (`Qwen/Qwen3.6-35B-A3B-FP8`) | Not feasible | ~35 GB exceeds 32 GB once KV + activations are added at any useful context length. |
| **AWQ 4-bit** (`QuantTrio/Qwen3.6-35B-A3B-AWQ`) | **Runner-up (fallback)** | Works via `awq_marlin` on SM 12.0; vision preserved; data-free quantization, no calibration dataset published, no quality-recovery number. Kernel path is INT4 Marlin emulation rather than native SM 12.0 FP4. Kept as documented fallback if NVFP4 hits an unexpected loader bug. Launch command would use `--quantization awq_marlin --dtype bfloat16 --kv-cache-dtype auto` in place of the NVFP4-specific flags. |
| Other community AWQ / GPTQ / MXFP4 / PrismaQuant variants | Rejected | Individually evaluated in §3.1. Reasons ranged from missing vision tensors (`caiovicentino`) to suspicious empty `ignore` lists (`Intel AutoRound`, `palmfuture`) to insufficient calibration (`rdtand`) to `compressed-tensors actorder=null` bug exposure (`cyankiwi`). |
| MXFP4 weights | Not beneficial | CUTLASS MXFP4 MoE is gated to SM 10.0; on SM 12.0 it falls back to Triton/Marlin dequant — no FP4 tensor-core benefit over Marlin. |
| FP8 KV cache | Not chosen | See §5.1. Uncalibrated scales cause long-context quality drift; the reasoning model in an agentic loop is the wrong workload to pay that quality tax on. |
| TurboQuant / RotorQuant KV cache | Not available *yet* | TurboQuant's dense PR #38479 merged 2026-04-15 but excludes hybrid models; the hybrid follow-up is vLLM PR #39931 (open, ready-for-review, names Qwen3.6-35B-A3B in its test matrix, independently validated on RTX 5090 by reviewer `@jhsmith409` across all four presets including `turboquant_4bit_nc`). Realistic merge: mid-May 2026. Measured 4-bit `turboquant_4bit_nc` quality on the closest model family (Qwen3.5-35B-A3B, H20): GSM8K 0.830 vs BF16 0.855 = −2.5 pp. 3-bit is −24 pp and rejected. RotorQuant is a separate single-author research project by Scrya (March 2026); not integrated into vLLM or SGLang and not in scope. |
| MTP speculative decoding | Disabled by explicit choice after measurement | §5.3 has the measured comparison: MTP-on is 166 tok/s vs MTP-off 153 tok/s (~8% faster decode) but costs 38% of concurrent-KV capacity. Appending `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` does enable it cleanly if you want it. MTP tensors are auto-skipped at load when the flag is absent (`qwen3_5.py:547` has `skip_prefixes=["mtp."]`). |
| Parallel tool calls via Responses API | Not used | vLLM issue #39584 crashes the Responses-API streaming path. Chat Completions is universally supported across OpenAI-SDK clients and sidesteps the bug entirely. |
| `--enable-chunked-prefill` as an explicit flag | Implicit | Default on; auto-forced by `--enable-prefix-caching` on this hybrid model. No need to specify. |

---

## 11. Update cadence

Re-evaluate the pinned versions in this README when any of the following happens:

1. **vLLM tags a `v0.19.2` final** or later, at which point we migrate from the nightly image to a semver-tagged image.
2. **A newer nightly image passes the `docker run --rm <image> vllm --help` smoke test** (i.e., does not exhibit the §6.14 transformers import regression). Upgrade the pinned digest.
3. **Upstream PR #39772 merges** — remove `monkey_patch_qwen3_coder.py` (patch 1) and its bind-mount; remove the launcher's `monkey_patch_qwen3_coder` entry in `_PATCH_MODULES` and the corresponding verifier in `_PATCH_VERIFICATION`. The patch's source-landmark check refuses against a fixed function and tells the operator exactly this.
4. **vLLM issue #38182 (MTP + prefix cache) closes with a verified fix** — reconsider enabling MTP speculative decoding. Our measurement shows MTP on this stack gives ~8% decode speedup (153 → 166 tok/s) at the cost of 38% of concurrent-KV capacity; if that tradeoff changes materially (e.g., larger acceptance rate, smaller VRAM cost), re-run the experiment.
5. **RedHatAI removes the "preliminary (and subject to change)" notice** on the NVFP4 quant, or publishes a v1.0-tagged revision — bump the pinned revision SHA.
6. **An `nvidia/Qwen3.6-35B-A3B-NVFP4` (NVIDIA-official) checkpoint is published** — reconsider which NVFP4 build to use, especially if NVIDIA publishes agentic-task benchmarks alongside it.
7. **vLLM PR #39931 (hybrid TurboQuant) merges** — reconsider moving from BF16 KV to rotation-preconditioned low-bit KV to unlock higher concurrent-KV capacity on this card. Our current reading: 4-bit `turboquant_4bit_nc` costs ~2.5 pp GSM8K on the closest model family (Qwen3.5-35B-A3B); 3-bit costs ~24 pp and is rejected; 4-bit is defensible if our workload saturates concurrent-KV and we accept the quality tax.
8. **vLLM PR #40384 (narrow hybrid KV scheduler fix) merges** — remove `monkey_patch_hybrid_kv_allocator.py` (patch 2) and its bind-mount; remove the launcher's `monkey_patch_hybrid_kv_allocator` entry in `_PATCH_MODULES` and the corresponding verifier. The patch's source-landmark check refuses against a fixed function and tells the operator exactly this. We are no longer waiting for this PR — the in-repo backport delivers its substance today; the upstream merge is a cleanup step.
9. **vLLM PR #37429 (broader hybrid KV allocator fix) merges** — fixes the underlying byte-level pool over-reservation that PR #40384 / patch 2 do NOT address. **Critical sequencing**: remove patch 2 BEFORE pulling an image that contains PR #37429 — the broader fix changes the underlying pool byte size, and patch 2's scheduler view would no longer match. The patch's landmark refuses on its own buggy-source check, but `vllm.v1.core.kv_cache_utils._report_kv_cache_config` may still carry the buggy substring even after #37429 changes the pool's physical layout (the two functions are related but distinct). Audit before bumping.
10. **PR #37429 lands and patch 2 has been removed** — the pool itself shrinks in bytes consumed (freeing VRAM for further use). Re-measure §5.2 / §5.8 numbers; the post-#37429 pool may be smaller in bytes but admit similar concurrent tokens (the inflation is moved out of the pool entirely rather than counted-around).
11. **vLLM ships OpenAI-standard `reasoning_content` on ChatMessage / DeltaMessage natively** — remove `monkey_patch_reasoning_field_egress.py` (patch 3) and its bind-mount; remove the launcher entries. Landmark #5 (no existing `reasoning_content` field declaration) refuses against an upstream that added the field. Concurrent: if upstream also widens ingest acceptance to include `reasoning_content`, remove `monkey_patch_reasoning_field_ingest.py` (patch 4) in the same pass — its landmark #4 refuses against a restructured `_parse_chat_message_content` body.
12. **vLLM adopts a reasoning parser that re-routes `<tool_call>` markup itself** — remove `monkey_patch_tool_call_in_think_rescue.py` (patch 5) and its bind-mount; remove the launcher entries. No upstream PR is tracking this today; operator must notice and decide.
13. **vLLM ships first-class metrics for tool-parser silent failures** — remove `monkey_patch_extract_tool_calls_metrics.py` (patch 6) AND `monkey_patch_extract_tool_calls_streaming_metrics.py` (patch 7) and their bind-mounts; remove the launcher entries. No source landmark refuses for these patches since there is no upstream bug landmark to check against — operator must notice the upstream metric and decide.

Each of these is tracked and none of them are urgent. The current pins booted, served a chat completion + tool call, held a 119,907-token single-request prompt, and measured 153 tok/s steady-state decode on the target hardware on 2026-04-21. 2026-04-22 cross-validation added 5 multimodal-config boot sweeps, 4 high-N image-limit sweeps (3 of which crashed, confirming the `image ≥ 10` ceiling), 8 request-time VRAM probes up to 128,333 tokens, and the empirical decomposition of the KV pool — all documented in §5.8 and §6.15.

---

## 12. File structure of this project

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

Eight files, all Python, all server-side. Seven runtime monkey-patches plus one container entrypoint that loads them in the required order. **No `client/` subtree; no client-side code.** Off-the-shelf OpenAI-compatible clients (Qwen Code CLI, Qwen-Agent, OpenAI Python SDK) connect to the patched server unmodified. There is no `docker/` or `tests/` subtree; the docker run command (§8.2) and smoke tests (§8.4) live inline in this README. Any additional artifacts are created alongside as the deployment evolves.
