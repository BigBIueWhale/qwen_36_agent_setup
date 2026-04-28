# Qwen3.6 on a single RTX 5090 — production-grade agentic deployment

**Last updated: 2026-04-28.** `QuantTrio/Qwen3.6-27B-AWQ` (dense VL, AWQ INT4) on **vLLM** (llama.cpp rejected, §5.4). Multi-tool agentic pipelines with vision, reasoning, preserved-thinking. Host: single NVIDIA RTX 5090, 32 GB VRAM, Blackwell SM 12.0, Linux, CUDA 13.0, nvidia-container-toolkit v1.19.0.

This README documents a specific, pinned, reproducible deployment. Every choice below is deliberate.

### Deployment target & load-bearing pins

**`QuantTrio/Qwen3.6-27B-AWQ`** (selected 2026-04-25, replacing an earlier `RedHatAI/Qwen3.6-35B-A3B-NVFP4` target). The 27B dense model is more capable per forward pass than a 35B-A3B MoE that activates only ~3B per token, fits on the card cleanly, and the AWQ recipe preserves the load-bearing layers (vision, `linear_attn.in_proj_a/b`, `lm_head`, embeddings, layer 0, MTP) at BF16 while quantizing MLPs. NVFP4 builds for 27B-dense were triaged and rejected — see §3.1.

| Slot | Pin |
|---|---|
| Model | `QuantTrio/Qwen3.6-27B-AWQ` at revision `9b507bdc9afafb87b7898700cc2a591aa6639461` |
| Quantization format | AWQ INT4 (gemm, group_size=128, zero_point=true), data-free calibration. Vision encoder, `linear_attn.in_proj_a/b`, all `self_attn.{q,k,v}_proj`, layer 0, embeddings, `lm_head`, MTP, norms/conv1d kept BF16 |
| Runtime | vLLM Docker image `vllm/vllm-openai@sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba` (build `0.19.2rc1.dev212+g8cd174fa3`, commit `8cd174fa358326d5cc4195446be2ebcd65c481ce`, container CUDA 13.0.2, PyTorch `2.11.0+cu130`, FlashInfer `0.6.8.post1`, transformers `5.6.2`, pydantic `2.13.3`, image built 2026-04-26 05:19 UTC) |
| KV cache dtype | BF16 (FP8 KV scales are not shipped with this checkpoint; rationale §5.1) |
| `--max-model-len` | **152,000** |
| Disk size | 20.36 GiB |
| VRAM at boot (measured) | 19.78 GiB resident (MTP head auto-skipped via `qwen3_5.py:701-706`'s `skip_prefixes=["mtp."]`) |
| KV pool available | 9.7 GiB at gmu=0.97 with `--skip-mm-profiling`, `--mm-processor-kwargs '{"max_pixels": 4194304}'`, `--max-num-batched-tokens 4096`. Boot log: `GPU KV cache size: ~158,368 tokens` (patch 3 installed; unpatched §6.3 would display ~40K) |
| Per-token attention KV at BF16 | `4 KV heads × 256 head_dim × 2 (K+V) × 16 attn layers × 2 bytes = 65,536 bytes/token` (16 GiB at the model's native 262K context) |
| Maximum concurrency at full max-len | **1.04×** at the §8.2 production flags (one 152K request fits with ~4% slack — `monkey_patch_hybrid_kv_allocator.py:336`). gmu=0.97 chosen over 0.98 to give the PyTorch allocator ~300 MiB cold-path coalesce headroom; otherwise the first 4-MP request after boot OOMs the LM-prefill MLP buffer despite total free memory being adequate (§5.2, B12) |
| `preserve_thinking` and `enable_thinking` | Both set true server-wide via `--default-chat-template-kwargs '{"preserve_thinking": true, "enable_thinking": true}'` (§5.7) |
| MTP speculative decoding | OFF (head present in the AWQ checkpoint as BF16 but auto-skipped at load; §5.3) |
| Patches in this repo | 10 strict, fail-loud Python monkey-patches (§7), all server-side, loaded before `vllm serve` starts. Zero client-side code |

---

## 1. What this project is

A production deployment of **`QuantTrio/Qwen3.6-27B-AWQ`** (a 27-billion-parameter dense vision-language model from Alibaba, AWQ INT4 quantized) served via **vLLM** behind an OpenAI-compatible HTTP API at `http://127.0.0.1:8001/v1/chat/completions`, intended for agentic coding via the Qwen Code CLI on a single RTX 5090.

Five non-negotiable correctness requirements:

1. **Tool calling that actually works in multi-turn agent loops** — no silent failures.
2. **Preserved thinking across tool turns** — historical `<think>` blocks remain visible to the model on subsequent turns.
3. **Vision input at full preprocessing fidelity** — vision encoder kept BF16; HF `Qwen3VLImageProcessor` semantics preserved.
4. **152,000-token `--max-model-len` at BF16 KV cache precision** (FP8 KV rejected; §5.1).
5. **Stability over a marginal throughput gain** — MTP off; rationale §5.3.

Every software pin, every launch flag, every server-side monkey-patch in this repo exists to uphold them simultaneously.

---

## 2. Hardware

| Component | Spec | Why it matters |
|---|---|---|
| GPU | NVIDIA RTX 5090, 32 GB VRAM | Largest consumer Blackwell card. 32 GB is the binding constraint: dictates 4-bit AWQ weights and the §3 `--max-model-len` ceiling. |
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
- **Native context**: `max_position_embeddings=262144` (we cap at 131,072; rationale §5.2).
- **Vocabulary**: `vocab_size=248320`.
- **Rotary positional embedding**: M-RoPE; used for both text and vision tokens.
- **Vision encoder**: 27-layer Qwen3-VL tower, BF16, bundled into the main checkpoint.
- **MTP (Multi-Token Prediction) head**: 1-layer module, BF16, present in the AWQ checkpoint (~0.68 GiB) but auto-skipped at load. See §5.3.

The hybrid attention pattern is why the KV-cache memory math diverges from a pure transformer: only 16 of 64 layers contribute to KV growth with context.

---

## 5. Decisions that shape the deployment

### 5.1 KV cache: BF16, not FP8

`--kv-cache-dtype auto` (resolves to BF16). FP8 KV halves KV memory but requires per-tensor or per-head scaling factors to preserve numerical range. Qwen3.6-27B-AWQ does not ship calibrated FP8 KV scales — vLLM falls back to scale=1.0, which is the documented cause of mild long-context quality drift. SGLang's docs flag the same hazard: *"these FP8 checkpoints do not include pre-calibrated KV cache scaling factors; SGLang defaults to scale 1.0, which may cause noticeable accuracy degradation on reasoning-heavy tasks."* For a reasoning model in an agentic pipeline this is unacceptable. Rotation-based low-bit KV (TurboQuant) does not yet support hybrid attention + GDN architectures in vLLM.

### 5.2 Max context length: 152,000 tokens

`--max-model-len 152000`. The byte budget on a 32 GiB card at gmu=0.97 with `--skip-mm-profiling` (vision capability preserved at runtime; §5.8) and `--max-num-batched-tokens 4096` leaves **9.7 GiB** for KV — the boot log reports `GPU KV cache size: 158,368 tokens` (patch 3 §7.3 installed; the unpatched line would read ~40K under #6.3). At the per-token attention KV pinned in §3 (65,536 bytes/token), one full-context request fits with **1.04× concurrency** — the most context we can keep within the runtime activation envelope while leaving the PyTorch allocator enough cold-path coalesce headroom to admit a 4-MP image as the *first* request after boot. The model's native 262K context would need 16.0 GiB of attention-only KV; that doesn't fit without dropping precision (FP8 KV rejected, §5.1).

`--gpu-memory-utilization 0.97`. The hard envelope ceiling on this hardware is `floor(free_memory / total_memory × 100) / 100 = 0.98` — vLLM's snapshot check at `vllm/v1/worker/utils.py:412` compares `requested = total × gmu` against `init_snapshot.free_memory`, which is taken **after** vLLM's own ~510 MiB CUDA-context init (PyTorch + cuBLAS/cuDNN + FlashAttention v2 + NIXL). That semantic — penalising vLLM for its own footprint — is closed by patch §7.8, which lets us cleanly land at 0.98. **We deliberately leave one percent on the table at 0.97**, not 0.98, because PyTorch's CUDA caching allocator can leave the activation pool fragmented on the first prefill after boot: a single 4-MP image-bearing request would need ~132 MiB contiguous for the LM-prefill MLP intermediate buffer (`(s≤mnbt=4096, intermediate=17408)` fp16) and at gmu=0.98 the freshly-booted allocator has only ~140 MiB free + ~127 MiB reserved-but-unallocated split across non-coalescable segments — total free ~267 MiB but no contiguous 132 MiB block is forgeable, OOM. Subsequent requests would coalesce fine; the breakage is purely cold-path. `expandable_segments:True` (§8.2) is necessary but not sufficient — it permits segment growth but doesn't pre-grow on cold start. The empirical proof is in §11 B12: at gmu=0.98 the first 4-MP cold request OOMs deterministically; at gmu=0.97 (which adds ~300 MiB of cold-path coalesce headroom by reducing the upfront KV-pool reservation) the same workload passes cleanly. The 1.04× full-context concurrency at gmu=0.97 still carries every realistic admission case (single-tenant agentic workload — a second 152K request waits in pending, no functional regression).

`--max-num-batched-tokens 4096`. vLLM's `_dummy_run(max_num_tokens, is_profile=True)` reserves activation peak ~linear in this number for the LM forward, and `vllm/config/scheduler.py:236` sets `encoder_cache_size = max_num_batched_tokens` exactly. At mnbt=8192, 130K-prompt + 20-image (=82K vision token) workloads OOM at the LM-prefill MLP intermediate buffer — `(s≈7884, intermediate=17408)` fp16 = ~262 MiB — by ~5 MiB on this hardware (verified empirically against a real high-resolution Pillars-of-Creation photograph + 47K text payload, §11 row B9). Dropping to **mnbt=4096 halves that buffer to ~142 MiB** and reroutes the freed activation budget into the KV pool at boot — at gmu=0.97 the boot KV pool sizes to 158,368 tokens, sufficient to admit a 152K full-context request at 1.04× concurrency. Cost: cold full-context prefill is ~10–15% slower (32 prefill chunks vs 16 for a 130K prompt; warm prefix-cache flows unaffected — most agent turns), and `encoder_cache_size=4096` fits one concurrent 4-MP image instead of two (irrelevant for byte-identical images via mm_hash dedup; ~2× slower encoder for fully-unique multi-image flows).

The admission gate (`scheduler_reserve_full_isl=True`, default) refuses any request whose ISL doesn't fit in the free KV pool — chunking is *within*-request, not *across* — so for a single in-flight max-len request to admit cleanly the pool must hold the full ISL. 158,368 ≥ 152,000 with ~6,400 slack — the 1.04× concurrency above.

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

Per the official QuantTrio/Qwen3.6-27B-AWQ HuggingFace model card "Best Practices" block, thinking-mode precise-coding/math values: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0, max_tokens=81920`. The "general tasks" thinking-mode alternative uses `temperature=1.0` (rest unchanged); 81,920 is the official recommendation for **highly complex math / programming-competition outputs** and is well inside `max_position_embeddings=262144` — no quality cliff vs the 32K-standard-queries alternative. Standing workload here is autonomous agentic coding + math, so we default high.

**Patch §7.6 enforces these defaults server-side** for fields the client did NOT explicitly send (Pydantic v2 `model_fields_set`); explicit client values pass through unchanged, so a client that wants free-form generation sends `temperature=1.0` and that wins. Verified empirically against `qwen-code/packages/core/src/core/openaiContentGenerator/pipeline.ts:411-432` — the `addParameterIfDefined` helper there sends NONE of these fields unless the user has explicitly configured them, so the patch lands on every default-config qwen-code request.

Earlier patch versions enforced `presence_penalty=1.5` based on Alibaba's documented mitigation for the **Qwen3.5-35B-A3B-MoE** thinking-mode loop pathology (LiveCodeBench: 17.4% of 1,400 outputs truncated). That recommendation does NOT apply to **Qwen3.6-27B-AWQ**: the official Qwen3.6 model card publishes `presence_penalty=0.0`, and the QuantTrio AWQ recipe additionally preserves `linear_attn.in_proj_a/b` at BF16 (§3.1), which `cpatonn`'s HF discussion #2 found *"significantly better on the infinite loop issue"* vs full INT4. Empirical loop rate on this specific config is in §11's known-unknowns; we follow Alibaba's published recommendation.

Qwen3.6 does NOT support soft `/think` or `/nothink` switches — thinking mode is controlled exclusively via `chat_template_kwargs.enable_thinking`. We default it server-side to `true` (§5.7) so every default-config request lands in thinking mode.

### 5.7 `preserve_thinking=true` and `enable_thinking=true` as server-side defaults

`--default-chat-template-kwargs '{"preserve_thinking": true, "enable_thinking": true}'` sets both as server-wide chat-template kwargs. Both are explicitly set:

- **`preserve_thinking=true`** — the Qwen3.6 chat template only emits `<think>` blocks from historical assistant turns when this is true; otherwise only the most recent assistant turn retains its `<think>` (template line 100). The Qwen Team HF model card states Qwen3.6 was *"additionally trained to preserve and leverage thinking traces from historical messages"* and that the capability is *"particularly beneficial for agent scenarios"* (Qwen3.6-27B README L811 and L839); the official Qwen-Agent reference snippet at L869 sets `preserve_thinking=True` for both DashScope and vLLM/SGLang. Predecessor Qwen3 documented the *opposite* (strip-history) as Best Practice — Qwen3.6 is a deliberate inversion. No controlled benchmark comparing `true` vs `false` has been published; the only public anecdote we are aware of is `badlogic/pi-mono#3325` (single user, Qwen3.6-35B-A3B + LM Studio: empty-arg tool calls after 2–3 turns without the flag), with two follow-ups (`pi-mono#3479`, `jundot/omlx#900`) reporting that the flag is necessary but not always sufficient. We default it `true` as the Qwen-recommended agentic shape (§11 B17 audit).
- **`enable_thinking=true`** — functionally equivalent to leaving it unset because the chat template at lines 147-154 only suppresses thinking on explicit `false`, but setting it explicitly locks intent at the server. The mechanism is structural: when `enable_thinking=false` the template emits `<|im_start|>assistant\n<think>\n\n</think>\n\n` as the assistant prefix (a pre-closed empty think block); when true (or unset) the prefix is `<|im_start|>assistant\n<think>\n` (model continues an open block). Empirically strict on this deployment: **0 leaks across 65 T=0.6 trials** spanning easy/medium/hard prompts, lengths 5K–132K tokens, and math/code/retrieval intents (§11 B13).
- **Per-request opt-out — wire shape matters.** Qwen3.6's model card documents the toggle as `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`. **That's an OpenAI Python SDK convention** — the SDK unwraps `extra_body` client-side into top-level body fields before sending. vLLM's request schema has `extra="allow"`, so a raw `curl` / `requests.post` body with `extra_body`-nested kwargs is silently accepted and dropped, falling through to the server default. **For raw HTTP, `chat_template_kwargs` must be at the TOP LEVEL of the JSON body**:

  ```json
  {"model": "Qwen3.6-27B-AWQ", "messages": [...], "chat_template_kwargs": {"enable_thinking": false}}
  ```

- **`/think` and `/nothink` system-message switches do NOT work** on Qwen3.6 — the model card at L838-839 explicitly disclaims them, and §11 B13 confirmed 5/5 leak (zero suppression) when used; an explicit "don't use chain-of-thought" instruction was *counter-productive* (increased reasoning length on hard prompts as the model deliberated about whether to comply).
- **Streaming caveat.** vllm-project/vllm issue #40816 reports leakage of the answer into `delta.reasoning` instead of `delta.content` under streaming on a different Qwen3.6 size (35B-A3B-NVFP4) even with `enable_thinking=false` set. We did not test streaming on the 27B-AWQ deployment; non-streaming was clean.

The separate `reasoning` vs `reasoning_content` wire-format mismatch (§6.1 ingest, §6.4 egress) is closed by the §7.1 and §7.4 patches.

### 5.8 Multimodal cost accounting

- **Vision encoder weights** (~0.83 GiB) are always resident in VRAM, whether or not any request contains an image. Subsumed into the boot weights footprint.
- **Vision encoder profiling at boot is intentionally skipped** via `--skip-mm-profiling` (§8.2). The flag bypasses `gpu_model_runner.profile_run()`'s encoder dummy pass (`vllm/v1/worker/gpu_model_runner.py:5754-5817`), reclaiming ~1.56 GiB of profiler reservation that maps to ~25.6K KV tokens. **Runtime vision is unaffected** — `_execute_mm_encoder` runs on demand on real images and writes into `encoder_cache[mm_hash]`. The flag's only safety cost is that vLLM no longer measures the encoder peak at boot, so it cannot cross-check that runtime peaks fit. We mitigate by capping per-image vision tokens (next bullet) so the runtime peak is bounded; the bound holds inside the gmu=0.97 budget.
- **Per-image vision-token cap**: `--mm-processor-kwargs '{"max_pixels": 4194304}'` (§8.2) bounds the largest admissible image at 4 megapixels (~2048×2048), which Qwen3-VL tokenizes into at most 4,096 vision tokens (`max_pixels / (patch_size × merge_size)² / merge_size² = max_pixels / 1024`). Without this cap, the default 16,777,216-pixel ceiling lets a single image consume 16,384 KV-pool slots and produces an unbounded `embed_multimodal` activation peak that would OOM `_execute_mm_encoder` unrecoverably with `--skip-mm-profiling` enabled. 4 MP covers screenshots and document images cleanly; photos at 4K resolution are downscaled at preprocessing.
- **Per-request transient activations** are small (+2–10 MiB measured for 896×896 to 1792×1792 images); absorbed by safety headroom.
- **Image and video tokens occupy the KV pool**: ~256 tokens for 512×512, ~787 for 896×896, ~3,136 for 1792×1792, up to **4,096 per image at the 4-MP cap above**. A small UI screenshot like `disc5_imgs/img1.png` (793×224) tokenizes to only ~155 vision tokens — the 4,096 figure is a per-image *ceiling*, not a typical cost. Videos scale dramatically — short clips ~750–1,500 tokens; medium clips ~10,000–20,000; long clips can exceed 50,000. Bound video size with `--mm-processor-kwargs '{"video": {"max_frames": 64, "fps": 1}}'` if accepting untrusted clips.
- **Per-request budget formula** (worst case, all images at the 4-MP cap): a request carrying `I` images can hold up to `131,072 − I × 4,096 − chat_template_overhead` text+system+completion tokens, where `chat_template_overhead` is ~23 tokens for a typical system+user turn (empirically measured by sweeping `max_tokens` against a 99,961-raw-token prompt: last-accepted boundary at `prompt + max_tokens = 131,072` exactly, with raw 99,961 → chat-formatted 99,984 = 23-token wrap delta). For images smaller than 4 MP, substitute the actual per-image vision token count from the bullet above; the formula then has more headroom. **Reject path: two ingest validators run synchronously before the engine sees the request, both fire in 0.15-0.22 s with HTTP 400, both leave engine state intact (`/health=200` after).**
  - `vllm/entrypoints/utils.py:174-185` (`get_max_tokens`): rejects when `prompt_tokens > max_model_len`. Message: `"Input length (X) exceeds model's maximum context length (152000)."`
  - `vllm/renderers/params.py:411-425` (`apply_length_check`): rejects when `prompt_tokens > max_model_len − max_output_tokens`. Message: `"This model's maximum context length is 152000 tokens. However, you requested K output tokens and your prompt contains at least M input tokens, for a total of at least K+M tokens."` Fires when the client explicitly sets `max_tokens`; if the client sends nothing (default qwen-code shape — `pipeline.ts:411-432` audited), `max_output_tokens=0` and the validator effectively reduces to the same `prompt ≤ max_model_len` check as the first.

`--limit-mm-per-prompt '{"image": 20, "video": 1, "audio": 0}'` is set generously. The flag is **load-bearing for boot** (omitting it defaults to 999 per modality, which crashes the boot on this nightly + SM 12.0 — post-profiling TensorRT-LLM `throwRuntimeError`), but otherwise it is **a pure per-request integer ceiling enforced at `chat_utils.py` ingest, not a memory reservation**. Verified: booting at `image:20` produces byte-identical KV reporting to `image:2` (9.7 GiB / 158,368 tokens / 1.04× concurrency at the §8.2 production flags); zero impact on boot, on text-only prompts, or on prompts carrying fewer images than the cap. The crash mode that fires at high N is *post*-profiling, which `--skip-mm-profiling` skips — values that would crash without the flag may also boot here, though we have empirically verified only up to 20. Set generously so realistic agentic flows never bump the ceiling; the cost of a high N is paid only when a request actually submits that many images (linear: each extra image at the 4-MP cap = 4,096 KV tokens, plus a sequential encoder forward beyond N=1 since `encoder_cache_size = max_num_batched_tokens = 4,096` fits exactly one 4-MP image at a time per `vllm/config/scheduler.py:236`). Audio stays 0: Qwen3.6 is text+vision only.

**Client-side modality gate hazard**. The Qwen Code CLI's `defaultModalities()` regex falls through to a text-only catch-all for `Qwen3.6-27B-AWQ` (and any `qwen-` model name not matching `qwen3-vl-*`, `qwen3.{5,6}-plus`, `qwen-vl-*`, or `coder-model`). With the default, the CLI's converter replaces every `image_url` part with `[Unsupported image file: …]` text **before the request leaves the client** — vision is silently disabled regardless of server config. Override in `settings.json`:

```json
{
  "model": {
    "name": "Qwen3.6-27B-AWQ",
    "generationConfig": {
      "modalities": { "image": true, "video": true },
      "splitToolMedia": true,
      "contextWindowSize": 131072
    }
  }
}
```

`splitToolMedia: true` on the client is **redundant** with patch §7.9 installed (the patch preserves list-shaped tool-role content with media so the chat template renders `<|vision_start|><|image_pad|><|vision_end|>` markers natively inside `<tool_response>` — exactly the training-distribution shape; see §7.9 for details). Enabling it on the client is harmless but no longer load-bearing. Without patch §7.9, qwen-code commit `414b330` (#3617)'s per-message-split path is the only mitigation: with its default `false`, MCP-returned screenshots and `read_file`-emitted images stay packed in the `role: "tool"` content array, where vLLM's `chat_utils.py:1549-1564` reduces tool content to `"\n".join(text_items)` — silently dropping every image.

### 5.9 Chat-template invariants the operator must respect

Six hard invariants from the shipped `chat_template.jinja` and Qwen3.6 model code. Violations either raise at render time (HTTP 500) or silently degrade quality.

1. **System messages cannot contain images.** `chat_template.jinja:9-10` raises `'System message cannot contain images.'`; Qwen3.6 was trained with text-only system prompts.
2. **System messages cannot contain video.** `chat_template.jinja:20-21` raises analogously.
3. **Tool messages render text **and** vision** when content is a list. `chat_template.jinja:131-142` calls `render_content(message.content, true)` on tool messages; the `render_content` macro at lines 3-41 dispatches on each list item and emits `<|vision_start|><|image_pad|><|vision_end|>` for image parts, `<|vision_start|><|video_pad|><|vision_end|>` for video parts, and `item.text` for text parts — all INSIDE the `<tool_response>...</tool_response>` block. Line 33 raises `'Unexpected item type in content.'` only on items that are neither text-typed nor image/video-keyed. Patch §7.9 ensures vLLM delivers the list to the template intact rather than flattening it to a string before the template runs.
4. **`add_vision_id` is opt-in convenience, not a training invariant — and currently structurally inert on this deployment** (`chat_template.jinja:15-17, 26-28`). The Qwen3.6-VL model card positions the `Picture N: ` / `Video N: ` prefix as opt-in for multi-image ordinal grounding ("By default, images and video content are directly included … When handling multiple images, it's helpful to add labels … Users can control this behavior" — Qwen3.6-VL README L690-691); the canonical default-invocation example at L725-728 omits it. Independently of training intent, on this deployment the flag is silently no-op regardless of value: vLLM's content-format auto-detector at `renderers/hf.py:230-244,269-271` walks the chat-template AST for top-level `for X in message['content']` loops, and our template performs that iteration inside the `render_content` macro (`chat_template.jinja:6-7`), so the detector misses it and falls back to `content_format='string'` (boot log: `Detected the chat template content format to be 'string'`). Under `'string'`, `chat_utils.py:1382-1421` flattens multi-image content into a single text prompt before the template runs, bypassing the `add_vision_id` branch. Activating the flag requires also setting `--chat-template-content-format openai`. We leave it off (see §11 B15 for the wire-test evidence and decision).
5. **Qwen3.6 does NOT support EVS** (efficient video sampling). `vllm/model_executor/models/qwen3_5.py:578` declares `supports_multimodal_pruning = False`; `--video-pruning-rate > 0` raises `NotImplementedError`. Don't set the flag.
6. **Multimodal cache content-hash dedup** is automatic. vLLM's `encoder_cache` is keyed by `mm_hash` derived from image bytes; agent loops re-sending byte-identical images get free encoder hits. Steady-state encoder cost is one forward per UNIQUE image, not per request.

---

## 6. vLLM issues — complete enumeration and per-issue disposition

Each issue below is classified as **A**. Runtime bug, **B**. Model OOD failure, **C**. Infrastructure bug, or **D**. Client-interop bug. Each §6 entry pairs with one §7 patch in the same numerical slot, except §6.6 which is sidestepped by API choice.

### 6.1 Ingest silently drops `reasoning_content` [Class A — vLLM internal inconsistency]

`vllm/entrypoints/chat_utils.py:1519` reads `message.get("reasoning")` and silently drops `reasoning_content`. The chat template at `chat_template.jinja:91-92` reads `message.reasoning_content` to render historical `<think>` blocks under `preserve_thinking=true`. **vLLM is feeding its own template a field its own ingest discards.** Without resolution, every multi-turn agent loop loses prior reasoning on replay; the model — RL-trained to expect prior-turn reasoning — re-derives context from scratch and tool-arg correctness degrades after 2-3 turns. Our two production clients (Qwen Code CLI, Qwen-Agent) both write `reasoning_content`.

**Affects us**: yes. **Resolution**: §7.1.

### 6.2 Issue #39771 — qwen3_coder crashes on truncated `<parameter=` tag [Class A]

`vllm/tool_parsers/qwen3coder_tool_parser.py:236` uses unsafe `str.index(">")`. When the model is truncated mid-`<parameter=NAME` (before `>`), `.index()` raises `ValueError`. The exception is caught at lines 320-324 and the parser returns `tools_called=False, tool_calls=[]` — collapsing **every well-formed sibling tool call in the same response**. Sibling code at line 227 already uses the safe `.find()/-1` pattern; line 236 is an internal inconsistency upstream PR #39772 acknowledges.

**Affects us**: yes, whenever a response is truncated by `max_tokens` or by client disconnect mid-generation. **Resolution**: §7.2; complemented by `max_tokens=81920` (§5.6) keeping truncation rare.

### 6.3 Issue #37121 — hybrid-KV log under-reports KV pool size and concurrency [Class D — observability bug]

vLLM's V1 paged KV cache manager forms one `KVCacheGroupSpec` per same-shape layer set. For Qwen3.6-27B that's 4 groups (1 full × 16 + 3 GDN × 16). The byte allocator at `vllm/v1/core/kv_cache_utils.py:1148-1169` allocates a single shared pool sized correctly. The bug is in two log-only functions: `_report_kv_cache_config:1305-1346` and `get_max_concurrency_for_kv_cache_config:802-820` both divide by `len(kv_cache_groups)` (4 for our model) instead of by the count of token-capacity-contributing groups (1 attention group; the 3 GDN groups hold O(1) state per request under `mamba_cache_mode != "all"`). The displayed `GPU KV cache size: X tokens` and `Maximum concurrency` are ~4× understated. Boot log unpatched says `~37K tokens`; with patch 3 installed → `~149K tokens`. Operators sizing `--max-model-len` against the displayed number under-utilize their hardware.

Call graph confirms the bug is observability-only for *this* deployment. Both functions have a single production caller each — `_report_kv_cache_config` is reached only from `unify_kv_cache_configs:1629`, and `get_max_concurrency_for_kv_cache_config` is reached only from inside `_report_kv_cache_config:1339` (no scheduler / KV-manager / admission caller anywhere in the tree). The actual admission gate is `KVCacheManager.can_fit_full_sequence` at `kv_cache_manager.py:218`, which uses `block_pool.get_num_free_blocks()` on the byte-correct shared pool — independent of the buggy reporting math. Empirically confirmed by B7: a 129,840-token request admits cleanly even when the unpatched log claims only ~37K available.

**Upstream status (2026-04-28)**: issue #37121 open since 2026-03-15. PR #40384 (our backport source) and competing PR #40694 both open. PR #37429 (broader byte-level redesign) blocked on RFC. PR #40384 also patches `Scheduler.__init__`'s `max_num_kv_tokens` divisor; that block is gated on `model_config.enable_return_routed_experts` (default `False`) and only matters for MoE+routed-experts deployments. Qwen3.6-27B is dense and we never set the flag, so the gated site doesn't apply here.

**Resolution**: §7.3. Observability-only for the deployment — byte allocation and admission are correct without it; the patch fixes the boot-log numbers operators rely on for capacity planning.

### 6.4 Egress emits non-standard `reasoning` field name [Class D]

vLLM emits the non-standard field name `reasoning` on the wire (since commit `c5113f60f2` deliberately removed `reasoning_content`). Qwen-Agent's OAI client at `Qwen-Agent/qwen_agent/llm/oai.py:111-112,126-127,169` strict-checks `reasoning_content` with **no fallback**; without the alias, every multi-turn agent loop loses prior reasoning on egress and degrades after 2-3 turns. Pydantic v2 compiled core schemas embed nested schemas by snapshot at build time, so a leaves-only rebuild still leaks `reasoning` through wrappers — every class on the dump chain must rebuild under `serialize_by_alias=True` for the leaf alias to reach the wire.

**Affects us**: yes. **Resolution**: §7.4 (egress). §7.1 closes the matching ingest half.

### 6.5 Issue #39056 — `<tool_call>` inside `<think>` is model OOD, not a parser bug [Class B]

Qwen3.6 occasionally emits `<tool_call>...</tool_call>` markup inside `<think>...</think>` (single-digit percent under agentic workloads). Upstream's `Qwen3ReasoningParser.extract_reasoning` correctly partitions on `</think>` first (`qwen3_reasoning_parser.py:142-144`) and routes mid-think markup to `reasoning`. **The parser is correct to its contract; the model is misbehaving.** Qwen3.6's chat template never renders historical tool_calls inside `<think>`, the Qwen3-Coder-Next training penalizes the pattern, and Alibaba's own evaluation in `Qwen-Agent/benchmark/deepplanning` strips everything up to `</think>` before parsing.

**Affects us**: yes, intermittently. **Resolution**: §7.5 (detect, don't rescue). The agent's retry policy decides what to do; the patch surfaces a structured WARNING so an operator can monitor the rate.

### 6.6 Issue #39584 — parallel tool calls crash Responses API [Class A — sidestepped]

`vllm/entrypoints/openai/responses/serving.py:1377` has a hardcoded `assert len(delta_message.tool_calls) == 1`. **Affects us**: no — we use Chat Completions.

### 6.7 Startup snapshot check semantically double-counts vLLM's own init footprint [Class C — infrastructure bug]

`vllm/v1/worker/utils.py:412-421`'s `request_memory()` raises `ValueError` whenever `init_snapshot.free_memory < total_memory * gpu_memory_utilization`. The intent — refuse to come up if external CUDA processes are starving the device — is correct, but the implementation is semantically wrong: `init_snapshot` is taken **inside the EngineCore worker after** PyTorch CUDA-context init, cuBLAS/cuDNN init, FlashAttention v2 workspace, and NIXL collectives have already allocated ~510 MiB on this stack. That ~510 MiB is part of vLLM's own footprint — accounted for downstream as `non_torch_memory + torch_peak_increase` and reserved out of `requested_memory` anyway — yet the check counts it as "external pressure" and refuses any `gpu_memory_utilization > ~0.984` on a single-tenant exclusive GPU. Operators with iGPU + dGPU layouts (the dGPU is exclusive) cannot land at gmu close to 1.0 even though no other process is using the card.

**Affects us**: yes — it caps the achievable KV pool at gmu=~0.98 minus the engine's init delta, leaving roughly 0.32 GiB of reachable KV unclaimable without patching. **Resolution**: §7.8.

### 6.8 Tool-role media stripped at chat_utils ingest [Class A — vLLM internal inconsistency]

vLLM strips media from tool-role list content via the flatten-to-string reducer at `vllm/entrypoints/chat_utils.py:1549-1564`'s `role == "tool"` branch — `result_msg["content"] = "\n".join(texts) if texts else ""` — preventing the chat template from rendering `<|vision_start|><|image_pad|><|vision_end|>` markers inside `<tool_response>`. Without this patch, even though the chat template has the rendering path (`render_content` at `chat_template.jinja:3-41` dispatches on list items and emits image markers natively when the tool branch at lines 131-142 calls into it), the content reaches the template as a flattened string with no media reference. Every non-text content part (`image_url`, `audio_url`, `video_url`, `file`, `input_image`, `image_pil`, `image_embeds`, `audio_embeds`, `input_audio`) is invisible to the template.

Compounding the bug: the per-content-part dispatcher upstream of the role-gate registered each media part with the `MultiModalItemTracker` *before* the reducer ran, so the encoder forwards data the rendered prompt never references — silent encoder/template desync, the same class A internal-inconsistency shape as §6.1 (the ingest path's "what counts as reasoning" disagreement with the chat template's view).

**Affects us**: yes — any client that returns tool media (MCP screenshots, `read_file`-emitted images, agent screenshots) is affected. The Qwen Code CLI's `splitToolMedia: true` is a client-side workaround that splits media into a follow-up user message — but the resulting wire shape diverges from the released Qwen3-VL-8B-Instruct training distribution (image markers OUTSIDE `<tool_response>`, empty `<tool_response></tool_response>` shells on media-only results, synthetic user message biasing the chat template's "last user query" reverse-walk). Closing the bug server-side by preserving list content lets the template render markers in the training-distribution shape.

**Resolution**: §7.9. Closes the silent strip server-side for ALL clients regardless of `splitToolMedia` setting; the rendered prompt matches Qwen3-VL training shape.

### 6.9 Validator throws after sender mm-cache populate poison the receiver assertion [Class A — vLLM internal inconsistency]

The chat-completions request path populates the renderer's `MultiModalProcessorSenderCache` (`vllm/multimodal/cache.py:379-434`) at `vllm/entrypoints/openai/chat_completion/serving.py:251` (`await self.render_chat_request(request)` runs `mm_processor.apply()`, which inserts the per-image mm_hash into the sender LRU). The same `create_chat_completion` then validates the length budget at `serving.py:284` (`get_max_tokens` raises `ValueError` from `vllm/entrypoints/utils.py:182` when input_length exceeds max_model_len). The renderer's own `apply_length_check` validator at `vllm/renderers/params.py:411-428` raises `VLLMValidationError` (`vllm/exceptions.py:9` — extends `ValueError`) on the symmetric over-budget condition. Either throw exits the request handler before the engine IPC fires, leaving the API-server's sender cache populated for hash H while the EngineCore's `MultiModalReceiverCache` (`cache.py:614-647`) has no entry for H. **Sender↔receiver mirror invariant broken.**

The next request carrying the same image hits the sender's IPC-saving short-circuit at `cache.py:415-416` (`return None, prompt_updates` — sender thinks engine has it). Engine's `preprocess_add_request` at `vllm/v1/engine/core.py:765-777` calls `mm_receiver_cache.get_and_update_features` (`cache.py:573-592`), which loops to `get_and_update_item` at `cache.py:636-647`. Hash H absent + payload data is None → `assert mm_item is not None, f"Expected a cached item for {mm_hash=}"` at line 644 fires. The `try/except Exception` at `core.py:1448-1452` catches it and routes through `_handle_request_preproc_error` (`core.py:1533-1540`): logs `"Unexpected error pre-processing request <id>"` and ships HTTP 500 (`"EngineCore encountered an issue. See stack trace for the root cause."`) back to the client. The poison persists for the lifetime of the sender LRU entry — thousands of requests under default `mm_processor_cache_gb`; potentially hours of agentic-workload churn for a hot image.

**User-visible failure**: a request goes over budget → HTTP 400 (correct). Operator clicks "Retry" with the same image and a smaller prompt → **HTTP 500**. Engine `/health=200` for most of the failure window; only after enough preprocessing-error accumulation does liveness flip. There is no API-level workaround.

Empirically reproduced in `/tmp/qwen36_research/mm_cache_bug_2026-04-28/repro_v6.py` (Subagent A, 2026-04-28). Sequence (per `repro_v6_results.json`): step 1 (`prompt 153,666 tokens > 152,000` + image) → HTTP 400 in 0.46 s; step 2 (clean small request, same image) → **HTTP 500 in 0.06 s** with the receiver-cache assertion in engine logs; step 3 (different image, large again) → HTTP 500 plus `/health=503` (engine degraded). Class A — vLLM is breaking its own sender↔receiver cache invariant on a path it owns end-to-end.

**Upstream status (2026-04-28)**: issue #31404 open against the receiver-cache assertion (different trigger, same end state). Draft PR #34749 attempts to soften the assertion to a typed error but is unmerged AND does not clear sender state on the rejecting path — even if merged, it would convert silent poisoning into LOUD errors, but the same retry would still fail because the sender↔receiver mirror is still broken. The fix has to clear sender state at the rejection site, not soften the engine-side assertion downstream.

**Affects us**: yes. Vision-bearing agentic workflows (MCP screenshots, screenshot-driven Computer Use) are the highest-exposure path because they reuse a small set of images across many turns. **Resolution**: §7.10.

---

## 7. The patches in this repo

Ten monkey-patches plus one container-entrypoint launcher plus one sitecustomize loader. Every patch is **server-side**, loaded into the vLLM Python process by `launch_with_patches.py` (in PID 1) and re-loaded by `sitecustomize.py` in spawned EngineCore subprocesses (load-bearing for patches 3 and 8; see §7.S). There is no client-side code in this repo.

Each patch addresses a specific defect named in §6 (or, for patches 6 and 7, a §5 design requirement) and only that defect. Each strictly validates its target's structure via landmarks, refuses to apply on any landmark mismatch with a typed exception that names the exact landmark that failed, stamps `__qwen36_patch__` on every target, and verifies via both `getattr` and `inspect.getattr_static` that its install took effect. The patch file itself is the source of truth; this section is a contract index.

| # | File | Defect addressed |
|---|---|---|
| 1 | [`monkey_patch_reasoning_field_ingest.py`](monkey_patch_reasoning_field_ingest.py) | §6.1 — accept `reasoning_content` on replayed assistant messages |
| 2 | [`monkey_patch_qwen3_coder.py`](monkey_patch_qwen3_coder.py) | §6.2 / #39771 — `_parse_xml_function_call` crash on truncated `<parameter=` |
| 3 | [`monkey_patch_hybrid_kv_allocator.py`](monkey_patch_hybrid_kv_allocator.py) | §6.3 / #37121 (PR #40384 backport) — boot-log under-reporting |
| 4 | [`monkey_patch_reasoning_field_egress.py`](monkey_patch_reasoning_field_egress.py) | §6.4 — rename `reasoning` → `reasoning_content` on response serialization |
| 5 | [`monkey_patch_tool_call_in_think_detector.py`](monkey_patch_tool_call_in_think_detector.py) | §6.5 / #39056 — detect `<tool_call>` emitted inside `<think>`, structured WARNING |
| 6 | [`monkey_patch_default_sampling_params.py`](monkey_patch_default_sampling_params.py) | §5.6 — server-side enforcement of Qwen3.6 sampling Best Practices for fields the client did not explicitly send |
| 7 | [`monkey_patch_qwen3_coder_grammar.py`](monkey_patch_qwen3_coder_grammar.py) | §5.6 / agentic-correctness — server-side xgrammar `structural_tag` constraint on tool emission; flips `supports_required_and_named=False` to close a latent JSON-list-path bug |
| 8 | [`monkey_patch_request_memory_snapshot.py`](monkey_patch_request_memory_snapshot.py) | §6.7 — make the startup snapshot check stop double-counting vLLM's own init footprint as "external pressure" |
| 9 | [`monkey_patch_tool_role_media_preserve.py`](monkey_patch_tool_role_media_preserve.py) | §6.8 — preserve list-shaped tool-role content with media so the chat template renders `<|vision_start|><|image_pad|><|vision_end|>` markers natively inside `<tool_response>`; closes silent image-strip for ALL clients regardless of qwen-code `splitToolMedia` setting |
| 10 | [`monkey_patch_mm_cache_validator_eviction.py`](monkey_patch_mm_cache_validator_eviction.py) | §6.9 / #31404 — clear renderer sender mm-cache when a length validator throws after the sender was populated; closes the validator-poisons-cache HTTP-500 retry pathology |
| L | [`launch_with_patches.py`](launch_with_patches.py) | Container entrypoint that imports the 10 patches in order, runs per-patch verification, then hands off to `vllm.entrypoints.cli.main` via `runpy.run_module(alter_sys=True)` |
| S | [`sitecustomize.py`](sitecustomize.py) | CPython auto-imports this from `PYTHONPATH=/opt/patches` at every interpreter startup — including the spawned EngineCore subprocess — so patches 3 and 8's targets are live in EngineCore's own `sys.modules` |

### 7.1 Patch 1 — `monkey_patch_reasoning_field_ingest.py`

Wraps `vllm.entrypoints.chat_utils._parse_chat_message_content`. When `role == "assistant"`, `reasoning is None`, and `reasoning_content` is a non-None string, synthesises a shallow copy with `reasoning` populated from `reasoning_content`. Both fields present with **different** values raises `ReasoningFieldAmbiguityError` (HTTP 400). Identical values pass through. Non-string `reasoning_content` (dict, list) refuses rather than being stringified. **Removal trigger**: vLLM widens ingest to accept `reasoning_content`.

### 7.2 Patch 2 — `monkey_patch_qwen3_coder.py`

Replaces `Qwen3CoderToolParser._parse_xml_function_call` to use `str.find(">")` and, on a malformed `<parameter=` tag, return `None` for the whole tool call rather than raising `ValueError` (which the upstream `try/except Exception` collapses into "drop all N tool calls in this response"). Sibling well-formed tool calls in the same response are preserved. The MRO walk for inherited `self.tools` / `self.tool_call_parameter_regex` attribute landmarks is load-bearing per the prior audit (the inherited `self.tools` from `ToolParser.__init__` would refuse incorrectly under a non-walking check). **Removal trigger**: PR #39772 merges.

### 7.3 Patch 3 — `monkey_patch_hybrid_kv_allocator.py`

Replaces `get_max_concurrency_for_kv_cache_config` and `_report_kv_cache_config` in `vllm.v1.core.kv_cache_utils`. Both functions divide by `len(kv_cache_groups)` (4 for our model: 1 attn + 3 GDN), making displayed token capacity and max-concurrency ~4× understated. Patched to filter `kv_cache_groups` to token-capacity-contributing specs only (`AttentionSpec` always; `MambaSpec` only when `cache_config.mamba_cache_mode == "all"`). Under §8.2 production flags, boot-log `GPU KV cache size` goes from `~37K tokens` to `~149K tokens`.

**Scope vs PR #40384**. PR #40384 in master modifies three sites — adds a `token_capacity_kv_cache_groups` helper, uses it in `_report_kv_cache_config`, uses it in `Scheduler.__init__`'s `max_num_kv_tokens` divisor. This patch backports `_report_kv_cache_config` and additionally fixes `get_max_concurrency_for_kv_cache_config` (same divisor pattern, second log line at boot, untouched by PR #40384). The scheduler.py site is deliberately not backported: it is gated on `model_config.enable_return_routed_experts` (default `False`) and only fires under MoE+routed-experts capture — Qwen3.6 is dense, we never enable the flag, and porting code under an off-by-default gate would only widen the patch's removal surface for no behavioural gain. Backport semantics, not literal port — the pinned commit's `MambaSpec.max_memory_usage_bytes` signature is `(self, vllm_config)`, not the master signature PR #40384 targets.

**Removal trigger**: PR #40384 or PR #40694 merges. **CRITICAL**: remove BEFORE pulling an image with PR #37429 (the broader byte-level redesign) — it changes the tensor layout and the patch's reporting view would no longer be coherent.

### 7.4 Patch 4 — `monkey_patch_reasoning_field_egress.py`

Installs Pydantic v2 `serialization_alias = "reasoning_content"` on the `reasoning` field of `ChatMessage` and `DeltaMessage`, flips `model_config["serialize_by_alias"] = True` on **all six** classes that vLLM dumps on the wire (the two leaves plus `ChatCompletionResponseChoice`, `ChatCompletionResponseStreamChoice`, `ChatCompletionResponse`, `ChatCompletionStreamResponse`), drops the cached `__pydantic_core_schema__` / `__pydantic_validator__` / `__pydantic_serializer__`, and calls `model_rebuild(force=True)`. A leaves-only patch is provably insufficient — Pydantic v2 compiled core schemas embed nested schemas by snapshot at build time. Phase 3 verification constructs real `ChatCompletionResponse` and `ChatCompletionStreamResponse` instances, dumps each via `model_dump_json()` and `model_dump_json(exclude_unset=True)`, and asserts wire bytes contain `"reasoning_content":` and not `"reasoning":`. The internal Python attribute stays `.reasoning`. **Removal trigger**: vLLM ships native `reasoning_content` on Chat Completions.

### 7.5 Patch 5 — `monkey_patch_tool_call_in_think_detector.py`

Wraps `Qwen3ReasoningParser.extract_reasoning` (non-streaming **only**) and emits a single structured WARNING (`model_emit_warning kind=tool_call_in_reasoning reasoning_len=N marker_count=M`) whenever the upstream-returned reasoning half contains literal `<tool_call>` markup. The wrapped return value is the upstream tuple **unchanged** — the agent's retry policy decides what to do. Detect, don't rescue: the parser is correct to its contract, the model is misbehaving, and a state-machine bandage across streaming deltas is high-complexity for a stochastic model-OOD failure mode. Streaming path is intentionally unwrapped — the rate is a model-side property, not per-modality, and a single non-streaming wrapper suffices. **Removal trigger**: Qwen3.6 retraining eliminates the OOD emission.

### 7.6 Patch 6 — `monkey_patch_default_sampling_params.py`

Wraps `ChatCompletionRequest.to_sampling_params` (`vllm/entrypoints/openai/chat_completion/protocol.py`). Runs the original, then for each Qwen3.6 Best-Practices default (`temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0, max_tokens=81920` — the official precise-coding/math thinking-mode values per §5.6) overrides the returned `SamplingParams.<field>` iff `field_name not in request.model_fields_set`. Pydantic v2's `model_fields_set` distinguishes explicit assignments from class defaults (load-bearing — verified at import time on a real `ChatCompletionRequest` with both an unset and a set-to-class-default probe). `max_tokens` is capped at `min(qwen_default, current)` — never raised above the serving layer's already-applied cap. Five behavioral cases run at import time and refuse the patch if any fail. **Removal trigger**: vLLM ships a first-class "model-recommended sampling defaults" mechanism (e.g. `--default-sampling-params` widened to respect `model_fields_set`) and the launch flags adopt it.

### 7.7 Patch 7 — `monkey_patch_qwen3_coder_grammar.py`

Overrides `Qwen3CoderToolParser.adjust_request` (`vllm/tool_parsers/qwen3coder_tool_parser.py`) and flips `Qwen3CoderToolParser.supports_required_and_named = False`. Fixes two distinct issues with one extension-point override:

1. **Tool-emission grammar.** With `--tool-call-parser qwen3_coder` and the default `tool_choice="auto"`, the inherited `ToolParser.adjust_request` (`vllm/tool_parsers/abstract_tool_parser.py:85-122`) is a no-op — `get_json_schema_from_tools` returns None for `"auto"`. The model emits XML tool calls **completely unconstrained**; a post-hoc parser then extracts what it can. The patched `adjust_request` builds an xgrammar `structural_tag` payload mapping `<tool_call>\n<function=NAME>\n` → per-tool body schema → `\n</function>\n</tool_call>` from `request.tools`, installs it on `request.structured_outputs`, and sets `request._grammar_from_tool_parser = True` (mirroring the `mistral_tool_parser.py:286` pattern). **What the constraint enforces**: (a) the function name on the wire is always one of the registered tools — xgrammar's literal `<function=NAME>` framing makes hallucinated names structurally unreachable; (b) the `<tool_call>...</tool_call>` and `<function=…>...</function>` markers cannot be elided or duplicated. **What the constraint does NOT enforce**: parameter-body shape. Qwen3.6 emits XML (`<parameter=KEY>VALUE</parameter>`) *inside* the framing, but `structural_tag.schema` is JSON-oriented at xgrammar `≥0.1.32, <1.0.0` and does not validate JSON Schemas against XML body text. We include `schema` in the payload regardless: it round-trips cleanly through `xgr.Grammar.from_structural_tag`, and a future xgrammar release that enforces XML bodies against JSON Schema activates it without changing this patch. Until that lands, the §7.5 detector and the post-hoc qwen3_coder parser remain responsible for argument-shape correctness, and clients with strict-shape requirements validate `tool_calls[i].function.arguments` themselves. xgrammar's bitmask FSM stays dormant during `<think>...</think>` (`enable_in_reasoning: bool = False` default at `vllm/config/structured_outputs.py:41`) and engages only after `</think>`, so thinking-mode reasoning remains unconstrained. Per-request overhead xgrammar paper-measured at <6% TPOT.
2. **Latent `tool_choice="required"` bug**. With the inherited `supports_required_and_named=True`, the standard JSON-list path at `engine/serving.py:646-665` runs `TypeAdapter(list[FunctionDefinition]).validate_json(content)` on the model's XML output. The validation fails, `contextlib.suppress(ValidationError)` swallows the error, and the response is silently empty. Setting the flag to `False` routes `"required"` and named tool_choice through `extract_tool_calls` instead — the same path GLM4 uses for the same XML-shape reason.

The override is gated: skipped if `request.structured_outputs` or `request.response_format` is already set (never tramples explicit client intent), if `tool_choice == "none"`, or if the request is a `ResponsesRequest` (which uses `request.text` rather than `request.structured_outputs`). Phase 9 behavioral verification at import time constructs a real `ChatCompletionRequest` with a synthetic single-tool list, calls the wrapped `adjust_request`, parses the resulting `structural_tag` JSON, asserts `structures` and `triggers` keys are present and shaped correctly, and round-trips the payload through `xgrammar.Grammar.from_structural_tag(tags, triggers)` without raising — load-bearing proof that what the patch generates is what xgrammar can compile. Three negative controls (empty tools, explicit client `structured_outputs`, `tool_choice="none"`) prove the gate fires.

Does NOT fix the `<tool_call>`-in-`<think>` model OOD case (§6.5 / patch 5). That emission happens BEFORE `</think>` where the FSM is dormant by design; the §7.5 detector remains correct and complementary.

**Removal triggers**: vLLM merges `Qwen3CoderToolParser.adjust_request` upstream OR ships a `qwen3_xml` parser with grammar enforcement (PR #25028 community recommendation).

### 7.8 Patch 8 — `monkey_patch_request_memory_snapshot.py`

Replaces `vllm.v1.worker.utils.request_memory` to fix the §6.7 semantic bug. The original compares `requested_memory = total × gmu` against `init_snapshot.free_memory`; we compare against `init_snapshot.free_memory + INIT_SLACK` instead, where `INIT_SLACK` is a generous bound on vLLM's own post-init CUDA-context footprint (1 GiB by default; tunable via `QWEN36_VLLM_INIT_SLACK_MIB` env var for hosts whose footprint differs). The replacement returns a byte-identical `int` to all downstream consumers (`gpu_worker.py:421-424`'s `available_kv_cache_memory_bytes` math is unchanged); only the gate threshold changes. The original safety intent — fail loud when EXTERNAL processes consume significant GPU memory — is preserved: the patch still raises if external pressure exceeds the slack. Three import-time behavioural probes prove (a) the buggy predicate is still present in the image (forces a re-audit if upstream landed a fix), (b) the patched function returns the expected `ceil(total × gmu)` on the failure-mode probe, (c) a negative-control probe with external pressure > slack still raises. Without this patch, `--gpu-memory-utilization 0.98` is unreachable on this stack — boot fails with `Free memory on device cuda:0 (30.85/31.36 GiB) on startup is less than desired GPU memory utilization (0.98, 30.73 GiB)` even on an exclusive dGPU. **Removal trigger**: vLLM upstream rewrites `request_memory` to either measure pre-init or accept a slack parameter; the buggy `init_snapshot.free_memory < requested_memory` predicate is the landmark — any rewrite changes its shape and forces this patch to refuse.

### 7.9 Patch 9 — `monkey_patch_tool_role_media_preserve.py`

Wraps the **inner per-message** function `_parse_chat_message_content` (`vllm/entrypoints/chat_utils.py:1510`) at the same address patch §7.1 wraps. The two patches guard on **disjoint role branches** — patch 1 on `role == "assistant"`, patch 9 on `role == "tool"` — so composition is order-independent: each wrapper handles only its own role and delegates to whatever it captured at install time. The `__wrapped_original__` chain walk in Phase 1 lets the landmark check operate on the original body regardless of how many patches have stacked at this address.

Algorithm: for each `role == "tool"` message, if content is a list with at least one media-typed part, snapshot a NEW list filtered to known TEXT (`text`, `input_text`, `output_text`, `refusal`, `thinking`) and MEDIA (`image`, `image_url`, `input_image`, `image_pil`, `image_embeds`, `video`, `video_url`, `video_embeds`) types BEFORE calling the original; let the original run unchanged (so `MultiModalItemTracker` registrations and `tool_call_id` linkage flow through the upstream path exactly as before — identical to the unpatched case); replace the resulting `ConversationMessage`'s flattened-string `content` field with the saved list. The chat template's `role == "tool"` branch at `chat_template.jinja:131-142` then calls `render_content(message.content, true)`; the `render_content` macro at lines 3-41 dispatches on each list item and emits `<|vision_start|><|image_pad|><|vision_end|>` for image parts INSIDE the `<tool_response>...</tool_response>` block — exactly the **training-distribution shape** of the released Qwen3-VL-8B-Instruct chat template (HF `Qwen/Qwen3-VL-8B-Instruct/chat_template.jinja`'s `role == "tool"` branch emits image markers in the same position).

Why preserve-list is correct vs split-into-follow-up-user (the prior design, deleted in this commit): the split design produced a wire shape that diverged from training in three ways — (1) image markers ended up OUTSIDE `<tool_response>`, breaking tool-result→image attribution; (2) media-only tool results produced empty `<tool_response></tool_response>` shells never seen in training; (3) the synthetic media-only user message became the chat-template's "last user query" via the reverse-walk at `chat_template.jinja:67-77`, biasing the model away from training-time semantics where the last query was textual. Plus a parallel-tool-calls contiguity bug (the split inserted a user turn between every tool message in a multi-tool round, while qwen-code PR #3617's canonical algorithm accumulates media into ONE follow-up user after the LAST tool message). The preserve-list redesign eliminates ALL of these: one wrap target instead of two, no message-list splicing, no synthetic dicts, no shim-text question.

Phase 4 carries five behavioural probes directly testing the pure filter function — A: text+image preserves order; B: image-only preserves single element; C: text-only returns None (no intervention; let the original's flatten run); D: string content returns None; E1: unknown+text (no media) returns None; E2: unknown+text+image filters unknown out and preserves [text, image]. Phase 6 verifies install via `getattr` + `inspect.getattr_static` agreement on the wrapped function.

Closes §6.8 server-side for ALL clients regardless of qwen-code's `splitToolMedia` setting; `splitToolMedia: true` on the client becomes redundant (still harmless — its output is a string, which hits the patch's no-op branch and passes through to the original's existing string-content path).

**Removal trigger**: vLLM upstream rewrites the `role:"tool"` content reducer at `vllm/entrypoints/chat_utils.py:1549-1564` to preserve list content. The buggy `"\n".join(texts) if texts else ""` predicate is the landmark — any reshape changes its body and forces this patch to refuse.

### 7.10 Patch 10 — `monkey_patch_mm_cache_validator_eviction.py`

Wraps `OpenAIServingChat.create_chat_completion` (`vllm/entrypoints/openai/chat_completion/serving.py:229-…`). On `ValueError` or `VLLMValidationError` (the typed exceptions the two length-budget validators emit at `vllm/entrypoints/utils.py:182` and `vllm/renderers/params.py:418`), calls `self.renderer.clear_mm_cache_async()` and re-raises the original exception unchanged. The eviction restores the sender↔receiver mm-cache mirror invariant that the validator throw broke (sender populated by `render_chat_request` at line 251, validator throws before the engine IPC fires at lines 284-303). Without the eviction, the next request carrying the same image hash hits the engine's `MultiModalReceiverCache` assertion at `vllm/multimodal/cache.py:644` and gets HTTP 500. **The poison persists for the lifetime of the sender LRU entry** — thousands of requests, easily several hours of agentic-workload churn for a hot image. Closing it server-side is the only API-level fix.

`self.renderer` is set in `OpenAIServing.__init__` (the base class, `vllm/entrypoints/openai/engine/serving.py:157`) as `engine_client.renderer` — the same `BaseRenderer` instance the API server's `OpenAIServingRender` wraps (`vllm/entrypoints/openai/api_server.py:369-371` constructs the latter with `renderer=engine_client.renderer`). The sender cache lives on that instance; clearing via `self.renderer` is equivalent to clearing via `self.openai_serving_render.renderer` and reaches the same object. Catching `(ValueError, VLLMValidationError)` is intentionally redundant — `VLLMValidationError` extends `ValueError` per `vllm/exceptions.py:9`, but naming both at the call site documents intent and forces a re-audit on upstream type-hierarchy refactors. We do **not** catch generic `Exception` — that would mask actual bugs that should bubble up. Defensive shape validation in the wrapper itself: if the renderer or its `clear_mm_cache_async` method has drifted (a future subclass that does its own thing), the wrapper logs a structured WARNING and re-raises the original exception — never less safe than upstream.

Phase 7 behavioural probes drive five harness cases via `asyncio.run` on a stub `OpenAIServingChat` whose `renderer.clear_mm_cache_async` is a counter-incrementing async method: (1) inner raises `ValueError` (mirrors `get_max_tokens` path) → counter == 1; (2) inner raises `VLLMValidationError` (mirrors `apply_length_check` path) → counter == 1; (3) inner raises `RuntimeError` (negative control — non-validator exceptions must NOT trigger eviction) → counter == 0; (4) inner returns cleanly (negative control — happy path must pass through with counter == 0); (5) defensive — `self.renderer` attribute missing → wrapper does NOT crash and original `ValueError` still propagates. The five cases run at import time and the patch refuses install on any failure.

**Scope vs upstream**: issue #31404 tracks a different trigger of the same end state (open). Draft PR #34749 attempts to soften the receiver-cache assertion to a typed error but is unmerged AND does not clear sender state on the rejecting path — even if it merged, it would convert silent poisoning into LOUD errors but the same retry would still fail because the sender↔receiver mirror is still broken; the next request still tries to short-circuit on a hash the engine doesn't know about. **Our patch is at the right layer** because the fix has to clear sender cache state at the rejection site, not soften the engine-side assertion downstream.

**Removal trigger**: vLLM upstream lands a fix that clears sender cache state on validator rejection (currently issue #31404 / draft PR #34749). The buggy behaviour is `serving.py:251` populating the sender cache before `serving.py:284`'s `get_max_tokens` validator can throw — any upstream restructuring that either (a) defers sender insertion until after all length validation succeeds or (b) wires a typed `finally`-block eviction into the rejecting path satisfies the removal trigger; the wrapper would then become redundant.

### 7.L Launcher — `launch_with_patches.py`

Container entrypoint, replacing `["vllm", "serve"]` with `["python", "/opt/patches/launch.py", "serve", ...]`. Imports every registered patch in `_PATCH_MODULES` order, runs the per-patch `_PATCH_VERIFICATION` verifier for each (re-imports the relevant vLLM target FROM SCRATCH and asserts the install took effect), then hands off to vLLM's CLI via `runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)`. Required because `PYTHONSTARTUP` does not fire under non-interactive entrypoints.

Verifiers split into two classes. Patches 2 and 3 carry **behavioural** verifiers — they instantiate the patched class with a synthetic input designed to expose the bug and assert the post-patch return value. Patches 1, 4, 5, 6, 7, 8, 9, 10 carry tag-only launcher verifiers (with `getattr` and `inspect.getattr_static` agreement); their patch-internal Phase verifications carry the load-bearing functional verification (the egress patch's wire-dump check; the ingest patch's static-lookup check; patch 6's five behavioural cases at Phase 7; patch 7's four behavioural cases at Phase 9 plus a load-bearing `xgr.Grammar.from_structural_tag` round-trip; patch 8's three behavioural probes — original-raises-on-buggy-input, patched-accepts-on-same-input, patched-still-raises-on-external-pressure; patch 9's five behavioural probes at Phase 4 directly testing the pure filter function — text+image preserves order, image-only preserves single element, text-only returns None (no intervention), string content returns None, and unknown-type filtering with both no-media-passthrough and media-present-filter sub-cases; patch 10's five harness cases at Phase 7 driving the wrapper via `asyncio.run` against a stub `OpenAIServingChat` — `ValueError` triggers eviction, `VLLMValidationError` triggers eviction, `RuntimeError` does NOT trigger eviction, happy-path passes through with no eviction, missing-renderer defensively skips eviction without crashing), so duplicating it here would only double the surface area. Patch 7's launcher verifier additionally asserts `Qwen3CoderToolParser.supports_required_and_named is False`.

Three pre-flight checks run BEFORE the per-patch import loop: sitecustomize-present (refuse if Debian's stub got loaded instead of ours), registry drift (refuse if `sitecustomize._PATCH_MODULES != launch_with_patches._PATCH_MODULES`), and a subprocess install probe (`subprocess.run([sys.executable, "-c", PROBE])` to confirm a freshly-spawned interpreter sees patched targets). All three are load-bearing for patch 3 — without them, the spawned EngineCore silently runs unpatched code while the launcher reports success. **Load order** matters: `reasoning_field_egress` (patch 4) must come before any patch that constructs `DeltaMessage` at request time, since the rebuild changes Pydantic's compiled schema.

### 7.S sitecustomize loader — `sitecustomize.py`

vLLM v1 spawns EngineCore as a `multiprocessing` child process via `spawn` (CUDA forbids `fork` after init). The spawned interpreter does not inherit `sys.modules`. Of the ten patches, **patches 3 and 8** target EngineCore-resident code (`monkey_patch_hybrid_kv_allocator` for the KV-cache reporting path, `monkey_patch_request_memory_snapshot` for the worker's startup `request_memory` check); patches 1, 2, 4, 5, 6, 7, 9, 10 target API-server-resident code and become live in PID 1 directly (chat_utils' `parse_chat_messages` and `OpenAIServingChat.create_chat_completion` both run in the request handler, never in EngineCore). Without `sitecustomize`, patches 3 and 8 are silently dead in EngineCore while the launcher's PID-1 verifier reports success. The pass/fail discriminator for patch 3 is the boot-log filename annotation: `[kv_cache_utils.py:NNN]` (unpatched, ~37K tokens) vs `[monkey_patch_hybrid_kv_allocator.py:NNN]` (patched, ~149K tokens). For patch 8 the discriminator is whether the boot ValueError fires at gmu > ~0.984 (unpatched, snapshot-check too strict) or the engine reaches "Application startup complete" (patched).

CPython's `site.py` auto-imports `sitecustomize` from `sys.path` at every interpreter startup, including spawned children. With `PYTHONPATH=/opt/patches`, `site.py` finds our file, which imports each patch in launcher order. Each patch's strict landmark check runs in EngineCore too; any refusal aborts startup loudly. The same flow runs in PID 1 — sitecustomize installs the patches, `launch.py` then hits cache via `importlib.import_module(...)`. Each patch's module-level code fires once.

**Load-bearing for**: patches 3 and 8. **Defense-in-depth for**: patches 1, 2, 4, 5, 6, 7, 9, 10. **Removal trigger**: when both patches 3 and 8 are removed; recommendation is to keep for defense-in-depth.

---

## 8. Deployment commands

End-to-end production sequence. Follow §8.1 → §8.5 in order from a fresh clone of this repo. The single canonical `docker run` block lives in §8.2 and is the only `docker run` in this README.

### 8.1 Pull the Docker image (one-time)

**There is no `Dockerfile` and no `docker build` step in this repo.** The deployment runs the upstream `vllm/vllm-openai` image **unmodified**, and the ten patches (plus `sitecustomize.py` and `launch_with_patches.py`) are bind-mounted into the container at `docker run` time via the `-v` flags in §8.2. The container's default `["vllm", "serve"]` entrypoint is replaced with `python3 /opt/patches/launch.py serve ...` so the launcher imports every patch — fail-loud on any landmark mismatch — *before* handing off to vLLM's CLI via `runpy`.

This is deliberate. Three properties fall out of it:

1. **The image stays digest-pinned to the audited upstream binary.** No local build, no rebuild on each patch change, no drift between what's on disk and what's in the image.
2. **The patches stay reviewable as text.** Each `monkey_patch_*.py` is a plain file in this repo; `git diff` shows exactly what's running. There is no opaque image layer carrying them.
3. **Removing a patch is one line.** Drop its `-v` mount and its entry in `_PATCH_MODULES`; the next launch comes up without it. (Per §12, every patch has an explicit removal trigger.)

Pull the digest now (one-time per host):

```bash
docker pull vllm/vllm-openai@sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba

docker inspect vllm/vllm-openai@sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba \
  --format '{{.Id}} {{.Architecture}}'
```

The published `vllm/vllm-openai` images are built by vLLM CI from each commit; we don't republish or modify them. If you ever need to rebuild from source — e.g. to bisect upstream — vLLM's own `docker/Dockerfile` is the canonical path; that's an upstream concern and out of scope here.

### 8.2 Launch vLLM

Run from the root of this repo (so `$PWD` resolves to the directory holding the patch files). The launcher refuses to come up if `sitecustomize.py` or any registered patch is missing from `/opt/patches/`, so a missed bind-mount fails loud at boot rather than silently at request time.

```bash
docker run -d --name qwen36 --gpus all \
  --restart unless-stopped \
  -p 127.0.0.1:8001:8001 \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --log-opt max-size=100m --log-opt max-file=5 \
  --health-cmd 'curl -fsS http://127.0.0.1:8001/health || exit 1' \
  --health-interval=30s --health-timeout=5s --health-retries=3 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/sitecustomize.py:/opt/patches/sitecustomize.py:ro" \
  -v "$PWD/monkey_patch_reasoning_field_ingest.py:/opt/patches/monkey_patch_reasoning_field_ingest.py:ro" \
  -v "$PWD/monkey_patch_qwen3_coder.py:/opt/patches/monkey_patch_qwen3_coder.py:ro" \
  -v "$PWD/monkey_patch_hybrid_kv_allocator.py:/opt/patches/monkey_patch_hybrid_kv_allocator.py:ro" \
  -v "$PWD/monkey_patch_reasoning_field_egress.py:/opt/patches/monkey_patch_reasoning_field_egress.py:ro" \
  -v "$PWD/monkey_patch_tool_call_in_think_detector.py:/opt/patches/monkey_patch_tool_call_in_think_detector.py:ro" \
  -v "$PWD/monkey_patch_default_sampling_params.py:/opt/patches/monkey_patch_default_sampling_params.py:ro" \
  -v "$PWD/monkey_patch_qwen3_coder_grammar.py:/opt/patches/monkey_patch_qwen3_coder_grammar.py:ro" \
  -v "$PWD/monkey_patch_request_memory_snapshot.py:/opt/patches/monkey_patch_request_memory_snapshot.py:ro" \
  -v "$PWD/monkey_patch_tool_role_media_preserve.py:/opt/patches/monkey_patch_tool_role_media_preserve.py:ro" \
  -v "$PWD/monkey_patch_mm_cache_validator_eviction.py:/opt/patches/monkey_patch_mm_cache_validator_eviction.py:ro" \
  -v "$PWD/launch_with_patches.py:/opt/patches/launch.py:ro" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e PYTHONPATH=/opt/patches \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint python3 \
  vllm/vllm-openai@sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba \
  /opt/patches/launch.py serve \
  --model QuantTrio/Qwen3.6-27B-AWQ \
  --revision 9b507bdc9afafb87b7898700cc2a591aa6639461 \
  --served-model-name Qwen3.6-27B-AWQ \
  --host 0.0.0.0 --port 8001 \
  --max-model-len 152000 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --mm-processor-kwargs '{"max_pixels": 4194304}' \
  --default-chat-template-kwargs '{"preserve_thinking": true, "enable_thinking": true}' \
  --limit-mm-per-prompt '{"image": 20, "video": 1, "audio": 0}' \
  -cc '{"cudagraph_capture_sizes":[1,2,4,8]}'
```

Each flag's rationale lives in §5. Load-bearing items: `--max-model-len 152000` and `--gpu-memory-utilization 0.97` (byte budget, §5.2 — patch §7.8 makes gmu>0.984 reachable; we land at 0.97 deliberately to leave the PyTorch caching allocator ~300 MiB of cold-path coalesce headroom, otherwise the first 4-MP image-bearing request after boot OOMs the LM-prefill MLP buffer despite total free memory being adequate; §5.2 / §11 B12); `--skip-mm-profiling` paired with `--mm-processor-kwargs '{"max_pixels": 4194304}'` (§5.8 — the cap bounds the runtime encoder peak, which is otherwise unmeasured and unsafe with `--skip-mm-profiling`); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (necessary but not sufficient on the cold path — permits segment growth but doesn't pre-grow on cold start; gmu=0.97 closes that gap by keeping ~300 MiB unreserved at boot); `--limit-mm-per-prompt` (default crashes boot, §5.8); `--max-num-batched-tokens 4096` (halves the LM-prefill MLP intermediate buffer from ~285 MiB at mnbt=8192 to ~142 MiB; combined with gmu=0.97 the KV pool boots at 158,368 tokens which admits 152K full-context requests at 1.04× concurrency; §5.2; B9 / B12 evidence in §11); `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` (tool calling); `--reasoning-parser qwen3` (server-side `<think>` extraction); `--default-chat-template-kwargs '{"preserve_thinking": true, "enable_thinking": true}'` (§5.7); `-cc '{"cudagraph_capture_sizes":[1,2,4,8]}'` (pins the auto-derivation that vLLM produces for `--max-num-seqs 4` at `vllm/config/vllm.py:1434-1586` — drift-immune across image bumps); `-p 127.0.0.1:8001:8001` + bridge networking (default; `--network host` removed) + `--host 0.0.0.0 --port 8001` inside the container (this is the vLLM CLI's bind, not the host bind). The container runs in its own network namespace, so vLLM's internal sockets — the OpenAI HTTP server, the EngineCore ZMQ IPC ports, NIXL, prometheus collector, distributed-init port, all of them — stay sealed inside the container and never reach a host interface. Only port 8001 crosses the boundary, and only via the explicit publish, and only on host loopback (`127.0.0.1:8001`). To re-expose publicly on another host, change the publish to `-p 8001:8001` (= 0.0.0.0:8001) and put an authenticating reverse proxy in front — vLLM has no built-in auth. **Do not use `--network host`**: that shares the host's network namespace, which makes EngineCore's ZMQ port (and any other internal RPC port vLLM opens) bind to the host's all-interfaces, which on a publicly-routable host = the public IP. `--host 127.0.0.1` only constrains the OpenAI HTTP server bind, not EngineCore's IPC bind; `sitecustomize.py` bind-mount (load-bearing for patches 3 and 8, see §7.S). Operational: `--restart unless-stopped` (auto-recover from engine crash); `--log-opt` rotation (bound docker log disk usage); `--health-cmd` (docker-native shallow liveness against `/health` — for deep wedge detection see §8.4). `--swap-space` is intentionally omitted (popped at LLM-API level and rejected by the CLI argparse in this image).

### 8.3 Wait for healthy, run warmup, run smoke test

These commands run in order. Don't proceed past a step that doesn't pass.

```bash
# Step 1: wait for docker State.Health.Status == healthy (60 s typical;
# patches and weight load take time). Refuse if not healthy in 5 min.
for i in $(seq 1 60); do
  status=$(docker inspect -f '{{.State.Health.Status}}' qwen36 2>/dev/null || true)
  [ "$status" = "healthy" ] && break
  sleep 5
done
[ "$status" = "healthy" ] || { echo "qwen36 not healthy: $status"; exit 1; }

# Step 2: confirm /v1/models reports the served name and 131,072 ctx.
curl -fs http://127.0.0.1:8001/v1/models | jq -e \
  '.data[0].id == "Qwen3.6-27B-AWQ" and .data[0].max_model_len == 152000'

# Step 3: confirm /metrics responds (Prometheus scrape target).
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/metrics  # expect 200

# Step 4: confirm all 10 distinct patch tags applied. Each patch loads in
# three processes (PID 1 launcher, pre-flight install probe, EngineCore
# spawn child) and emits one applied: line per loading, so the un-deduped
# line count is ~30; dedupe to count distinct tags.
docker logs qwen36 2>&1 | grep -oE 'qwen36-agent-setup-[a-z0-9-]+' | sort -u | wc -l   # expect 10
```

```bash
# Step 5: boot warmup — two concurrent moderate-prompt requests, slightly
# different bodies so prefix cache doesn't collapse them into one. Pushes
# decode-batch=2 cudagraph + allocator high-water-mark; ~1-3 s saving on
# first real agent turn. enable_thinking:false keeps each <8 decode tokens.
sys='You are an expert coding assistant. Use tools when needed. Think step-by-step before acting.'
for i in 1 2; do
  curl -fs --max-time 60 http://127.0.0.1:8001/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"Qwen3.6-27B-AWQ\",
         \"messages\":[{\"role\":\"system\",\"content\":\"$sys\"},
                       {\"role\":\"user\",\"content\":\"warmup turn $i: reply ok\"}],
         \"max_tokens\":8,
         \"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    >/dev/null &
done
wait
```

```bash
# Step 6: end-to-end smoke test (chat + thinking + tool schema). After
# §7.4 egress patch: choices[0].message.reasoning_content populated;
# choices[0].message.tool_calls[0].function.name == "calculator" with
# valid JSON arguments; finish_reason == "tool_calls". With patch 7
# installed, xgrammar's structural_tag pins the function name and
# parameter shape — hallucinated names cannot reach the wire.
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-AWQ",
    "messages": [{"role": "user", "content": "What is 127 * 349?"}],
    "tools": [{"type": "function", "function": {
      "name": "calculator", "description": "Evaluate a math expression.",
      "parameters": {"type": "object", "properties": {"expr": {"type": "string"}},
                     "required": ["expr"]}
    }}],
    "temperature": 0.6, "max_tokens": 4096
  }' | jq .
```

Operational: vLLM exports Prometheus metrics at `/metrics` — scrape for KV cache pressure, queue depth, request latency. Patch §7.5 emits `model_emit_warning kind=tool_call_in_reasoning` to docker logs when the model misbehaves; the host-side forwarder in §8.5 turns those into structured JSON Lines for alerting and ad-hoc queries.

### 8.4 Set up the wedge-recovery probe (deep liveness)

`--health-cmd 'curl -fsS http://127.0.0.1:8001/health'` from §8.2 detects the API-server process being dead, but **not** engine wedges where the HTTP layer answers `/health` with 200 while the EngineCore subprocess is stuck (CUDA hang, deadlocked allocator, livelocked scheduler — see vLLM forum thread *"V1 Engine child process dies unnoticed; check_health() is a no-op"*). [`health_probe.sh`](health_probe.sh) at this repo's root issues a real `/v1/chat/completions` decode and asserts `(choices[0].message.content // choices[0].message.reasoning_content) != null` AND `usage.completion_tokens >= 1`. The disjunction over content lanes is load-bearing on this stack: with `--reasoning-parser qwen3` enabled (§8.2), the first generated tokens route to `reasoning_content` until `</think>` is observed, so a 1-token probe lands its proof-of-decode in `reasoning_content` while `content` stays `null`. Either lane non-null is a real "engine decoded" signal.

```bash
# Step 1: install the script on the host (not inside the container —
# `docker restart` runs from the host).
sudo install -m 0755 health_probe.sh /usr/local/bin/qwen36_deep_probe.sh

# Step 2: confirm the probe exits 0 once against the running container.
/usr/local/bin/qwen36_deep_probe.sh && echo "deep probe OK"

# Step 3: install a recurring schedule. Cron line: 60 s cadence, retry
# once after 5 s, restart on second consecutive failure.
sudo tee /etc/cron.d/qwen36_deep_probe >/dev/null <<'CRON'
* * * * * root /usr/local/bin/qwen36_deep_probe.sh \
  || (sleep 5 && /usr/local/bin/qwen36_deep_probe.sh) \
  || docker restart qwen36
CRON
```

systemd-timer equivalent (substitute for the cron line if your host uses systemd timers): `OnUnitActiveSec=60s` unit running the same `||`-chained command.

### 8.5 Set up the §7.5 model_emit_warning forwarder

Patch §7.5 emits one structured WARNING per request whose reasoning contains `<tool_call>` markup, with `marker_count` reporting how many substrings were seen (rate is single-digit % under agentic workloads). The line is machine-parseable by design: `model_emit_warning kind=tool_call_in_reasoning reasoning_len=<int> marker_count=<int>`. By default it lands in `docker logs` and stays there. [`host_logs/qwen36_warning_forwarder.py`](host_logs/qwen36_warning_forwarder.py) tails `docker logs -f`, parses each line with a strict regex anchored on the patch's exact format string (line 84 of the patch source), and appends one JSON Lines record per event to `/var/log/qwen36/warnings.jsonl`. A line that matches the marker prefix but fails the full regex is treated as a LOUD parse error: it goes to stderr/journald and increments `parse_errors` in the state file — silent drops are precisely what makes monitoring useless.

JSONL schema, one object per line:

```
{"ts": "<ISO-8601 UTC>", "kind": "tool_call_in_reasoning", "reasoning_len": <int>, "marker_count": <int>, "raw": "<original docker line>"}
```

Install (idempotent — rerun safe):

```bash
sudo bash host_logs/install.sh
```

That copies the script to `/usr/local/bin/`, the unit to `/etc/systemd/system/`, creates `/var/log/qwen36/` and `/var/lib/qwen36/`, runs `systemctl daemon-reload`, and enables/starts the service. Verify:

```bash
systemctl status qwen36-warning-forwarder.service
journalctl -u qwen36-warning-forwarder -f
tail -f /var/log/qwen36/warnings.jsonl
cat /var/lib/qwen36/forwarder_state.json   # events_total, restart_count, parse_errors
```

Sample query — tool-call-in-reasoning events in the last hour:

```bash
HOUR_AGO="$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)"
jq -r --arg t "$HOUR_AGO" 'select(.ts > $t)' /var/log/qwen36/warnings.jsonl | wc -l
```

Without systemd: run `python3 host_logs/qwen36_warning_forwarder.py --output … --state …` directly, or fold it into your own supervisor. `--no-stdout` suppresses the JSON line on stdout when only file output is wanted. `kill -HUP $(pgrep -f qwen36_warning_forwarder)` rotates the output file in place; `kill -TERM` graceful-shuts with a `shutdown: events=N restarts=M parse_errors=K` banner to stderr.

### 8.6 Production-ready checklist

Sign off all of these before declaring the deployment ready:

- [ ] `docker inspect -f '{{.State.Health.Status}}' qwen36` reports `healthy`.
- [ ] `docker logs qwen36 2>&1 | grep -oE 'qwen36-agent-setup-[a-z0-9-]+' | sort -u | wc -l` returns **10** (the ten distinct patch tags). Patches load in three processes — PID 1 launcher, the launcher's pre-flight subprocess install probe, and the spawned EngineCore — so the un-deduped `applied:` line count is **~30**.
- [ ] `curl -fs http://127.0.0.1:8001/v1/models | jq '.data[0].max_model_len'` returns **152000**.
- [ ] `curl -fs -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/metrics` returns **200**.
- [ ] `/usr/local/bin/qwen36_deep_probe.sh` returns **0**.
- [ ] `systemctl is-active qwen36-warning-forwarder.service` returns **active** and `cat /var/lib/qwen36/forwarder_state.json | jq .parse_errors` returns **0** (a non-zero parse_errors means the §7.5 format string drifted upstream — re-audit before declaring ready).
- [ ] §8.3 step 6 smoke test returns `finish_reason == "tool_calls"` with `tool_calls[0].function.name == "calculator"` (patch §7.7 pins the function name to a registered tool — name hallucination cannot reach the wire). Argument-body schema is **not** xgrammar-enforced (see §7.7 "What this does NOT close"); validate `tool_calls[0].function.arguments` against your schema client-side if the agent needs strict shape.

---

## 9. Known unknowns

| Item | Evidence status |
|---|---|
| `<tool_call>`-in-`<think>` rate, thinking-mode loop rate under our AWQ config | `<tool_call>`-in-`<think>` community-reported single-digit-percent; patch §7.5 surfaces it as structured WARNING (`model_emit_warning kind=tool_call_in_reasoning`). **Loop rate** measured 17.4% on Qwen3.5-35B-A3B unquantized; not re-measured under AWQ + BF16-linear-attn + BF16-KV. |
| Image count boot ceiling above `image:20` with `--skip-mm-profiling` | Verified to boot at `image:20` (KV reporting byte-identical to `image:2`); higher values with `--skip-mm-profiling` likely also boot since the crash mode is post-profiling and we skip profiling, but not empirically tested past 20. Without `--skip-mm-profiling`, `image:N ≥ 10` crashes boot via TRT-LLM `throwRuntimeError`. |

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

210-prompt corpus served 2026-04-25 (Hebrew + ancient-Egyptian + emoji, 0% true garbling). Production §8.2 flags exercised against container `qwen36` on `127.0.0.1:8001` over 2026-04-27 (B4/B6/B7 close, §8.2 baseline) and 2026-04-28 (high-context expansion to 131K). Artifacts under `/tmp/qwen36_research/validation_2026-04-27/` and `/tmp/qwen36_research/max_context_tests_2026-04-28/`.

| ID | Path | Status | Coverage |
|---|---|---|---|
| **B1** | Tool-bearing requests | **Partial** | 5-turn agentic round-trip of `tool_calls[]`; parallel tools, deeply nested params, `anyOf` not yet a dedicated corpus. |
| **B2** | `<tool_call>`-in-`<think>` | **Detector-validated** | §7.5 emits structured WARNING; retry policy belongs to the agent. |
| **B3** | Multi-turn `reasoning_content` round-trip | **Validated** | PRESERVE/STRIPPED/CORRUPTED arms distinguishable; 5-turn round-trip. |
| **B4** | Vision input | **Validated** | 2 real images correctly described (`disc5_imgs/img1.png`, `img2.png`); also re-validated under `--skip-mm-profiling` (§5.8) — runtime vision unaffected. Negative control without image hallucinates unrelated content. |
| **B5** | Streaming correctness | **Validated** | `reasoning_content` round-trips across deltas. |
| **B6** | Concurrency stress | **Validated** | 4 concurrent at 30K input tokens each fully admitted (peak Running=3, peak KV usage 76.6%, no OOM); 4×10K thinking-on shows 1.50× speedup over serial (peak Running=4). |
| **B7** | Long-context retrieval | **Validated (single-needle to 99%)** | At 131K `--max-model-len`: needle recalled at 125,661 tokens depth 10%, 125,662 depth 90%, and **129,840 (99.06% of max) at depth 50%** — every recall correct. Earlier 32K @ depth 10/50/90% and 60K @ depth 50% also hold. RULER multi-needle / multi-distractor not exercised. |
| **B8** | Multi-image + long-context stress | **Validated at the boundary** | At §8.2 production flags: 2× 4-MP images + 115,190-token document = **123,421 prompt tokens (94.2% of `--max-model-len`)** admitted, both images correctly described, needle `AC582B0` retrieved at depth ~50%, GPU memory peak +318 MiB transient (returned to baseline post-request), 8 s warm-cache / 51 s cold. At **133,889 prompt tokens** (over by 2,817): clean HTTP 400 `"Input length (133889) exceeds model's maximum context length (131072)."` at ingest, engine state intact, deep liveness probe still passes. `image:N` ceiling enforced at ingest with HTTP 400 `"At most N image(s) may be provided in one prompt."` and is byte-identical between N=2 and N=20 in KV reporting (`image:N` is a per-request integer cap, not a memory reservation; §5.8). |
| **B9** | OOM / admission boundary sweep (2026-04-28, prior config mnbt=8192/max-len=131K) | **Validated boundaries; ONE real OOM found and fixed** | (a) **20× 4-MP synthetic images + 36K text = 118,261 prompt tokens, no client `max_tokens`**: HTTP 200 in 320 s, image correctly described, VRAM flat. (b) **20× 4-MP + 66K text = 146,651 prompt tokens (over) + `max_tokens=200`**: HTTP 400 in 0.221 s from `entrypoints/utils.py:174-185`, engine `/health=200` after. (c) **Long-decode under thinking** (4,848-token prompt + `max_tokens=126,128`, ran 287.8 s, 20,465 generated, finish=stop): VRAM held flat at 32,054/56 for 4m48s. (d) **Exact admission boundary**: last-accepted `max_tokens=31,088` for a 99,984-token prompt. (e) **`prompt + max_tokens > max_model_len`** rejected via `renderers/params.py:411-425`. (f) **CRITICAL: 20× 4-MP REAL Webb Pillars-of-Creation crops + 47K text = 129K prompt** at mnbt=8192 → `torch.OutOfMemoryError: Tried to allocate 262.00 MiB. Of which 257.88 MiB is free` — LM-prefill MLP buffer `(s≈7884, intermediate=17408)` fp16 tipped over by 4 MiB. Engine crashed cleanly via `EngineDeadError`, container auto-restarted via `--restart unless-stopped`. **Root cause**: at mnbt=8192 the prefill MLP buffer is ~285 MiB; combined with weight pages, KV-block fills, and encoder cache for 20-image expansions, the runtime activation peak exceeded the ~280 MiB workspace headroom. **Fix**: drop mnbt to 4096 (next row). Artifacts: `/tmp/qwen36_research/oom_admission_tests_2026-04-28/`, `/tmp/qwen36_research/oom_image_aggressive_2026-04-28/`. |
| **B10** | RULER-style multi-needle at the new 152K ceiling (2026-04-28) | **STRONG_PASS at the new boundary** | At §8.2 production flags (mnbt=4096, max-model-len=152000): (a) **5 needles + 50 distractors at 135,986 prompt tokens (warm)**: 5/5 correct in correct ascending depth order in 19.9 s. (b) **7 needles + 100 distractors + 6% fake-`[REF-CODE]`-marker traps + asymmetric depths (12/24/38/51/65/79/92) + reverse-depth output, 133,867 prompt tokens (cold cache after `docker restart`)**: 7/7 correct in correct reverse order in 87.3 s end-to-end. (c) **Same hardness pushed to 148,337 prompt tokens (97.6% of max-model-len, cold)**: all 7 codes correctly enumerated in reasoning content (model's thought stream printed all seven correct codes); decode truncated by max_tokens=2000 mid-output but grading on reasoning-content scores **STRONG_PASS**. Cold-prefill throughput at mnbt=4096 measured at ~2,400–2,480 tokens/s; ~10–15% slower than mnbt=8192 (which would have OOMed anyway). Engine `healthy`, `/health=200` after every test. Artifacts at `/tmp/qwen36_research/ruler_150k_2026-04-28/`. |
| **B11** | Validator-poisons-mm-cache: bug repro + patch §7.10 live re-validation (2026-04-28 → 2026-04-29) | **Bug confirmed against unpatched; patch §7.10 verified live end-to-end** | **Pre-patch repro at `/tmp/qwen36_research/mm_cache_bug_2026-04-28/repro_v6.py`** (Subagent A): Step 1 (`prompt 153,666 > 152,000` + image) → HTTP 400 in 0.46 s (correct validator rejection from `entrypoints/utils.py:182`). Step 2 (clean small + same image) → **HTTP 500 in 0.06 s** with `AssertionError: Expected a cached item for mm_hash='0f21b…'` from `vllm/multimodal/cache.py:644`; engine `/health=200`. Step 3 (different image, large again) → HTTP 500; engine `/health=503` (now degraded). **Post-patch re-run on 2026-04-29** against the patched container: Step 1 still 400 in 0.27 s (validator unchanged). Step 2 now **HTTP 200 in 1.32 s** with prompt_tokens=4111 — the fix took effect. Step 3 200 in 1.59 s (sanity). Engine `/health=200` after every step; **zero `AssertionError: Expected a cached item` in the engine log** during/after step 2's processing; the patch wrapper present in the rejection traceback at `monkey_patch_mm_cache_validator_eviction.py:447`. Triple-live artefacts at `/tmp/qwen36_research/triple_live_2026-04-28/`. |
| **B12** | Cold-path PyTorch allocator fragmentation OOM (2026-04-28) | **Bug pin-pointed; gmu=0.98→0.97 fix verified end-to-end** | At gmu=0.98 with mnbt=4096 / max-model-len=152000, the **first** request after engine boot carrying a 4-MP image deterministically OOMs at the LM-prefill MLP intermediate buffer (`(s≈3920, intermediate=17408)` fp16 = 132 MiB needed). At OOM time: 140 MiB free + 127 MiB reserved-but-unallocated, total ~267 MiB free — but PyTorch's caching allocator can't coalesce a contiguous 132 MiB block from the cold-state segments. `s18=3920 ≤ mnbt=4096` rejects the unbounded-chunk hypothesis; warm-path same-shape passes consistently rejects the activation-profile-miss hypothesis; the smoking gun is the reserved-but-unallocated split. **Hypothesis (c) confirmed**: cold-path coalesce failure. `expandable_segments:True` permits segment growth but doesn't pre-grow at boot. **Fix**: drop gmu by one percent point (0.98 → 0.97), which keeps ~300 MiB unreserved at boot. KV pool drops from 161,504 → 158,368 tokens (−2.9%), 152K-concurrency drops from 1.06× → 1.04×. **Validation at gmu=0.97**: cold 4-MP request HTTP 200 in 26.3 s with 4,115 vision tokens; same-image cached 0.37 s; different-image 1.81 s; 4× distinct 4-MP images admit cleanly; full-pipeline RULER 7/7 cold at prompt_tokens=148,337 in 74.6 s; restart_count=0 throughout. Sub-agent C v2 artifacts at `/tmp/qwen36_research/oom_pin_2026-04-28/`. |
| **B13** | Thinking-mode toggle ground truth (2026-04-29) | **Toggle is empirically strict at production T=0.6 when wired correctly** | 95-trial sweep at T=0.6 with **top-level** `chat_template_kwargs.enable_thinking=false`: **0 leaks** across (a) 30 trials matrix of easy/medium/hard difficulties × T=0/T=0.6, (b) 30 trials prompt-length sweep at 5K/30K/60K/100K/130K/150K targets (132,618 actual tokens at the 150K cell), (c) 20 trials math/code/retrieval intents including a 148K-token RULER single-needle. Cross-engine bug citations re-examined: vllm #30477 was a user-error (wrong `extra_body` wire shape), #40816 is a streaming-only bug on a different model size, sglang/llama.cpp issues are engine-specific. **Critical wire-shape finding**: vLLM's request schema has `extra="allow"`, so `extra_body={"chat_template_kwargs": {...}}` in raw HTTP is silently dropped — the OpenAI Python SDK unwraps `extra_body` client-side, but raw `curl` does not. Earlier in-session payloads using the `extra_body`-nested shape were silent no-ops; the apparent "leakage" they showed was the model thinking under the un-overridden server default and the parser routing `max_tokens`-truncated thinking (no `</think>`) to `reasoning_content` per its documented contract at `vllm/reasoning/qwen3_reasoning_parser.py:75-93`. Sub-agent E artifacts at `/tmp/qwen36_research/thinking_mode_ground_truth_2026-04-29/`. |
| **B14** | Patch 9 (tool-role media preserve) live empirical confirmation (2026-04-29) | **Validated end-to-end** | Sent a 4-turn conversation `system → user → assistant(tool_calls) → tool` with the `tool` message's content as a 2-element list `[{type:text}, {type:image_url}]` carrying `disc5_imgs/img1.png` (793×224, ~155 vision tokens). HTTP 200; model response **"Query=Hi!, Reasoning panel, AI assistant interface"** (accurately describes the screenshot). **Smoking gun: 164-token prompt-token delta** between list-with-image (258 prompt tokens) and a plain-string negative control (94 prompt tokens). The 164-token delta matches the expected ~155 vision tokens for a 793×224 image plus framing — direct, unambiguous evidence that the image data flowed through the multimodal renderer despite being inside a `role:"tool"` message. Plain-string control hallucinates an empty UI, no vision tokens. Engine `/health=200` throughout. Triple-live artefacts at `/tmp/qwen36_research/triple_live_2026-04-28/`. |
| **B15** | `add_vision_id` decision + content-format autodetect interaction (2026-04-29) | **Not training-load-bearing AND inert on this deployment until `content_format='openai'` is set** | (i) Training-side: Qwen3.6-VL model card L690-691 frames `add_vision_id` as opt-in convenience ("helpful to add labels … users can control this behavior"); canonical default example L725-728 omits it; the technical report does not mention the flag — flipping the server default is not training-fidelity-required. (ii) Live wire test: per-request `chat_template_kwargs.add_vision_id=true` (top-level, the B13 shape) produces token sequences **byte-identical** to the server default for 1/2/3-image requests via both `/tokenize` and `/v1/chat/completions`; full-prompt decode shows no `Picture N: ` prefix in either shape. n=1 multi-image ordinal probe (`THIRD image only` → `BLUE TEST`): both shapes correctly identify the third image; identical prompt_tokens (488). The override is silently dropped. (iii) Root cause: vLLM `_detect_content_format` (`renderers/hf.py:230-244`) walks the chat-template AST for top-level `for X in message['content']` loops; our `chat_template.jinja:6-7` performs the iteration inside the `render_content` macro and the detector misses it, returning `'string'` (`hf.py:271`; boot log `[hf.py:314] Detected the chat template content format to be 'string'`). Under `'string'`, `chat_utils.py:1382-1421` flattens multi-image content to a single string before the template runs, bypassing the `add_vision_id` branch (`chat_template.jinja:6-18`). To activate `add_vision_id`, add `--chat-template-content-format openai` to §8.2 (also makes `chat_template_kwargs.add_vision_id` per-request toggle work). **Decision: leave `add_vision_id` off** — opt-in by training, n=1 ordinal probe shows the model resolves multi-image grounding correctly without it, and the cost (~5 extra text tokens per image, total token budget) is unjustified for our workload mix. Sub-agent G artifacts at `/tmp/qwen36_research/add_vision_id_2026-04-29/`. |
| **B16** | 130K–152K needle gap closure + §8.4 wedge-recovery cron operational status (2026-04-29) | **9/9 PASS recall verified; cron + probe NOT installed on this host** | (i) Needle: 9-cell matrix at temperature 0, top-level `chat_template_kwargs.enable_thinking=false`, `max_tokens=60`, fresh per-trial random suffixes, deterministic `[entry NNNNNN] <topic>` filler over a 12-topic Qwen-stack pool. Three lengths × three depths: `L130K_D{10,50,90}` (actual prompt 129 903–130 136), `L145K_D{10,50,90}` (144 925–145 071), `L152K_D{10,50,90}` (151 044–151 074). All 9 cells return verbatim `The secret token is XYZ123-<suffix>` with `finish_reason=stop`. Latencies 50–62 s per cell (130K cold ~50 s, 145K ~59 s, 151K ~62 s); engine `/health=200` throughout. The originally-`152K`-targeted cells were re-targeted to 151 000 user-text tokens because the original target=152 000 calibrated to user-text 152 023–152 091 and the chat-template framing tipped framed-prompt + `max_tokens=60` over the 152 000 cap (HTTP 400 with validator-reported `at least 151 941 input tokens`). The 151K cells exercise 99.4% of context length and complement §11 B12's single-needle 148 337-token cold-prefill probe; the strict 152 000 cap is a function of `prompt + max_tokens ≤ max_model_len` per `renderers/params.py:411-425` (B9-(e)). (ii) §8.4 wedge-recovery cron / probe script: `crontab -l` returns "no crontab for user"; `/etc/cron.d/qwen36_deep_probe` absent; no `qwen36*` `systemctl list-timers`; `/usr/local/bin/qwen36_deep_probe.sh` does not exist. Source `health_probe.sh` is present and executable in the repo. README §8.4 install steps were never executed on this host — the §8.6 production-readiness checklist row at line 629 ("`/usr/local/bin/qwen36_deep_probe.sh` returns 0") therefore currently fails for an operational reason, not a code reason. Sub-agent H artifacts at `/tmp/qwen36_research/needle_gap_2026-04-29/`. |
| **B17** | `preserve_thinking=true` provenance + §5.7 wording audit (2026-04-29) | **Default kept; wording tightened to remove unsupported claims** | The prior §5.7 wording asserted Qwen3.6 was "RL-trained expecting reasoning to persist across turns" and that `preserve_thinking=true` is "load-bearing for multi-turn agentic correctness". Source audit (Sub-agent J): the Qwen3.6-27B HF model card actually says "**additionally trained** to preserve and leverage thinking traces from historical messages" and "particularly beneficial for agent scenarios" (Qwen3.6-27B-README.md L811, L839; identical wording on Qwen3.6-35B-A3B). The Qwen-Agent reference deployment in the same model card (L869) sets `preserve_thinking=True` for DashScope and vLLM/SGLang. Predecessor Qwen3 (Qwen3-32B model card "Best Practices" → "No Thinking Content in History") explicitly recommended STRIPPING `<think>` from history — Qwen3.6 is a deliberate inversion. **Critical caveat**: there is no published controlled ablation of `true` vs `false`; the only public empirical claim is `badlogic/pi-mono#3325` (single-user anecdote, Qwen3.6-35B-A3B + LM Studio, empty-arg tool calls after 2–3 turns without the flag), and two follow-ups (`badlogic/pi-mono#3479`, `jundot/omlx#900`) report `preserve_thinking=true` necessary but not always sufficient. vLLM is mechanism-only — the `--default-chat-template-kwargs` plumbing was introduced by PR #31343 (commit `dc837bc23`, 2025-12-30) as a generic feature predating Qwen3.6, so the recommendation is Qwen-canonical, not vLLM-canonical. Decision: keep the server default `preserve_thinking=true` (Qwen-recommended), tighten §5.7 to cite the model-card text verbatim and label the failure-mode evidence as a single anecdote. Sub-agent J artifacts at `/tmp/qwen36_research/preserve_thinking_research_2026-04-29/` (eight verbatim-excerpt evidence files). |

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
9. **vLLM ships first-class server-side model-recommended sampling defaults** (e.g., `--default-sampling-params` widened to respect `model_fields_set`) — remove patch 6.
10. **vLLM merges `Qwen3CoderToolParser.adjust_request` upstream OR ships `qwen3_xml` with grammar enforcement** — remove patch 7.
11. **vLLM rewrites `request_memory()` to either measure pre-init or accept a slack parameter** (the buggy `init_snapshot.free_memory < requested_memory` predicate is the landmark) — remove patch 8.
12. **vLLM ships workload-aware CUDA-graph capture-size defaults** — drop the `-cc '{"cudagraph_capture_sizes":[...]}'` pin from §8.2.
13. **vLLM issue #38182 (MTP + prefix cache) closes** — reconsider enabling MTP.
14. **vLLM upstream rewrites the `role:"tool"` content reducer at `vllm/entrypoints/chat_utils.py:1549-1564`** to preserve media (the buggy `"\n".join(texts) if texts else ""` predicate is the landmark) — remove patch 9.
15. **vLLM upstream lands a fix that clears sender cache state on validator rejection** — currently issue #31404 / draft PR #34749. Either (a) defers the sender-cache populate at `serving.py:251` until after all length validation succeeds, or (b) wires a typed `finally`-block eviction into the rejecting path. Either satisfies the removal trigger — remove patch 10.

Each is tracked; none urgent.

---

## 13. File structure of this project

```
.
├── README.md                                              # this document
├── launch_with_patches.py                                 # §7.L — container entrypoint; imports the 10 patches then runpys vLLM
├── sitecustomize.py                                       # §7.S — auto-loads patches in EngineCore (and PID 1) at interpreter startup
├── health_probe.sh                                        # §8.4 — host-side deep liveness probe; engine-decoded-a-token check
├── monkey_patch_reasoning_field_ingest.py                 # §7.1 — accept reasoning_content on inbound assistant messages
├── monkey_patch_qwen3_coder.py                            # §7.2 — parser crash fix on truncated <parameter=
├── monkey_patch_hybrid_kv_allocator.py                    # §7.3 — hybrid-KV boot-log under-reporting fix (PR #40384 backport)
├── monkey_patch_reasoning_field_egress.py                 # §7.4 — Pydantic serialization rename reasoning → reasoning_content
├── monkey_patch_tool_call_in_think_detector.py            # §7.5 — detect <tool_call> emitted inside <think>; structured WARNING
├── monkey_patch_default_sampling_params.py                # §7.6 — server-side Qwen3.6 sampling defaults for unset fields
├── monkey_patch_qwen3_coder_grammar.py                    # §7.7 — xgrammar structural_tag on tool emission; supports_required_and_named=False
├── monkey_patch_request_memory_snapshot.py                # §7.8 — startup snapshot check stops double-counting vLLM's own init footprint
├── monkey_patch_tool_role_media_preserve.py               # §7.9 — preserve list-shaped tool-role content with media so the chat template renders <|vision_start|><|image_pad|><|vision_end|> inside <tool_response>
├── monkey_patch_mm_cache_validator_eviction.py            # §7.10 — clear renderer sender mm-cache on validator throw; closes #31404 retry-poison HTTP-500
├── host_logs/
│   ├── qwen36_warning_forwarder.py                        # §8.5 — host-side; tails docker logs, parses §7.5 warnings into JSONL
│   ├── qwen36-warning-forwarder.service                   # §8.5 — systemd unit (Type=simple, After=docker.service)
│   └── install.sh                                         # §8.5 — idempotent installer for script + unit + dirs
└── tests/
    ├── test_patches_against_master.py                     # static + structural-mirror suite (runs without torch/CUDA)
    └── test_warning_forwarder.py                          # §8.5 forwarder unit tests (no docker; pure Python)
```

Twelve Python files (10 patches + launcher + sitecustomize) plus the host-side probe shell script, the host-side warning forwarder bundle (`host_logs/`), and two test files.
