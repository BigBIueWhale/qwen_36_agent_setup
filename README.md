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
| `--max-model-len` | **131,072** |
| Disk size | 20.36 GiB |
| VRAM at boot (measured) | 19.78 GiB resident (MTP head auto-skipped via `qwen3_5.py:701-706`'s `skip_prefixes=["mtp."]`) |
| KV pool available | 9.13 GiB at gmu=0.98 with `--skip-mm-profiling` and `--mm-processor-kwargs '{"max_pixels": 4194304}'`, `--max-num-batched-tokens 8192`. Boot log: `GPU KV cache size: ~148,960 tokens` (patch 3 installed; unpatched §6.3 would display ~37K) |
| Per-token attention KV at BF16 | `4 KV heads × 256 head_dim × 2 (K+V) × 16 attn layers × 2 bytes = 65,536 bytes/token` (16 GiB at the model's native 262K context) |
| Maximum concurrency at full max-len | **1.13×** at the §8.2 production flags (one full-context request fits with ~13% slack — `monkey_patch_hybrid_kv_allocator.py:335`) |
| `preserve_thinking` | Set as server-wide default via `--default-chat-template-kwargs '{"preserve_thinking": true}'` |
| MTP speculative decoding | OFF (head present in the AWQ checkpoint as BF16 but auto-skipped at load; §5.3) |
| Patches in this repo | 9 strict, fail-loud Python monkey-patches (§7), all server-side, loaded before `vllm serve` starts. Zero client-side code |

---

## 1. What this project is

A production deployment of **`QuantTrio/Qwen3.6-27B-AWQ`** (a 27-billion-parameter dense vision-language model from Alibaba, AWQ INT4 quantized) served via **vLLM** behind an OpenAI-compatible HTTP API at `http://127.0.0.1:8001/v1/chat/completions`, intended for agentic coding via the Qwen Code CLI on a single RTX 5090.

Five non-negotiable correctness requirements:

1. **Tool calling that actually works in multi-turn agent loops** — no silent failures.
2. **Preserved thinking across tool turns** — historical `<think>` blocks remain visible to the model on subsequent turns.
3. **Vision input at full preprocessing fidelity** — vision encoder kept BF16; HF `Qwen3VLImageProcessor` semantics preserved.
4. **131,072-token `--max-model-len` at BF16 KV cache precision** (FP8 KV rejected; §5.1).
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

### 5.2 Max context length: 131,072 tokens

`--max-model-len 131072`. The byte budget on a 32 GiB card at gmu=0.98 with `--skip-mm-profiling` (vision capability preserved at runtime; §5.8) and `--max-num-batched-tokens 8192` leaves **9.13 GiB** for KV — the boot log reports `GPU KV cache size: 148,960 tokens` (patch 3 §7.3 installed; the unpatched line would read ~37K under #6.3). At the per-token attention KV pinned in §3 (65,536 bytes/token), one full-context request fits with **1.13× concurrency** — the maximum we can keep without crossing into runtime-OOM territory. The model's native 262K context would need 16.0 GiB of attention-only KV; that doesn't fit without dropping precision (FP8 KV rejected, §5.1).

`--gpu-memory-utilization 0.98`. The hard ceiling on this hardware is `floor(free_memory / total_memory × 100) / 100 = 0.98` — vLLM's snapshot check at `vllm/v1/worker/utils.py:412` compares `requested = total × gmu` against `init_snapshot.free_memory`, which is taken **after** vLLM's own ~510 MiB CUDA-context init (PyTorch + cuBLAS/cuDNN + FlashAttention v2 + NIXL). That semantic — penalising vLLM for its own footprint — is closed by patch §7.8, which lets us cleanly land at 0.98. Beyond 0.98 (i.e. gmu=0.99) the runtime hits a torch.compile MLP-buffer OOM under fragmentation (272 MiB allocation needed, ~200 MiB physically free); 0.98 leaves ~520 MiB of slack and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (§8.2) eliminates the fragmentation pattern.

`--max-num-batched-tokens 8192`. vLLM's `_dummy_run(max_num_tokens, is_profile=True)` reserves activation peak ~linear in this number for the LM forward; the encoder-cache budget is `max(mnbt, max_tokens_per_mm_item)` so reducing mnbt below 16,384 does **not** shrink encoder memory (it stays at the per-image cap). 8192 is the auto-default sweet spot: prefill is chunked (33 chunks for one 130K-token cold prompt vs 17 at 16384) but the saved ~200-400 MiB of LM-activation peak is needed to clear gmu=0.98 with safety. Warm-path agent latency (90%+ prefix-cache hits) is invariant to the budget; cold-session full-context prefills pay the extra chunks **once per session**.

The admission gate (`scheduler_reserve_full_isl=True`, default) refuses any request whose ISL doesn't fit in the free KV pool — chunking is *within*-request, not *across* — so for a single in-flight max-len request to admit cleanly the pool must hold the full ISL. 148,960 ≥ 131,072 with ~17K slack — the 1.13× concurrency above.

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

**Patch §7.6 enforces these defaults server-side** for fields the client did NOT explicitly send (Pydantic v2 `model_fields_set`); explicit client values pass through unchanged, so the precise-coding alternative still works by sending `temperature=0.6, presence_penalty=0.0` from the client. `presence_penalty=1.5` is Alibaba's shipped mitigation for the Qwen3-family thinking-mode loop pathology (LiveCodeBench: 17.4% of 1,400 outputs truncated with no `</think>`, 27.5% on hard problems) — load-bearing whether or not the client sends it. May cause language mixing and slight perf decrease per Qwen's docs. Community alternative: `min_p=0.2` with `temperature=1.0` (drop `presence_penalty` back to 0). `max_tokens=16384` gives headroom against the truncation crash mode patched in §7.2. Qwen3.6 does NOT support soft `/think` or `/nothink` switches — thinking mode is controlled exclusively via `chat_template_kwargs`.

### 5.7 `preserve_thinking=true` as a server-side default

`--default-chat-template-kwargs '{"preserve_thinking": true}'` sets `preserve_thinking=true` as a server-wide default. The Qwen3.6 chat template only emits `<think>` blocks from historical assistant turns when `preserve_thinking=true`; otherwise only the most recent assistant turn retains its `<think>`. The model was RL-trained expecting reasoning to persist across turns; stripping it degrades tool-argument correctness after 2–3 turns (`badlogic/pi-mono#3325`).

`enable_thinking` is *not* set server-side — clients pass `"chat_template_kwargs": {"enable_thinking": true}` in the request body when they want thinking on.

The separate `reasoning` vs `reasoning_content` wire-format mismatch (§6.1 ingest, §6.4 egress) is closed by the §7.1 and §7.4 patches.

### 5.8 Multimodal cost accounting

- **Vision encoder weights** (~0.83 GiB) are always resident in VRAM, whether or not any request contains an image. Subsumed into the boot weights footprint.
- **Vision encoder profiling at boot is intentionally skipped** via `--skip-mm-profiling` (§8.2). The flag bypasses `gpu_model_runner.profile_run()`'s encoder dummy pass (`vllm/v1/worker/gpu_model_runner.py:5754-5817`), reclaiming ~1.56 GiB of profiler reservation that maps to ~25.6K KV tokens. **Runtime vision is unaffected** — `_execute_mm_encoder` runs on demand on real images and writes into `encoder_cache[mm_hash]`. The flag's only safety cost is that vLLM no longer measures the encoder peak at boot, so it cannot cross-check that runtime peaks fit. We mitigate by capping per-image vision tokens (next bullet) so the runtime peak is bounded; the bound holds inside the gmu=0.98 budget.
- **Per-image vision-token cap**: `--mm-processor-kwargs '{"max_pixels": 4194304}'` (§8.2) bounds the largest admissible image at 4 megapixels (~2048×2048), which Qwen3-VL tokenizes into at most 4,096 vision tokens (`max_pixels / (patch_size × merge_size)² / merge_size² = max_pixels / 1024`). Without this cap, the default 16,777,216-pixel ceiling lets a single image consume 16,384 KV-pool slots and produces an unbounded `embed_multimodal` activation peak that would OOM `_execute_mm_encoder` unrecoverably with `--skip-mm-profiling` enabled. 4 MP covers screenshots and document images cleanly; photos at 4K resolution are downscaled at preprocessing.
- **Per-request transient activations** are small (+2–10 MiB measured for 896×896 to 1792×1792 images); absorbed by safety headroom.
- **Image and video tokens occupy the KV pool**: ~256 tokens for 512×512, ~787 for 896×896, ~3,136 for 1792×1792, up to **4,096 per image at the 4-MP cap above**. Videos scale dramatically — short clips ~750–1,500 tokens; medium clips ~10,000–20,000; long clips can exceed 50,000. Bound video size with `--mm-processor-kwargs '{"video": {"max_frames": 64, "fps": 1}}'` if accepting untrusted clips.
- **Per-request budget formula** at the 4-MP cap: a request carrying `I` images can hold up to `131,072 − I × 4,096 − ~50 (framing)` tokens of text + system + completion. Reject path: vLLM raises HTTP 400 `"Input length (X) exceeds model's maximum context length (131072)."` at ingest if the total exceeds `--max-model-len`. Engine state intact on rejection.

`--limit-mm-per-prompt '{"image": 20, "video": 1, "audio": 0}'` is set generously. The flag is **load-bearing for boot** (omitting it defaults to 999 per modality, which crashes the boot on this nightly + SM 12.0 — post-profiling TensorRT-LLM `throwRuntimeError`), but otherwise it is **a pure per-request integer ceiling enforced at `chat_utils.py` ingest, not a memory reservation**. Verified: booting at `image:20` produces byte-identical KV reporting to `image:2` (9.13 GiB / 148,960 tokens / 1.13× concurrency); zero impact on boot, on text-only prompts, or on prompts carrying fewer images than the cap. The crash mode that fires at high N is *post*-profiling, which `--skip-mm-profiling` skips — values that would crash without the flag may also boot here, though we have empirically verified only up to 20. Set generously so realistic agentic flows never bump the ceiling; the cost of a high N is paid only when a request actually submits that many images (linear: each extra image at the 4-MP cap = 4,096 KV tokens, plus a ~1-2 s sequential encoder forward beyond N=2 since `encoder_cache_size = max(mnbt, max_tokens_per_mm_item) = 8,192` only fits two 4-MP images concurrently). Audio stays 0: Qwen3.6 is text+vision only.

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

`splitToolMedia: true` on the client is **redundant** with patch §7.9 installed (the auto-split closes the silent image-drop server-side for ALL clients regardless of the per-client flag); enabling it on the client is harmless but no longer load-bearing. Without patch §7.9, qwen-code commit `414b330` (#3617)'s per-message-split path is the only mitigation: with its default `false`, MCP-returned screenshots and `read_file`-emitted images stay packed in the `role: "tool"` content array, where vLLM's `chat_utils.py:1549-1564` reduces tool content to `"\n".join(text_items)` — silently dropping every image.

### 5.9 Chat-template invariants the operator must respect

Six hard invariants from the shipped `chat_template.jinja` and Qwen3.6 model code. Violations either raise at render time (HTTP 500) or silently degrade quality.

1. **System messages cannot contain images.** `chat_template.jinja:9-10` raises `'System message cannot contain images.'`; Qwen3.6 was trained with text-only system prompts.
2. **System messages cannot contain video.** `chat_template.jinja:20-21` raises analogously.
3. **Tool messages render text-only.** `chat_template.jinja:33` raises `'Unexpected item type in content.'` on any non-text part. Tool-result media must travel via a follow-up `role: "user"` message — closed automatically by patch §7.9.
4. **`add_vision_id` defaults to `false`.** The optional `Picture N: ` / `Video N: ` prefix for multi-image grounding (`chat_template.jinja:15-17, 26-28`) is OFF unless the operator sets `add_vision_id: true` via `--default-chat-template-kwargs`. ~5 extra text tokens per image when on; useful only for ordinal-index comparisons.
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

**Affects us**: yes, whenever a response is truncated by `max_tokens` or by client disconnect mid-generation. **Resolution**: §7.2; complemented by `max_tokens=16384` keeping truncation rare.

### 6.3 Issue #37121 — hybrid-KV scheduler/log under-reports concurrent capacity [Class D — observability bug]

vLLM's V1 paged KV cache manager forms one `KVCacheGroupSpec` per same-shape layer set. For Qwen3.6-27B that's 4 groups (1 full × 16 + 3 GDN × 16). The byte allocator at `vllm/v1/core/kv_cache_utils.py:1148-1169` allocates a single shared pool sized correctly. The bug is purely in *reporting*: `_report_kv_cache_config:1305-1346` and `get_max_concurrency_for_kv_cache_config:802-820` divide by `len(kv_cache_groups)` (4 for our model), so the displayed `GPU KV cache size: X tokens` and `Maximum concurrency` are ~4× understated. For the 27B-AWQ build under §8.2 production flags: boot log says `GPU KV cache size: ~37K tokens` unpatched; with patch 3 installed → `~149K tokens`. Operators sizing `--max-model-len` against the displayed number under-utilize their hardware.

**Upstream status (2026-04-28)**: issue #37121 open since 2026-03-15. PR #40384 (narrow scheduler-reporting fix, our backport source) and competing PR #40694 both open. PR #37429 (broader byte-level redesign) blocked on RFC.

**Resolution**: §7.3. Cosmetic for the deployment — byte allocation and admission are correct without it.

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

### 6.8 Tool-role media silently dropped at chat_utils ingest [Class A — vLLM internal inconsistency]

`vllm/entrypoints/chat_utils.py:1549-1564`'s `role == "tool"` branch reduces any list-shaped tool message content to `"\n".join(text_items)`. Every non-text content part (`image_url`, `audio_url`, `video_url`, `file`, `input_image`, `image_pil`, `image_embeds`, `audio_embeds`, `input_audio`) is silently discarded before the chat template runs. Compounding the bug: the per-content-part dispatcher upstream of the role-gate registered each media part with the `MultiModalItemTracker` *before* the reducer ran, so the encoder forwards data the rendered prompt never references — silent encoder/template desync, the same class A internal-inconsistency shape as §6.1 (the ingest path's "what counts as reasoning" disagreement with the chat template's view).

**Affects us**: yes — any client that returns tool media (MCP screenshots, `read_file`-emitted images, agent screenshots) is affected. The Qwen Code CLI's `splitToolMedia: true` is a client-side mirror of the server fix and is the only out-of-the-box mitigation absent a server patch; defaults to `false`.

**Resolution**: §7.9. Closes the silent drop server-side for ALL clients regardless of `splitToolMedia` setting.

---

## 7. The patches in this repo

Nine monkey-patches plus one container-entrypoint launcher plus one sitecustomize loader. Every patch is **server-side**, loaded into the vLLM Python process by `launch_with_patches.py` (in PID 1) and re-loaded by `sitecustomize.py` in spawned EngineCore subprocesses (load-bearing for patches 3 and 8; see §7.S). There is no client-side code in this repo.

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
| 9 | [`monkey_patch_tool_media_autosplit.py`](monkey_patch_tool_media_autosplit.py) | §6.8 — split `role: "tool"` content with media into a follow-up `role: "user"` message at ingest; closes silent image-drop for ALL clients regardless of qwen-code `splitToolMedia` setting |
| L | [`launch_with_patches.py`](launch_with_patches.py) | Container entrypoint that imports the 9 patches in order, runs per-patch verification, then hands off to `vllm.entrypoints.cli.main` via `runpy.run_module(alter_sys=True)` |
| S | [`sitecustomize.py`](sitecustomize.py) | CPython auto-imports this from `PYTHONPATH=/opt/patches` at every interpreter startup — including the spawned EngineCore subprocess — so patches 3 and 8's targets are live in EngineCore's own `sys.modules` |

### 7.1 Patch 1 — `monkey_patch_reasoning_field_ingest.py`

Wraps `vllm.entrypoints.chat_utils._parse_chat_message_content`. When `role == "assistant"`, `reasoning is None`, and `reasoning_content` is a non-None string, synthesises a shallow copy with `reasoning` populated from `reasoning_content`. Both fields present with **different** values raises `ReasoningFieldAmbiguityError` (HTTP 400). Identical values pass through. Non-string `reasoning_content` (dict, list) refuses rather than being stringified. **Removal trigger**: vLLM widens ingest to accept `reasoning_content`.

### 7.2 Patch 2 — `monkey_patch_qwen3_coder.py`

Replaces `Qwen3CoderToolParser._parse_xml_function_call` to use `str.find(">")` and, on a malformed `<parameter=` tag, return `None` for the whole tool call rather than raising `ValueError` (which the upstream `try/except Exception` collapses into "drop all N tool calls in this response"). Sibling well-formed tool calls in the same response are preserved. The MRO walk for inherited `self.tools` / `self.tool_call_parameter_regex` attribute landmarks is load-bearing per the prior audit (the inherited `self.tools` from `ToolParser.__init__` would refuse incorrectly under a non-walking check). **Removal trigger**: PR #39772 merges.

### 7.3 Patch 3 — `monkey_patch_hybrid_kv_allocator.py`

Replaces `get_max_concurrency_for_kv_cache_config` and `_report_kv_cache_config` in `vllm.v1.core.kv_cache_utils`. Both reporting sites divide by `len(kv_cache_groups)` (4 for our model: 1 attn + 3 GDN), making displayed token capacity ~4× understated. Patched to filter `kv_cache_groups` to token-capacity-contributing specs only (`AttentionSpec` always; `MambaSpec` only when `cache_config.mamba_cache_mode == "all"`). Under §8.2 production flags, boot-log `GPU KV cache size` goes from `~37K tokens` to `~149K tokens`. Backport semantics, not literal port — the pinned commit's `MambaSpec.max_memory_usage_bytes` signature is `(self, vllm_config)`, not the master signature PR #40384 targets. **Removal trigger**: PR #40384 or PR #40694 merges. **CRITICAL**: remove BEFORE pulling an image with PR #37429 (the broader byte-level redesign) — it changes the tensor layout and the patch's reporting view would no longer be coherent.

### 7.4 Patch 4 — `monkey_patch_reasoning_field_egress.py`

Installs Pydantic v2 `serialization_alias = "reasoning_content"` on the `reasoning` field of `ChatMessage` and `DeltaMessage`, flips `model_config["serialize_by_alias"] = True` on **all six** classes that vLLM dumps on the wire (the two leaves plus `ChatCompletionResponseChoice`, `ChatCompletionResponseStreamChoice`, `ChatCompletionResponse`, `ChatCompletionStreamResponse`), drops the cached `__pydantic_core_schema__` / `__pydantic_validator__` / `__pydantic_serializer__`, and calls `model_rebuild(force=True)`. A leaves-only patch is provably insufficient — Pydantic v2 compiled core schemas embed nested schemas by snapshot at build time. Phase 3 verification constructs real `ChatCompletionResponse` and `ChatCompletionStreamResponse` instances, dumps each via `model_dump_json()` and `model_dump_json(exclude_unset=True)`, and asserts wire bytes contain `"reasoning_content":` and not `"reasoning":`. The internal Python attribute stays `.reasoning`. **Removal trigger**: vLLM ships native `reasoning_content` on Chat Completions.

### 7.5 Patch 5 — `monkey_patch_tool_call_in_think_detector.py`

Wraps `Qwen3ReasoningParser.extract_reasoning` (non-streaming **only**) and emits a single structured WARNING (`model_emit_warning kind=tool_call_in_reasoning reasoning_len=N marker_count=M`) whenever the upstream-returned reasoning half contains literal `<tool_call>` markup. The wrapped return value is the upstream tuple **unchanged** — the agent's retry policy decides what to do. Detect, don't rescue: the parser is correct to its contract, the model is misbehaving, and a state-machine bandage across streaming deltas is high-complexity for a stochastic model-OOD failure mode. Streaming path is intentionally unwrapped — the rate is a model-side property, not per-modality, and a single non-streaming wrapper suffices. **Removal trigger**: Qwen3.6 retraining eliminates the OOD emission.

### 7.6 Patch 6 — `monkey_patch_default_sampling_params.py`

Wraps `ChatCompletionRequest.to_sampling_params` (`vllm/entrypoints/openai/chat_completion/protocol.py`). Runs the original, then for each Qwen3.6 Best-Practices default (`temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, max_tokens=16384`) overrides the returned `SamplingParams.<field>` iff `field_name not in request.model_fields_set`. Pydantic v2's `model_fields_set` distinguishes explicit assignments from class defaults (load-bearing — verified at import time on a real `ChatCompletionRequest` with both an unset and a set-to-class-default probe). `max_tokens` is capped at `min(qwen_default, current)` — never raised above the serving layer's already-applied cap. Five behavioral cases run at import time and refuse the patch if any fail. **Removal trigger**: vLLM ships a first-class "model-recommended sampling defaults" mechanism (e.g. `--default-sampling-params` widened to respect `model_fields_set`) and the launch flags adopt it.

### 7.7 Patch 7 — `monkey_patch_qwen3_coder_grammar.py`

Overrides `Qwen3CoderToolParser.adjust_request` (`vllm/tool_parsers/qwen3coder_tool_parser.py`) and flips `Qwen3CoderToolParser.supports_required_and_named = False`. Fixes two distinct issues with one extension-point override:

1. **Tool-emission grammar.** With `--tool-call-parser qwen3_coder` and the default `tool_choice="auto"`, the inherited `ToolParser.adjust_request` (`vllm/tool_parsers/abstract_tool_parser.py:85-122`) is a no-op — `get_json_schema_from_tools` returns None for `"auto"`. The model emits XML tool calls **completely unconstrained**; a post-hoc parser then extracts what it can. The patched `adjust_request` builds an xgrammar `structural_tag` payload mapping `<tool_call>\n<function=NAME>\n` → per-tool body schema → `\n</function>\n</tool_call>` from `request.tools`, installs it on `request.structured_outputs`, and sets `request._grammar_from_tool_parser = True` (mirroring the `mistral_tool_parser.py:286` pattern). **What the constraint enforces**: (a) the function name on the wire is always one of the registered tools — xgrammar's literal `<function=NAME>` framing makes hallucinated names structurally unreachable; (b) the `<tool_call>...</tool_call>` and `<function=…>...</function>` markers cannot be elided or duplicated. **What the constraint does NOT enforce**: parameter-body shape. Qwen3.6 emits XML (`<parameter=KEY>VALUE</parameter>`) *inside* the framing, but `structural_tag.schema` is JSON-oriented at xgrammar `≥0.1.32, <1.0.0` and does not validate JSON Schemas against XML body text. We include `schema` in the payload regardless: it round-trips cleanly through `xgr.Grammar.from_structural_tag`, and a future xgrammar release that enforces XML bodies against JSON Schema activates it without changing this patch. Until that lands, the §7.5 detector and the post-hoc qwen3_coder parser remain responsible for argument-shape correctness, and clients with strict-shape requirements validate `tool_calls[i].function.arguments` themselves. xgrammar's bitmask FSM stays dormant during `<think>...</think>` (`enable_in_reasoning: bool = False` default at `vllm/config/structured_outputs.py:41`) and engages only after `</think>`, so thinking-mode reasoning remains unconstrained. Per-request overhead xgrammar paper-measured at <6% TPOT.
2. **Latent `tool_choice="required"` bug**. With the inherited `supports_required_and_named=True`, the standard JSON-list path at `engine/serving.py:646-665` runs `TypeAdapter(list[FunctionDefinition]).validate_json(content)` on the model's XML output. The validation fails, `contextlib.suppress(ValidationError)` swallows the error, and the response is silently empty. Setting the flag to `False` routes `"required"` and named tool_choice through `extract_tool_calls` instead — the same path GLM4 uses for the same XML-shape reason.

The override is gated: skipped if `request.structured_outputs` or `request.response_format` is already set (never tramples explicit client intent), if `tool_choice == "none"`, or if the request is a `ResponsesRequest` (which uses `request.text` rather than `request.structured_outputs`). Phase 9 behavioral verification at import time constructs a real `ChatCompletionRequest` with a synthetic single-tool list, calls the wrapped `adjust_request`, parses the resulting `structural_tag` JSON, asserts `structures` and `triggers` keys are present and shaped correctly, and round-trips the payload through `xgrammar.Grammar.from_structural_tag(tags, triggers)` without raising — load-bearing proof that what the patch generates is what xgrammar can compile. Three negative controls (empty tools, explicit client `structured_outputs`, `tool_choice="none"`) prove the gate fires.

Does NOT fix the `<tool_call>`-in-`<think>` model OOD case (§6.5 / patch 5). That emission happens BEFORE `</think>` where the FSM is dormant by design; the §7.5 detector remains correct and complementary.

**Removal triggers**: vLLM merges `Qwen3CoderToolParser.adjust_request` upstream OR ships a `qwen3_xml` parser with grammar enforcement (PR #25028 community recommendation).

### 7.8 Patch 8 — `monkey_patch_request_memory_snapshot.py`

Replaces `vllm.v1.worker.utils.request_memory` to fix the §6.7 semantic bug. The original compares `requested_memory = total × gmu` against `init_snapshot.free_memory`; we compare against `init_snapshot.free_memory + INIT_SLACK` instead, where `INIT_SLACK` is a generous bound on vLLM's own post-init CUDA-context footprint (1 GiB by default; tunable via `QWEN36_VLLM_INIT_SLACK_MIB` env var for hosts whose footprint differs). The replacement returns a byte-identical `int` to all downstream consumers (`gpu_worker.py:421-424`'s `available_kv_cache_memory_bytes` math is unchanged); only the gate threshold changes. The original safety intent — fail loud when EXTERNAL processes consume significant GPU memory — is preserved: the patch still raises if external pressure exceeds the slack. Three import-time behavioural probes prove (a) the buggy predicate is still present in the image (forces a re-audit if upstream landed a fix), (b) the patched function returns the expected `ceil(total × gmu)` on the failure-mode probe, (c) a negative-control probe with external pressure > slack still raises. Without this patch, `--gpu-memory-utilization 0.98` is unreachable on this stack — boot fails with `Free memory on device cuda:0 (30.85/31.36 GiB) on startup is less than desired GPU memory utilization (0.98, 30.73 GiB)` even on an exclusive dGPU. **Removal trigger**: vLLM upstream rewrites `request_memory` to either measure pre-init or accept a slack parameter; the buggy `init_snapshot.free_memory < requested_memory` predicate is the landmark — any rewrite changes its shape and forces this patch to refuse.

### 7.9 Patch 9 — `monkey_patch_tool_media_autosplit.py`

Wraps the **outer funnel** `parse_chat_messages` and `parse_chat_messages_async` (`vllm/entrypoints/chat_utils.py:1600,1636`) with a pre-ingest pass that splits any `role: "tool"` message containing media parts into two messages: the original tool message with content reduced to its text-only parts (`tool_call_id` preserved), followed by a synthetic `role: "user"` message carrying ONLY the media parts. No text shim is added — qwen-code's shim is a strict-OpenAI workaround, not a model-quality requirement; Qwen3.6's chat template renders media-only user content cleanly. The wrapped wrapper composes with patch §7.1 (the inner-per-message `_parse_chat_message_content` wrapper) — patch 9 splits the messages list first, then the original `parse_chat_messages` calls patch 1's wrapper per (already-split) message.

The split is idempotent (re-applying the transform on its own output is identity by reference — already-split tool messages are text-only and the synthesized user message has no media parts of an unrendered shape) and preserves the chat-template invariant (§5.9 fact 3) that tool messages are text-only. The `multi_step_tool` reverse-walk at `chat_template.jinja:67-77` correctly identifies the synthetic user message as the last user query because its rendered content (a media marker, not text) does NOT start with `<tool_response>`. The split also restores encoder/template alignment: the `MultiModalItemTracker` registration now flows through the user message that the rendered prompt actually references.

Phase 4 carries five behavioural probes — text+media split, media-only tool, text-only tool (identity passthrough), string-content tool (idempotency under double-application), non-tool message with media (out-of-remit no-op) — plus a reflexive idempotency probe asserting two passes equal one pass by reference. Phase 6 verifies install via `getattr` and `inspect.getattr_static` on both targets.

Closes §6.8 server-side for ALL clients regardless of qwen-code's `splitToolMedia` setting; `splitToolMedia: true` on the client becomes redundant (still harmless — its output already hits the patch's no-op branches).

**Removal trigger**: vLLM upstream rewrites the `role:"tool"` content reducer at `vllm/entrypoints/chat_utils.py:1549-1564` to preserve media (or hoist the media into a follow-up message itself). The buggy `"\n".join(texts) if texts else ""` predicate is the landmark — any reshape changes its body and forces this patch to refuse.

### 7.L Launcher — `launch_with_patches.py`

Container entrypoint, replacing `["vllm", "serve"]` with `["python", "/opt/patches/launch.py", "serve", ...]`. Imports every registered patch in `_PATCH_MODULES` order, runs the per-patch `_PATCH_VERIFICATION` verifier for each (re-imports the relevant vLLM target FROM SCRATCH and asserts the install took effect), then hands off to vLLM's CLI via `runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)`. Required because `PYTHONSTARTUP` does not fire under non-interactive entrypoints.

Verifiers split into two classes. Patches 2 and 3 carry **behavioural** verifiers — they instantiate the patched class with a synthetic input designed to expose the bug and assert the post-patch return value. Patches 1, 4, 5, 6, 7, 8, 9 carry tag-only launcher verifiers (with `getattr` and `inspect.getattr_static` agreement); their patch-internal Phase verifications carry the load-bearing functional verification (the egress patch's wire-dump check; the ingest patch's static-lookup check; patch 6's five behavioural cases at Phase 7; patch 7's four behavioural cases at Phase 9 plus a load-bearing `xgr.Grammar.from_structural_tag` round-trip; patch 8's three behavioural probes — original-raises-on-buggy-input, patched-accepts-on-same-input, patched-still-raises-on-external-pressure; patch 9's five behavioural probes at Phase 4 — text+media, media-only, text-only, string-content, and non-tool message-list cases — plus a reflexive idempotency probe), so duplicating it here would only double the surface area. Patch 7's launcher verifier additionally asserts `Qwen3CoderToolParser.supports_required_and_named is False`.

Three pre-flight checks run BEFORE the per-patch import loop: sitecustomize-present (refuse if Debian's stub got loaded instead of ours), registry drift (refuse if `sitecustomize._PATCH_MODULES != launch_with_patches._PATCH_MODULES`), and a subprocess install probe (`subprocess.run([sys.executable, "-c", PROBE])` to confirm a freshly-spawned interpreter sees patched targets). All three are load-bearing for patch 3 — without them, the spawned EngineCore silently runs unpatched code while the launcher reports success. **Load order** matters: `reasoning_field_egress` (patch 4) must come before any patch that constructs `DeltaMessage` at request time, since the rebuild changes Pydantic's compiled schema.

### 7.S sitecustomize loader — `sitecustomize.py`

vLLM v1 spawns EngineCore as a `multiprocessing` child process via `spawn` (CUDA forbids `fork` after init). The spawned interpreter does not inherit `sys.modules`. Of the nine patches, **patches 3 and 8** target EngineCore-resident code (`monkey_patch_hybrid_kv_allocator` for the KV-cache reporting path, `monkey_patch_request_memory_snapshot` for the worker's startup `request_memory` check); patches 1, 2, 4, 5, 6, 7, 9 target API-server-resident code and become live in PID 1 directly (chat_utils' `parse_chat_messages` runs in the request handler, never in EngineCore). Without `sitecustomize`, patches 3 and 8 are silently dead in EngineCore while the launcher's PID-1 verifier reports success. The pass/fail discriminator for patch 3 is the boot-log filename annotation: `[kv_cache_utils.py:NNN]` (unpatched, ~37K tokens) vs `[monkey_patch_hybrid_kv_allocator.py:NNN]` (patched, ~149K tokens). For patch 8 the discriminator is whether the boot ValueError fires at gmu > ~0.984 (unpatched, snapshot-check too strict) or the engine reaches "Application startup complete" (patched).

CPython's `site.py` auto-imports `sitecustomize` from `sys.path` at every interpreter startup, including spawned children. With `PYTHONPATH=/opt/patches`, `site.py` finds our file, which imports each patch in launcher order. Each patch's strict landmark check runs in EngineCore too; any refusal aborts startup loudly. The same flow runs in PID 1 — sitecustomize installs the patches, `launch.py` then hits cache via `importlib.import_module(...)`. Each patch's module-level code fires once.

**Load-bearing for**: patches 3 and 8. **Defense-in-depth for**: patches 1, 2, 4, 5, 6, 7, 9. **Removal trigger**: when both patches 3 and 8 are removed; recommendation is to keep for defense-in-depth.

---

## 8. Deployment commands

End-to-end production sequence. Follow §8.1 → §8.5 in order from a fresh clone of this repo. The single canonical `docker run` block lives in §8.2 and is the only `docker run` in this README.

### 8.1 Pull the Docker image (one-time)

**There is no `Dockerfile` and no `docker build` step in this repo.** The deployment runs the upstream `vllm/vllm-openai` image **unmodified**, and the nine patches (plus `sitecustomize.py` and `launch_with_patches.py`) are bind-mounted into the container at `docker run` time via the `-v` flags in §8.2. The container's default `["vllm", "serve"]` entrypoint is replaced with `python3 /opt/patches/launch.py serve ...` so the launcher imports every patch — fail-loud on any landmark mismatch — *before* handing off to vLLM's CLI via `runpy`.

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
  --network host \
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
  -v "$PWD/monkey_patch_tool_media_autosplit.py:/opt/patches/monkey_patch_tool_media_autosplit.py:ro" \
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
  --host 127.0.0.1 --port 8001 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.98 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --mm-processor-kwargs '{"max_pixels": 4194304}' \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  --limit-mm-per-prompt '{"image": 20, "video": 1, "audio": 0}' \
  -cc '{"cudagraph_capture_sizes":[1,2,4,8]}'
```

Each flag's rationale lives in §5. Load-bearing items: `--max-model-len 131072` and `--gpu-memory-utilization 0.98` (byte budget, §5.2 — gmu=0.98 is reachable only with patch §7.8 installed; without it, the snapshot check refuses anything > ~0.984); `--skip-mm-profiling` paired with `--mm-processor-kwargs '{"max_pixels": 4194304}'` (§5.8 — the cap bounds the runtime encoder peak, which is otherwise unmeasured and unsafe with `--skip-mm-profiling`); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (eliminates the runtime fragmentation pattern that otherwise hits a 272 MiB MLP-buffer OOM at gmu near 0.98); `--limit-mm-per-prompt` (default crashes boot, §5.8); `--max-num-batched-tokens 8192` (auto-default for our hardware bucket; the encoder cache stays at `max(mnbt, 16384) = 16384` regardless, so reducing mnbt only shrinks LM-activation peak — and that peak is what gmu=0.98 needs to clear; §5.2); `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` (tool calling); `--reasoning-parser qwen3` (server-side `<think>` extraction); `--default-chat-template-kwargs '{"preserve_thinking": true}'` (§5.7); `-cc '{"cudagraph_capture_sizes":[1,2,4,8]}'` (pins the auto-derivation that vLLM produces for `--max-num-seqs 4` at `vllm/config/vllm.py:1434-1586` — drift-immune across image bumps); `--host 127.0.0.1` + `--network host` (loopback-only — do not change to `0.0.0.0` on a publicly-routable host without an authenticating reverse proxy; vLLM has no built-in auth); `sitecustomize.py` bind-mount (load-bearing for patches 3 and 8, see §7.S). Operational: `--restart unless-stopped` (auto-recover from engine crash); `--log-opt` rotation (bound docker log disk usage); `--health-cmd` (docker-native shallow liveness against `/health` — for deep wedge detection see §8.4). `--swap-space` is intentionally omitted (popped at LLM-API level and rejected by the CLI argparse in this image).

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
  '.data[0].id == "Qwen3.6-27B-AWQ" and .data[0].max_model_len == 131072'

# Step 3: confirm /metrics responds (Prometheus scrape target).
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/metrics  # expect 200

# Step 4: confirm all 9 distinct patch tags applied. Each patch loads in
# three processes (PID 1 launcher, pre-flight install probe, EngineCore
# spawn child) and emits one applied: line per loading, so the un-deduped
# line count is ~27; dedupe to count distinct tags.
docker logs qwen36 2>&1 | grep -oE 'qwen36-agent-setup-[a-z0-9-]+' | sort -u | wc -l   # expect 9
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
         \"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}" \
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
    "temperature": 0.6, "max_tokens": 4096,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}}
  }' | jq .
```

Operational: vLLM exports Prometheus metrics at `/metrics` — scrape for KV cache pressure, queue depth, request latency. Patch §7.5 emits `model_emit_warning kind=tool_call_in_reasoning` to docker logs when the model misbehaves; alert on it via your log-forwarder.

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

### 8.5 Production-ready checklist

Sign off all of these before declaring the deployment ready:

- [ ] `docker inspect -f '{{.State.Health.Status}}' qwen36` reports `healthy`.
- [ ] `docker logs qwen36 2>&1 | grep -oE 'qwen36-agent-setup-[a-z0-9-]+' | sort -u | wc -l` returns **9** (the nine distinct patch tags). Patches load in three processes — PID 1 launcher, the launcher's pre-flight subprocess install probe, and the spawned EngineCore — so the un-deduped `applied:` line count is **~27**.
- [ ] `curl -fs http://127.0.0.1:8001/v1/models | jq '.data[0].max_model_len'` returns **131072**.
- [ ] `curl -fs -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/metrics` returns **200**.
- [ ] `/usr/local/bin/qwen36_deep_probe.sh` returns **0**.
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

Each is tracked; none urgent.

---

## 13. File structure of this project

```
.
├── README.md                                              # this document
├── launch_with_patches.py                                 # §7.L — container entrypoint; imports the 9 patches then runpys vLLM
├── sitecustomize.py                                       # §7.S — auto-loads patches in EngineCore (and PID 1) at interpreter startup
├── health_probe.sh                                        # §8.4 — host-side deep liveness probe; engine-decoded-a-token check
├── monkey_patch_reasoning_field_ingest.py                 # §7.1 — accept reasoning_content on inbound assistant messages
├── monkey_patch_qwen3_coder.py                            # §7.2 — parser crash fix on truncated <parameter=
├── monkey_patch_hybrid_kv_allocator.py                    # §7.3 — hybrid-KV scheduler-budget fix (PR #40384 backport)
├── monkey_patch_reasoning_field_egress.py                 # §7.4 — Pydantic serialization rename reasoning → reasoning_content
├── monkey_patch_tool_call_in_think_detector.py            # §7.5 — detect <tool_call> emitted inside <think>; structured WARNING
├── monkey_patch_default_sampling_params.py                # §7.6 — server-side Qwen3.6 sampling defaults for unset fields
├── monkey_patch_qwen3_coder_grammar.py                    # §7.7 — xgrammar structural_tag on tool emission; supports_required_and_named=False
├── monkey_patch_request_memory_snapshot.py                # §7.8 — startup snapshot check stops double-counting vLLM's own init footprint
├── monkey_patch_tool_media_autosplit.py                   # §7.9 — split role:"tool" content with media into a follow-up role:"user" message
└── tests/
    └── test_patches_against_master.py                     # static + structural-mirror suite (runs without torch/CUDA)
```

Eleven Python files (9 patches + launcher + sitecustomize) plus the host-side probe shell script and one test file.
