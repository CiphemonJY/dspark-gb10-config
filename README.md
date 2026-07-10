# DeepSeek-V4-Flash-DSpark on 2× DGX Spark GB10 — tuned 1M / RoCE config

Sanitized, field-verified configuration for serving **DeepSeek-V4-Flash-DSpark (NVFP4)** with
vLLM TP=2 across two NVIDIA DGX Spark (GB10, sm_121) nodes at **1M context** over **RoCE v2 RDMA**.

This is a config overlay for
[tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
— all Dockerfiles, compose files, and launch scripts come from that recipe (credit: tonyd2wild,
Keys, drowzeys, roady001, Wpnx330, 0rand, paulbrav, and the DSpark PR authors). This repo adds the
tuned serving profile and one launch-script fix we validated in production, plus measured results.

## Results (A/B on the same hardware, 256-token code-completion probes, temp 0)

| concurrency | before: 262k ctx / seqs 12 / MTP 3 / TCP NCCL | after: 1M ctx / seqs 6 / MTP 5 / RoCE | Δ |
|---|---|---|---|
| 1 | 36.7 tok/s | **46.6** | +27% |
| 2 | 58.0 | **81.8** | +41% |
| 4 | 79.0 | **129.2** | +64% |
| 6 | 115.1 | **130.0** | +13% |

Context ceiling 262,144 → **1,048,576** tokens, KV pool 1.44M → **2.94M tokens** (2.80×
simultaneous-full-1M concurrency), boots fully offline from the HF cache.

## The three findings that matter

1. **"RoCE is flaky" was a config bug, not a fabric problem.** The upstream start script copies
   `.env.dspark` to the worker verbatim, so the worker inherits the *head's* HCA name. If your
   nodes enumerate HCAs differently (ours: `mlx5_1` vs `rocep1s0f1`), NCCL on rank 1 finds no
   device and dies — which reads as random RDMA flakiness and pushes you to the TCP fallback,
   costing 20–40% throughput. Fix: per-node exact-match HCA pinning
   ([patches/worker-hca-override.patch](patches/worker-hca-override.patch)). Verify with
   `docker logs <container> | grep NET/IB` → must show `Using [0]<hca>:1/RoCE`.

2. **`MAX_NUM_SEQS` 12 → 6 doubled the KV cache pool** — 1.44M → 2.99M tokens at the same
   `gpu-memory-utilization 0.85` (measured at MTP=3; the MTP=5 bump below trims it to the
   deployed 2.94M) — fewer reserved decode slots and a smaller cudagraph ladder. The pool is
   demand-allocated, so short-prompt traffic keeps all 6 slots usable.

3. **`MTP_NUM_TOKENS` (DSpark speculative depth) 3 → 5 only pays on RDMA.** On TCP NCCL it was
   throughput-neutral: each extra draft pass is another sequential hop over the interconnect, and
   TCP per-op latency eats exactly what deeper speculation buys. On RoCE it delivers the +24–27%
   the upstream measurements promise. Keep `draft_sample_method=probabilistic` — greedy drafts
   are the documented cold-start garble root cause. Note: vLLM truncates the requested capture
   size 36 → 32; the ladder still contains a multiple of (k+1)=6, which is what DSpark requires.

## Files

- [`env.dspark.example`](env.dspark.example) — the full tuned profile (sanitize-adapted: replace
  hostnames/IPs/HCA names with yours).
- [`patches/worker-hca-override.patch`](patches/worker-hca-override.patch) — the one-line
  start-script fix for asymmetric HCA names.

## Caveats

- GB10/sm_121 cannot run the official FP8 checkpoint (DeepGEMM asserts `Unsupported architecture`);
  the NVFP4 model + B12X MoE backend image from the upstream recipe is mandatory.
- If you front the server with a proxy, check its timeouts: a genuine ~1M-token prefill takes
  minutes; a 300s proxy timeout will kill it mid-prefill and evict your prefix cache.
- The upstream `correctness_test.py` PASS threshold (`acceptance > 0.4` under churn) is calibrated
  for MTP=3; at MTP=5 churn acceptance runs ~0.33–0.37 with byte-identical output. Trust the
  `OUTPUT IDENTICAL` line.
