# Memory tuning: the KV-pool coupling (why "free some RAM" bricks the boot)

The tuned profile here runs at `gpu-memory-utilization 0.85` / `max-model-len 1048576` /
`max-num-seqs 6`. Sooner or later you'll want to reclaim some of that unified memory — to co-host
another process on the head node, or just to shrink the standing footprint. On the GB10's
**unified memory** the three memory knobs are coupled in a way that isn't obvious, and the natural
move — *"just lower `gpu-memory-utilization`"* — will brick the serve on the next restart. Here's
the model and the safe recipe, measured.

## The three knobs actually control one pool

- **`--gpu-memory-utilization`** sizes the KV pool: vLLM reserves `util × total`, loads weights
  (~78 GB/node for this model), and the *leftover* becomes KV. On unified memory the un-reserved
  remainder is also what other host processes can use — so this is your "free RAM" lever, but it
  trades directly against KV.
- **`--max-model-len`** is a *feasibility gate*, not an allocation. vLLM refuses to boot unless the
  KV pool can hold at least one request of this length. Lowering it frees nothing by itself — it
  just lets you lower util without tripping the gate.
- **`--max-num-seqs`** caps concurrent decode **and** sizes the CUDA-graph capture ladder
  (≈ `seqs × (MTP+1)` — the deployed `6 × (5+1)`; see finding #3 in the main README). More seqs ⇒
  more captured graphs ⇒ less KV. Finding #2 is this same coupling from the other side: dropping
  seqs 12→6 doubled the pool.

## The gotcha (measured)

At the tuned config, the boot banner reports:

```
Available KV cache memory: 19.5 GiB
GPU KV cache size: 2,902,470 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.77x
```

~2.9M tokens of KV — but a real workload rarely needs more than a few concurrent ≤20K-token
requests (<1 GiB). That ~18 GiB looks like free-for-the-taking slack.

**Naive reclaim — lower util, and bump seqs while you're at it — fails to boot:**

```
util 0.85 → 0.70,  max-num-seqs 6 → 16     (max-model-len left at 131072)
→ ValueError: To serve at least one request with the model's max seq len (131072),
  3.79 GiB KV cache is needed, which is larger than the available KV cache memory (3.12 GiB).
  Based on the available memory, the estimated maximum model length is 7740.
```

Two effects stacked, both invisible if you treat the knobs as independent: util 0.70 collapsed the
pool to **3.12 GiB** (weights eat nearly the whole reduced budget, so the KV leftover falls far
faster than the util delta), and seqs 6→16 ballooned the capture ladder 36→96, eating more of
what's left. vLLM then refused to start because the pool can't hold even one `max-model-len`
request.

**The fix — couple util down *with* max-model-len down, leave seqs alone:**

```
util 0.85 → 0.75,  max-model-len 1048576 → 65536,  max-num-seqs 6 (unchanged)
→ Available KV cache memory: 8.35 GiB
  GPU KV cache size: 173,197 tokens
  Maximum concurrency for 65,536 tokens per request: 2.64x
```

Boots first try, frees ~12 GiB of unified memory for a co-tenant process, and the pool still holds
~20× the real workload's peak. (64K covers the vast majority of real traffic; the 1M reservation
was pure standing cost.)

## Rules

- **To free host RAM:** lower `gpu-memory-utilization` **and** `max-model-len` *together* — the
  smaller pool must still pass the feasibility gate. Do **not** raise `max-num-seqs` at the same
  time (it pushes KV the same direction as lowering util).
- **To add concurrency:** raise `max-num-seqs` **alone** (keep util / max-model-len). Streams are
  cheap in KV — the pool already holds far more than the default — but watch the capture-ladder
  growth (finding #3).
- **Never predict the KV pool from byte math.** MLA + NVFP4 + sparse attention make per-token KV
  cost non-obvious (this model's feasibility check implies ~29 KB/token, while the compressed pool
  counts ~6.7 KB/token — different accountings). Change one variable, restart, and read vLLM's
  `Available KV cache memory` / `GPU KV cache size` banner — it's authoritative; your spreadsheet
  isn't. When a config won't boot, the `ValueError` prints the *estimated maximum model length* for
  the current memory, which tells you exactly how far to drop `max-model-len`.
- **Wrap the restart so it auto-reverts** if the new config doesn't come up healthy within ~10 min:
  a KV-starve failure looks identical to a slow boot until it isn't.

## References

- vLLM "Conserving Memory" docs — `gpu_memory_utilization`, `max_model_len`, CUDA-graph capture.
- [`JustVugg/colibri`](https://github.com/JustVugg/colibri) — the disk-streaming extreme of the same
  active-vs-total-params physics: a 744B MoE from NVMe on 25 GB of RAM.
- The small-unified-memory end of the family (Jetson Orin Nano, llama.cpp):
  [`jetson-orin-llama-cpp-gpu`](https://github.com/CiphemonJY/jetson-orin-llama-cpp-gpu).
