# HW2 sharding and communication — source analysis and measured status

Source: the frozen HW1 adaptation of the course fork, especially
`python/tokenspeed/runtime/models/qwen3_5.py`,
`runtime/distributed/{mapping,comm_manager,process_group_manager}.py`,
`runtime/distributed/comm_backend/hccl.py`, and
`runtime/execution/distributed_initializer.py`.
HCCL and all four TP2 models have now run; [latest evidence](final-validation.md) separates completion from strict numerical correctness. The shapes below remain source analysis, not measured trace shapes. [Bounded NPU communication results](communication-results.md) independently verify actual operations/payloads for 0.8B only.

## Launch and topology

`run_model.py` explicitly selects world size, attention TP, dense TP, MoE TP and LM-head TP all equal to 1 or 2; PP/EP remain 1. This is one sharded model, not two replicas. Engine `_launch_subprocesses` starts one scheduler process for each global rank. `Mapping.gpu_id` maps local rank 0/1 onto logical NPU 0/1, after `ASCEND_RT_VISIBLE_DEVICES` selects physical IDs.

The distributed initializer selects backend `hccl` for NPU, initializes HCCL/Gloo groups, and checks free-memory balance between ranks. Do not bypass its imbalance error or terminate another job to pass it. Fresh `npu-smi` inventory must establish that both IDs belong to the **same physical card**; adjacency alone is not proof. Visibility is not a reservation. Prior inventory is historical, not current allocation evidence.

## Representative 0.8B layer, TP=2

Let T be scheduled token rows. Dimensions below are logical BF16 weight shapes `[out,in]`, excluding implementation padding. Both ranks receive identical token IDs and hidden rows; weights and head-local state are sharded.

| Component | Each rank | Communication / state |
| --- | --- | --- |
| Embedding | Logical weight `[124160,1024]` of vocab 248320 | Each rank masks out-of-shard IDs; partial embeddings combine across TP |
| GDN fused input projection | `[4112,1024]`; logical Q/K/V/Z each 1024 channels plus b/a each 8 | Column-parallel, independent local GEMM; no exchange of head-local recurrence |
| GDN conv | Weight `[3072,1,4]`; raw history `[pool,3072,3]` | Channels are sharded in Q/K/V segments, not arbitrary mixed slices |
| GDN recurrence | Q/K/V `[T,8,128]`; FP32 state `[pool,8,128,128]`, K-last | Independent head recurrence; cache allocator still owns page/request identities on both ranks |
| GDN output projection | Weight `[1024,1024]`; partial output `[T,1024]` | Row-parallel partial sums must combine across ranks |
| Full attention | 4 Q heads, 1 KV head, D=256 per rank; output gate has the same local Q-head width | KV cache `[pages,page_size,1,256]` per K/V; causal attention is head-local |
| Attention output projection | `[1024,1024]` on each rank; partial `[T,1024]` | Row-parallel combine across TP |
| MLP gate/up | Two logical `[1792,1024]` shards, packed `[3584,1024]` | Local SwiGLU activation `[T,1792]` |
| MLP down | `[1024,1792]`; partial `[T,1024]` | Row-parallel combine across TP |
| Norm weights / residual | Hidden norm `[1024]`, GDN per-head norm `[128]` replicated | Identical logical residual rows after required collective; weight offset stays FP32 |
| LM head | Logical `[124160,1024]` per rank, tied embedding where configured | Vocab-sharded logits; selection/logprob operations require distributed reductions/gathers |

`CommManager.use_all_reduce` chooses AllReduce when attention TP equals dense TP (our configuration). RowParallelLinear can defer reductions to this manager; do not count a missing inline reduction as a missing collective. Alternate unequal group widths can select ReduceScatter/AllGather, but are **not** the proposed baseline. HCCL's small INT64 MAX branch implements packed greedy selection by INT32-bit-view AllGather followed by local exact MAX, rather than unsupported AIV INT64 AllReduce. Logs/profiling must verify the actual selected paths.

A replicated BF16 `[T,1024]` partial tensor has `2048*T` payload bytes per rank before AllReduce protocol traffic. This is tensor payload, **not measured link bytes**. Decode rows 1 versus 2 change payload from 2048 to 4096 bytes. Ring/tree algorithms and transport overhead must not be inferred from those numbers.

## Four-model metadata and capacity planning

Official per-file revisions, SHA256 and config files are retained under `metadata/`; `catalog.json` is derived from those files. All four checkpoints were available for the retained server runs; local metadata is not itself proof of resident weights or current free memory. The 0.8B checkpoint came from HW1.

| Model | Layers / hidden / MLP | Global Q/KV heads | Global GDN K/V heads | Checkpoint bytes |
| --- | --- | --- | --- | ---: |
| 0.8B | 24 / 1024 / 3584 | 8 / 2 | 16 / 16 | 1,746,942,600 |
| 4B | 32 / 2560 / 9216 | 16 / 4 | 16 / 32 | 9,319,828,096 |
| 9B | 32 / 4096 / 12288 | 16 / 4 | 16 / 32 | 19,306,310,880 |
| 27B | 64 / 5120 / 17408 | 24 / 4 | 16 / 48 | 55,563,022,432 |

All have full-attention D=256 and GDN K/V dimensions 128. TP=2 yields local GDN K/V heads 8/8, 8/16, 8/16 and 8/24 respectively. Grouped-value-head coverage matters for the larger models. The saved checkpoints include vision assets; disk bytes are not exact text-only resident HBM. These counts are planning inputs, not proof that any larger model fits alongside current jobs. Check free disk/RAM and each chip's HBM before downloading/loading; test models sequentially.

## Experiment status and communication hypothesis

1. **HCCL completed:** both ranks passed FP32/BF16 collectives, with project-backend regression and original card-0 inventory retained. When physical 0/1 later became busy/unbalanced, the 4B prefix failure was preserved; fresh inventory established **physical 4/5 on card 2** for subsequent TP2 audits/profiles. No imbalance guard was bypassed.
2. **Four-model generation and selected-config comparisons completed:** each model generates 16 tokens for inputs 5/32/256 on TP2. Matched pairs pass for 0.8B BF16 (c1/c2), 9B FP16, 27B BF16/fraction 0.9, and **4B FP16 with explicit FP32 final projection**. The 4B rounding tie was [captured and fixed](4B-discrepancy.md), with two fresh full-generation pairs and a same-source native control. Earlier native/BF16 failures remain historical; no tolerance changed or narrower audit substituted for full generation. See [completion](../COMPLETION.md).
3. **Unprofiled matrix and HBM completed:** refreshed 0.8B input 32/256 × c1/c2 × TP1/2, three repeats, 16 outputs, all 36 measured requests checked. Complete configuration-matched child-owned HBM snapshots now exist; old missing rows remain historical unknowns. New 27B TP2 benchmark matches its passing direct TP1/TP2 configuration. These are batch wall latency/aggregate throughput and memory snapshots, not TTFT/TPOT, peaks or isolated communication timing.
4. **Bounded communication experiment completed:** fixed 0.8B TP2, input 32, output 2, one warmed profile batch at each c1/c2, both ranks. [Actual traces](communication-results.md) show **102 collectives/request/rank** at either concurrency, actual BF16 decode AllReduce input **2,048→4,096 bytes**, and 51→25.5 collectives per returned token. This supports launch-count amortization descriptively, **not general communication-time savings**: envelope ms/token worsens on rank0, improves on rank1. Selected compute/envelope union overlap is zero only in the stated windows; unclassified gaps are not pure CPU wait, and 51 tail ops/rank are excluded. Do not add overlapping intervals or sum rank times into end-to-end latency.

[Explicit NPU capture](npu-profile-run.md) supersedes the old missing-rank profile within this short-window scope; long/256-input vendor-parser failures remain limitations. Technical deliverables are complete within the recorded scope, including the resolved 4B numerical case. Student review, actual contributions and submission remain separate; nothing has been submitted.
