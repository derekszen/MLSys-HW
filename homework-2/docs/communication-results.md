# Actual bounded NPU communication analysis

**Four local CANN traces validate real TP2 device computation and HCCL communication. Batch 2 doubles BF16 collective input sizes at the same collective count, but this single profile pair does not establish a communication-time or general performance improvement.** The exact request windows contain 102 collectives per rank, not the 153 present in each whole capture.

Machine-readable evidence: [`npu-communication-analysis.json`](../results/npu-communication-analysis.json). Reproducer: [`analyze_npu_traces.py`](../scripts/analyze_npu_traces.py). [Actual NPU capture and test record](npu-profile-run.md) documents the successful explicit `activities=["NPU"]` route and superseded `msprof` failures. The analyzer reads only local files; original results/raw traces remain unchanged. Deterministic gzip archives provide compact evidence; archive-aware reproduction matches the official metrics, with selected-file provenance differences described below.

## Scope and provenance checks

These are `results/0.8B-npu-batch-experiment-c{1,2}/`: Qwen3.5-0.8B, BF16, eager TP2, physical devices **4/5 on card 2**, 32 input tokens, two returned output tokens per request. Each concurrency has **one profiled request batch after a matching warmup**, not repeated independent measurements. Prefix caching is disabled; output metadata reports zero cached tokens.

- Both profile results have exactly equal engine configuration, recorded source inventories/tree hashes, runtime versions, and selected environment. Recorded versions: torch `2.9.0+cpu`, torch_npu `2.9.0`, triton `3.2.0`. Local manifest, model configuration, and cases SHA256 match both runs and canonical `0.8B-correctness-tp2-c1/result.json`. The manifest identifies `Qwen/Qwen3.5-0.8B`; its config entry matches the local config hash. Checkpoint weights were **not independently rehashed** by this analysis.
- Every warmup/profile output is `[9064, 11988]`, exactly the first two canonical `length_32` IDs. This is a two-token-prefix check, not a new 16-token/full-model numerical certification.
- The canonical correctness run predates the NPU profile-hook change. Its source inventory differs **only** at `python/tokenspeed/runtime/engine/request_handler.py` (`775fd01a…` canonical; `99b7571d…` profile). Other source files match. The analyzer permits only that historical file difference; it requires the **actual c1/c2 profile sources to match exactly**. Inventories alone do not reconstruct or audit the old file's code diff.
- The existing JSON contains full SHA256 and sizes for both result files, canonical result, cases, model metadata, collection/analysis scripts, relevant HBM snapshots, and all four raw traces. Each raw trace's SHA256, byte size, and event count match its recorded result entry. Archive-aware regeneration preserves these uncompressed trace hash/size semantics and additionally fingerprints the actual selected file, as described below. Ordered collective names/counts/dtypes match between TP peers within each concurrency.

Trace SHA256 prefixes (full hashes in JSON): c1 TP0 `1358fc202bfd`, TP1 `6fa94df5f1c9`; c2 TP0 `0c57d11bc36f`, TP1 `bf2f1a014160`. Event counts respectively: 164,543; 164,512; 251,829; 251,796. Input bounds are 80 MB and 400,000 events per trace, parsed one trace at a time without printing event arrays.

## Raw and archived trace formats

Each expected logical `hw2-npu-c{1,2}-input32-TP{0,1}.trace.json` may be supplied as raw JSON, `.trace.json.gz`, or both. The analyzer selects gzip when both exist, without reading or hashing the unselected raw file. It compares the **selected representation's uncompressed SHA256, byte size, and event count** with the original `result.json` entry. A mismatching or invalid preferred archive fails closed; there is no fallback to raw. The two representations are one logical grid entry, not extra ranks. Unknown trace filenames and extra suffixes are rejected. Ownership validation ignores only the optional final `.gz` when checking the TP suffix.

In regenerated reports, `trace.sha256` and `trace.bytes` remain the **uncompressed** hash and size; `trace.path` names the actual selected file. `trace.stored_file` separately records that file's exact path, on-disk byte size and SHA256 (the compressed hash for gzip), and `trace.compression` is `gzip` or `none`. The selected file also has a top-level provenance entry with its stored-file hash/size plus explicit `uncompressed_sha256` and `uncompressed_bytes`. Thus archive bytes and original trace identity are independently auditable. A previous raw-only summary predated these additional fields. The currently linked `npu-communication-analysis.json` is already archive-aware and contains `trace.stored_file`, `trace.compression`, and compressed-file provenance. Regeneration in another checkout can change recorded file paths without changing trace identities or measured metrics.

Both stored input and actual decompression are capped at **80,000,000 bytes**. Inflation reads at most that bound plus one rejection byte before JSON parsing; it does not trust gzip's size trailer. Truncated/corrupt streams and concatenated members exceeding the aggregate limit are rejected. Decimal parsing and the 400,000-event limit are unchanged.

The four adjacent archives use gzip level 9, `mtime=0`, and no embedded filename. Their decompressed bytes were compared exactly with the untouched raw files and their SHA256/size verified against `result.json`; regeneration with the same settings produced identical archive bytes.

| Archive (within its c1/c2 result directory) | Uncompressed bytes | Archive bytes | Archive SHA256 prefix |
|---|---:|---:|---|
| `hw2-npu-c1-input32-TP0.trace.json.gz` | 38,189,624 | 2,193,213 | `81c6feb7b7d2` |
| `hw2-npu-c1-input32-TP1.trace.json.gz` | 38,178,383 | 2,201,903 | `5bf0d0266301` |
| `hw2-npu-c2-input32-TP0.trace.json.gz` | 57,032,958 | 3,269,180 | `8d577321e677` |
| `hw2-npu-c2-input32-TP1.trace.json.gz` | 57,022,601 | 3,288,503 | `2d3d7b9fee58` |

Total: **10,952,799 archive bytes versus 190,423,566 raw bytes**, a 94.248% reduction in trace bytes to track. Only these four uncompressed paths are ignored by `.gitignore`; the gzip files remain eligible for tracking. Uncompressed originals are preserved; no whole-results archive is required.

## Ownership and lane attribution

| Capture | TP / logical device | Owned scheduler PID | Hardware Chrome PID | Communication Chrome PID | Physical / card / chip |
|---|---:|---:|---:|---:|---|
| c1 | 0 | 2934402 | 3004828068 | 3004828132 | 4 / 2 / 0 |
| c1 | 1 | 2934403 | 3004829093 | 3004829157 | 5 / 2 / 1 |
| c2 | 0 | 2936113 | 3006580132 | 3006580196 | 4 / 2 / 0 |
| c2 | 1 | 2936114 | 3006581157 | 3006581221 | 5 / 2 / 1 |

The `TP0/TP1` filename tags agree with owned result children named `tokenspeed::scheduler_tp0/1`, their runner PPIDs, **CANN main-thread metadata identifying those exact worker PIDs**, and device HCCL subtask `src rank` values 0/1. There is **no literal `RankId` metadata field in these exports**; none is invented. Chrome process IDs above are virtual lane identifiers, not Linux PIDs. No PID arithmetic is used as ownership proof.

`process_name = Ascend Hardware` and `process_name = Communication` each have `process_labels = NPU 4/5` as appropriate. The HCCL group lane is `Group group_name_1 Communication`, thread 0; plane lanes contain actual transport/notify tasks. Logical 0/1 follows `base_gpu_id=0`, step 1 and `ASCEND_RT_VISIBLE_DEVICES=4,5`, independently checked against the trace's physical labels. Saved `hbm-loaded.txt` maps physical 4/5 to card 2 chips 0/1 and lists each owned worker on its expected chip. A worker can also hold a small context on another chip; PID presence is not exclusive allocation. Other jobs are visible in these snapshots, so no exclusive-device or peak-HBM claim follows.

## Clock alignment and exact clipping

All timestamp arithmetic uses **Decimal microseconds**, including JSON numeric durations and timestamp strings. Unix nanoseconds are divided by exactly 1000, never converted through float. The saved request intervals are:

```text
c1: [1792817202870924.380, 1792817203186481.100) us
c2: [1792817251847974.610, 1792817252301412.340) us
```

For **each rank**, the CANN `PROFILING_ENABLE`/`PROFILING_DISABLE` hardware points bracket the request on that epoch, with each boundary margin below one second. Runner Unix and monotonic durations agree within 1.960 us (c1) / 1.700 us (c2). This is the operational same-epoch check, not just a guess from timestamp magnitude. No offset is fitted. JSON retains every profiler point and check result. This verifies request-window clipping for these captures, **not external sub-microsecond clock calibration**. If the check fails, the analyzer withholds request metrics and reports only device-active first/last spans, never as request wall time.

Runner wall durations are **315.556720 ms (c1)** and **453.437730 ms (c2)**. Whole captures include additional device activity ending 69.316–74.298 ms after those request ends. Each rank has **49 AllReduce + 1 AllGather + 1 Broadcast starting after request end**. The cause is not established here: this must not be silently counted as request latency or attributed to an extra forward/draining mechanism without separate evidence.

## Interval definitions and double-count prevention

Compute is the union of hardware tasks typed by `args['Task Type']`: `AI_CORE`, `AI_VECTOR_CORE`, `MIX_AIC`, `MIX_AIV`, plus non-HCCL `AI_CPU`. These are device tasks, not CPU API timings. Local operations such as `aclnnReduceSum…` remain compute; a name containing “reduce” is not sufficient communication evidence.

Communication is the union of **device HCCL operation envelopes** on the metadata-established communication group lane: `hcom_allReduce__…`, `hcom_allGather__…`, and `hcom_broadcast__…`, with actual `count`, `data_type`, and connection IDs. Overlapping/nested parents are rejected rather than counted ambiguously. Plane subtasks are **not added to the envelopes**. Every one of the 7,245 TP0 / 7,230 TP1 subtasks maps to an identical-time hardware stream/task, independently proving device communication evidence.

Each full capture also contains 153 HCCL `AI_CPU` kernels (`allreduceAicpuKernel`, `allgatherAicpuKernel`, `broadcastAicpuKernel`). Their ordered kinds match the envelopes; host `Node@launch` connection/item IDs and hardware notify dependencies verify their association. They are excluded from compute and **not added again** to communication envelopes. Host `HcclAllReduce` API timing would likewise not count as device communication; these NPU-only exports contain zero host events with that literal API name. Host launch/notify records are present but are not substituted for device operation envelopes.

An important exporter limitation is retained, not hidden: **6 / 70 / 112 / 12** plane subtasks are not contained within their parent envelopes (c1 TP0/TP1, c2 TP0/TP1). Likewise 1 / 28 / 40 / 4 ordered HCCL kernels have no temporal overlap with their corresponding envelope. Kernel association uses launch/dependency evidence and ordered operation kinds, not assumed temporal containment. JSON reports HCCL kernel time outside envelopes separately. This makes envelope-based communication a precise trace metric, **not an exhaustive communication-engine busy-time measurement**.

Explicit control is the hardware union of `EVENT_WAIT/RECORD`, `NOTIFY_WAIT/RECORD` (including `_SQE`) and `WRITE_VALUE_SQE`. These can overlap work on other streams: they are not whole-device stall or CPU wait. DMA and HCCL auxiliary kernels outside envelopes are not reclassified as ordinary compute. Gaps mean time not covered by the selected compute/envelope/control unions, including known-but-unbinned auxiliary/device activity, transfers, host scheduling, and launch gaps. **They are not “pure CPU waiting.”** The vendor-derived `Overlap Analysis`/`Free` lane is not used as primary evidence.

## Exact request-window results

All table times are **milliseconds**, rounded to six decimals. Compute/communication are per-rank unions, never sums across ranks. Every request window has **zero compute ∩ communication-envelope overlap**, so compute-only and communication-only equal their respective union columns. This does not prove zero overlap with every auxiliary HCCL kernel or CPU API.

| Capture / rank | Compute union | Communication envelope union | Explicit control union | Control outside compute ∪ comm | Unclassified gap | Compute tasks intersecting window |
|---|---:|---:|---:|---:|---:|---:|
| c1 / TP0 | 92.499580 | 26.079120 | 24.611200 | 0.559940 | 196.418080 | 13,168 |
| c1 / TP1 | 94.123420 | 63.765240 | 68.441320 | 0.407620 | 157.260440 | 13,183 |
| c2 / TP0 | 161.496960 | 89.631500 | 99.037920 | 0.681780 | 201.627490 | 22,641 |
| c2 / TP1 | 160.328520 | 20.709020 | 19.448580 | 0.496240 | 271.903950 | 22,639 |

Partition identity: `span = compute-only + comm-only + overlap + control-outside-both + gap`. Do **not** add the entire control union to compute/communication: much is already overlapped. Small peer task-count differences arise from their different timelines at the fixed request boundary; full-span compute counts agree within each pair.

Per-kind request communication union (ms):

| Capture / rank | AllReduce | AllGather | Broadcast | Communication ms / returned output token |
|---|---:|---:|---:|---:|
| c1 / TP0 | 25.437460 | 0.202780 | 0.438880 | 13.039560 |
| c1 / TP1 | 62.584320 | 0.324220 | 0.856700 | 31.882620 |
| c2 / TP0 | 88.531960 | 0.234320 | 0.865220 | 22.407875 |
| c2 / TP1 | 20.193040 | 0.078840 | 0.437140 | 5.177255 |

The denominator is **batch concurrency × two returned output tokens**: 2 for c1, 4 for c2. It includes prefill and decode work; it is **not isolated decode TPOT**. No collective straddles a request boundary in these four captures.

For comparison only, the **whole device-active first/last spans** (profiling enable/disable tasks excluded) are:

| Capture / rank | Active span ms, not request wall | Compute union ms | Communication union ms | Compute tasks | Collectives |
|---|---:|---:|---:|---:|---:|
| c1 / TP0 | 382.371980 | 101.364840 | 35.664500 | 14,846 | 153 |
| c1 / TP1 | 382.064100 | 103.171360 | 93.884440 | 14,846 | 153 |
| c2 / TP0 | 524.778100 | 171.760400 | 113.922760 | 24,498 | 153 |
| c2 / TP1 | 525.298700 | 171.277020 | 35.082980 | 24,498 | 153 |

Representative **actual** device intervals, in exact epoch microseconds, for c1 TP0 communication PID 3004828132 / group thread 0:

```text
hcom_allReduce__909_0_1:  [1792817202882730.090, 1792817202883180.730)
hcom_allGather__909_49_1: [1792817203099296.670, 1792817203099406.710)
```

JSON includes one example of each collective kind per rank with its original name/args, plus all aggregate and clock values. These are not host `HcclAllReduce` API intervals.

## Actual payloads and the batching hypothesis

These traces **do provide** HCCL envelope `count` and `data_type`. Vendor spelling `BFP16` is BF16 (two bytes); `INT32` is four bytes. The following are **logical per-operation input bytes**, not wire bytes or measured bandwidth. Both TP ranks have the same payload histogram.

| Operation / payload class | Operations per exact request window | c1 input elements → bytes each | c2 input elements → bytes each |
|---|---:|---:|---:|
| Hidden AllReduce, prefill-sized | 49 | 32,768 → 65,536 | 65,536 → 131,072 |
| Hidden AllReduce, decode-sized | 49 | 1,024 → 2,048 | 2,048 → 4,096 |
| AllGather | 2 | 124,160 → 248,320 | 248,320 → 496,640 |
| Broadcast, token-sized | 1 | 1 → 4 | 2 → 8 |
| Broadcast, control-sized | 1 | 6 → 24 | 6 → 24 |

The “prefill/decode-sized” interpretation follows count sizes, model hidden size 1024, and temporal order; no per-event stage annotation is invented. The hidden AllReduce calculation `1024 × T × sizeof(BF16) = 2048*T` agrees with the actual arguments: prefill T=32/64; decode T=1/2. AllGather counts agree with the TP-local half of vocab 248,320 times batch size. No assertion is made about transport expansion, reduction traffic, packet overhead, or a sum of link-level transfer bytes. Plane `size(Byte)` rows are not summed into a fictional wire volume.

**Hypothesis result:**

1. **Launch-count amortization is supported descriptively.** Each request batch has 98 AllReduce, 2 AllGather and 2 Broadcast on each rank at either concurrency. Per returned token that is 51 total collectives at c1 versus 25.5 at c2. Full captures also have identical counts (147/3/3), but their post-request tail must not enter the request denominator.
2. **Decode payload doubling is directly supported.** Actual hidden AllReduce inputs are 2,048 versus 4,096 bytes; prefill-sized AllReduce and AllGather inputs double too. This is not merely a theoretical assumption.
3. **Time savings are not consistent across ranks.** Request compute union grows by 1.746× (TP0) / 1.703× (TP1); compute task counts grow from roughly 13.2k to 22.6k. Communication time per returned token changes by **1.718× on TP0 (worse)** but **0.162× on TP1 (better)**. The rank asymmetry changes direction. These envelopes include peer synchronization effects and have the extent limitations above; this pair cannot identify a general launch-latency benefit or assign a cause to the asymmetry.

There is one warmed profile batch per concurrency, no uncertainty estimate, a very short output window, and visible shared-device activity. Profiling overhead and the post-request tail further limit interpretation. The older benchmark matrix with **three repetitions** is a **separate experiment**; its throughput/latency statistics are not mixed into these profile metrics. No per-rank latency sum is presented as end-to-end latency.

## Reproduction and focused verification

From the repository root, write to a temporary path for review rather than replacing the official summary:

```bash
review_dir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 python homework-2/scripts/analyze_npu_traces.py \
  --output "$review_dir/npu-communication-analysis.json"
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s homework-2/tests -p test_npu_trace_analysis.py -v
```

**18 stdlib tests pass.** Coverage includes interval union/intersection/Decimal clipping, partial envelopes, disjoint partitions, nested/overlapping/duplicate parent rejection, both event-array layouts, host API exclusion, HCCL task/launch dependency association, malformed/name-only device evidence, incorrect PID/rank/physical labels, clock-failure fallback, card/chip evidence, canonical-prefix mismatch and c1/c2 provenance/grid mismatch. Archive tests add raw/gzip metric equivalence and exact selected-file provenance, uncompressed hash/size/event-count mismatches, raw/gzip/mixed logical grids, unknown extra traces, gzip preference without reading raw duplicates, no invalid-archive fallback, truncated/corrupt streams, exact-bound acceptance and bounded inflation (including concatenated members).

The real analyzer completed with the four preferred archives, writing only to temporary output. **All non-provenance fields, including every metric, clock, ownership record, diagnostic, and comparison, exactly match both the pre-change raw-only analyzer output and the existing official summary.** An archive-only temporary input tree (no raw traces) generated byte-identical output to the preferred-archive run. The raw-only versus archive-aware comparison differs in selected-file paths/provenance and analyzer fingerprint, not measured metrics. The currently retained summary already uses the archive-aware schema; the reproduction command writes a separate temporary result and does not overwrite it. The analyzer fails closed on unknown device task types, missing/ambiguous device ownership, unverified HCCL task identities, overlapping parents, or provenance/grid mismatch. An unverifiable clock with otherwise valid evidence yields only clearly labelled active-span metrics.
