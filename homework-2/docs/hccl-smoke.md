# Bounded two-chip HCCL smoke test

**Actual two-rank raw HCCL smoke passed:** FP32/BF16 AllReduce, AllGather and
ReduceScatter on physical 0/1 ([result](../results/hccl-port-range/result.json)).
[Server inventory](../results/server-inventory.txt) establishes these as chip 0/1
of card 0; adjacency alone is not the evidence. VPN access is restored, but its
session lifetime is not guaranteed. Local helper tests remain separate from NPU
execution evidence.

## Recorded failures and resolution

The [initial run](../results/hccl-first/rank-0.json) failed with `EI0020`: HCCL's
default device-socket port `16666` was already bound. The retest passed both ranks
after setting **job-local** `HCCL_NPU_SOCKET_PORT_RANGE=41000-41031`; no shared
configuration or other jobs were changed. HCCL's range is separate from the
localhost rendezvous `--port`, so changing only rendezvous did not fix this failure.

The raw smoke uses list-based APIs and did not validate the project's tensor
collective wrappers. Torch 2.9 lacks `all_gather_single` / `reduce_scatter_single`;
the private project backend now uses `all_gather_into_tensor` /
`reduce_scatter_tensor`. Its separate actual **two-rank regression passed (1 test)**
([log](../results/hccl-backend-tests.txt)); the shared environment and HW1 frozen
patch were not changed. The separate incremental HW2 patch remains pending.

The subsequent actual 0.8B TP=1/2 comparison at concurrency 1 also passed for
inputs 5/32/256 and output 16, with exact greedy tokens and maximum selected-logprob
error 0.0576621294 ≤0.15 ([comparison](../results/0.8B-tp-comparison-c1.json)).
That is not full-logit equivalence or a substitute for larger-model runs,
benchmarking or device communication profiling, all still pending.

## Preconditions: establish the physical pair first

Use the existing Python 3.11 course environment with Torch 2.9, torch_npu 2.9,
and CANN 9 on the Ascend server. This script installs nothing, downloads no
models, uses no credentials, and initiates no SSH connections. It lives outside
the HW1 course-fork submodule and does not modify that checkout.

Before executing on the server:

1. Obtain permission/reservation for **two chips of the same physical 910C card**.
   Inspect current occupancy with `npu-smi info`; visibility and free memory are
   **not reservation**, isolation, or evidence of an uncontended run.
2. Use the server's actual `npu-smi` inventory/topology/card-to-chip information
   (and administrator confirmation if ambiguous) to establish the physical device
   IDs and their common card. Save that output alongside the run. Output formats
   vary by driver; this script deliberately does not guess a mapping from them.
3. Check that the physical IDs expected by `ASCEND_RT_VISIBLE_DEVICES` match that
   inventory. **Adjacent IDs such as `0,1` do not prove same-card membership.**
   The example below uses the recorded pair, not a future reservation.

The script records requested physical IDs and the resulting logical device
indices, not an independently queried physical-card identity. Its summary always
contains `same_card_verified: false`; external topology evidence is still required
before claiming the homework's same-card requirement. For the recorded successful
run, the linked external inventory supplies that evidence despite the script's
`same_card_verified: false`. A successful collective between chips on different
cards would not satisfy that requirement.

## Usage

From the repository root, after independently verifying and reserving the pair:

```bash
# Recorded pair; recheck topology/permission. Use a NEW output directory.
HCCL_NPU_SOCKET_PORT_RANGE=41000-41031 \
python3.11 -B homework-2/scripts/hccl_smoke.py \
  --physical-devices 0,1 \
  --port auto \
  --dist-timeout 30 \
  --run-timeout 120 \
  --output-dir /tmp/hw2-hccl-smoke-run-001
```

Alternatively, use `--port 29517` to require that exact localhost TCP port. The
port option is mandatory: `auto` probes an ephemeral free port; a numeric port
must be in `1..65535` and free at the initial probe. Rendezvous uses
`tcp://127.0.0.1:<port>` explicitly, not inherited `MASTER_ADDR`/`MASTER_PORT`.
There is an unavoidable close-before-TCPStore-bind race after the probe. A port
collision fails the run; the script does not silently switch ports or connect to
remote rendezvous hosts. HCCL's own local-device communication is managed by the
installed runtime; loopback rendezvous is not a claim that all HCCL traffic uses TCP
loopback.

`ASCEND_RT_VISIBLE_DEVICES` is set **before any Torch/torch_npu imports** in the
parent and both spawned children. The pair is kept in CLI order: for `7,3`, rank 0
selects logical `npu:0` (requested physical 7), and rank 1 selects logical `npu:1`
(requested physical 3). Exactly two visible NPUs are required. The variable limits
visibility; it does not reserve devices or stop other jobs.

The output directory must not already exist, preventing stale rank files from
turning a failed attempt into a pass. Do not pre-create it to save inventory;
save inventory separately or copy it into the directory after the run.

## What is checked

`torch.multiprocessing.spawn` launches exactly two children, with logical device
set to rank and a real `hccl` process group of world size 2. Both ranks perform
these operations for **each of float32 and bfloat16**, in the same order:

| Collective | Rank-dependent input | Exact output on rank r |
| --- | --- | --- |
| SUM `all_reduce` | `[4*r+i for i in 0..3]` | `[4, 6, 8, 10]` |
| `all_gather` | `[4*r+i for i in 0..3]` | `[[0,1,2,3], [4,5,6,7]]`, ordered by source rank |
| SUM `reduce_scatter` | destination d gets `[8*r+4*d+i for i in 0..3]` | rank 0: `[8,10,12,14]`; rank 1: `[16,18,20,22]` |

The gather and scatter use the actual list-based distributed APIs; no emulation,
CPU/Gloo fallback, alternate dtype, or unsupported-operation skip is accepted.
All values and sums are exactly representable in BF16. Each operation is followed
by NPU synchronization and an exact CPU-side `torch.equal` check, with no numerical
tolerance. Actual and expected arrays are recorded. A final synchronization
precedes memory sampling. Allocated/reserved and peak allocated/reserved bytes are
Torch's per-rank allocator metrics, not total device occupancy; current allocation
can be zero once these tiny temporary tensors are released.

Each `rank-0.json` / `rank-1.json` records rank, world size, PID, requested physical
ID, logical device, visible-device list, actual backend, device name, Torch and
torch_npu versions, six checks, and allocator memory metrics. A fresh run ID ties
both files to the parent. Failures that reach Python also record a traceback;
a killed/hung child may leave no rank file.

`result.json` and the final stdout JSON say `passed: true` **only if** both children
exit successfully, both rank files exist for this run, ranks/world size/backend/
requested mappings are correct, PIDs are distinct, memory metrics are valid, and
all six checks on each rank contain the exact independently recomputed expected
outputs. Group cleanup errors also fail the result. Exit code is 0 only for such
a pass; missing Torch, missing NPUs, unsupported BF16/API behavior, bad results,
timeouts, and import/runtime errors are failures, never successful skips. Parse
errors return argparse's nonzero exit status; failure to create a new output
directory returns nonzero without overwriting anything.

## Bounds and cleanup

The process-group timeout defaults to **30 seconds**. The parent separately
bounds the spawned-worker phase, including device setup, rendezvous, collectives,
synchronization and group teardown, to **120 seconds** by default. Both timeout
flags accept integer seconds in `1..1800`, and the run timeout must be at least
the distributed timeout plus 10 seconds. The parent deadline starts just before
`spawn`; parent Torch import and local filesystem operations are outside it.
This is not a hard real-time guarantee against OS/driver failure.

Each child destroys only its own successfully initialized process group in
`finally`. On failure, interrupt, or deadline expiry the parent terminates only
its own spawned `Process` objects, waits up to five seconds total, then kills any
survivors and waits up to five more seconds total. It never kills unrelated
processes, searches by process name, sends process-group-wide signals, or resets
NPUs. Forced termination cannot guarantee execution of a child's Python `finally`
block; it is the bounded fallback if HCCL or group teardown hangs. An externally
killed parent or uninterruptible driver/OS state may require operator attention.

This is a small correctness smoke test, **not** model tensor parallelism,
continuous inference, a throughput/latency benchmark, an overlap study, or proof
of same-card topology/reservation. It cannot establish any of those HW2 results.

## Local standard-library verification

```bash
python3.11 -B -m unittest discover \
  -s homework-2/tests -p 'test_hccl_helpers.py' -v
python3.11 -B homework-2/scripts/hccl_smoke.py --help
```

Tests explicitly block Torch imports when loading the helper module. They cover
physical-ID and CLI validation, loopback port-selection behavior with mocked
sockets, independent expected constants, strict two-file/result aggregation,
missing/corrupt/stale/incorrect results, deadline and owned-child cleanup logic,
output-directory reuse refusal, failure (not pass/skip) when Torch is absent,
and mocked unavailable/wrong-count NPUs. They use synthetic rank JSON solely to
test aggregation, not to claim collective execution. The linked server results
provide actual HCCL and rank/device evidence for the recorded configuration;
synthetic helper tests do not validate driver behavior or every failure/teardown
path. Future runs still require fresh topology/occupancy checks and bounded execution.
