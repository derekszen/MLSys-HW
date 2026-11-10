# Homework 2 — two-chip tensor parallelism and communication

**Technical work complete for the recorded assignment scope: all four models run TP2 and have passing matched TP1/TP2 configurations.** 4B's last discrepancy was diagnosed as an FP16 final-logit rounding tie and fixed with an explicit FP32 final projection, verified in two fresh launch pairs. See [completion](COMPLETION.md), [requirements](REQUIREMENTS.md), [4B diagnosis/fix](docs/4B-discrepancy.md), [latest evidence](docs/final-validation.md), and the [Chinese report](report.md). Historical failures remain unchanged; nothing has been submitted.

Homework 1's technical baseline and frozen patch are unchanged. The reports record **Zhuoxi Zeng (2501213387)** and **Steven Walxim (2501112130)** with their respective contributions. Steven's new NPU runs reproduce the HW1 prerequisite; his HW2 contribution currently covers offline verification of retained results. The HW2 report also records **陈子睿 (2601112045)** and the supplied contribution to dual-chip adaptation, correctness validation, performance/HBM comparisons and communication timeline analysis; all three member entries are complete. See the [submission guide](../SUBMISSION.md). Nothing has been submitted. Reproduction requires authorized course-server access; personal VPN/keyring configuration is not included in the submission packages.

## Current results

| Model | Matched TP1/TP2 precision and budget | Strict comparison |
| --- | --- | --- |
| 0.8B | BF16, fraction0.12 | All three input cases pass at concurrency1/2; max selected-logprob error0.0576621294 |
| 4B | FP16 weights/activations, **FP32 final projection**, fraction0.25 | All three cases pass in two fresh launch pairs; max error0.0059497356 |
| 9B | FP16, fraction0.45 | All three cases pass; max error0.0058279037 |
| 27B | BF16, fraction0.9 | All three cases pass; max error0.0773688555 |

All use FP32 recurrent state, greedy16 outputs, fixed5/32/256 inputs, no prefix reuse and unchanged `atol=0.15`. The assignment does not specify that exact-token rule or require TP2 speedup; our failed verdicts are nonetheless retained.

- **Execution/sharding:** both workers are logged; same-card topology is recorded. The HCCL Torch2.9 adaptation uses native `all_gather_into_tensor/reduce_scatter_tensor`, with a job-local socket range. [Layer shapes/state/collectives](docs/parallelism-plan.md) distinguish sharding from replication.
- **Performance/HBM:** [new 0.8B matrix](results/final-matrix/benchmark-summary.json) validates all36 measured outputs, with [complete per-chip snapshots](results/final-matrix/hbm-summary.json). TP2 is slower in all four observed cells. [New 27B](results/final-27B/validation-summary.json): input32 **3.638593s /4.397304 output tok/s**, input256 **6.038791s /2.649537 tok/s**; all6 repeats match TP2 correctness. [Model HBM](results/final-model-hbm-summary.json) includes successful27B TP1 **53,603MB post-generation**. These are shared-server descriptive medians/snapshots, not peaks, TTFT/TPOT or independent accuracy certification.
- **Communication experiment:** [both-rank device analysis](docs/communication-results.md) covers input32→2 outputs, batch1→2. **102 collectives/request/rank**, decode AllReduce logical input2048→4096bytes; normalized time worsens on rank0 and improves on rank1. No general speedup claim; overlap-aware interval unions, unclassified gaps not pure wait. Longer vendor exports remain limited.
- **Implementation/tests:** FP16 conv history follows activation dtype; explicit FP16 KV leaves `auto` BF16 and SSM FP32. Optional `--lm-head-compute-dtype float32` preserves stored weights and widens the final projection before rounding. [67 precision/cache tests passed](results/lm-head-precision-tests.txt), [178 local tests passed](results/local-tests-head-fix.txt); existing logits suite50 passed/1 skipped. Existing real-NPU profiler tests10 passed; compatible profile subset29 passed/1 deselected, not full upstream coverage.
- **Packaging/history:** current [15-file patch](patches/manifest.json), **89,738 bytes**, [reconstructs2972 files byte/mode-exact](patches/patch-verification.json). HW1 unchanged. Old failures remain in [historical evidence](docs/run-evidence.md); a fresh [native-head4B control](results/head-fix-regression/4B-native-validation.json) still fails, while the explicit FP32-head pairs pass. No tolerance or token-specific sampling rule was changed.

## Reproduce offline summaries and focused tests

From repository root, with Python and NumPy for the decode-analysis tests (no NPU needed). The tested local dependency is recorded in [requirements-analysis.txt](requirements-analysis.txt); install it only in a local private environment, not the shared course environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s homework-2/tests -v
review_dir=$(mktemp -d)
python homework-2/scripts/summarize_benchmarks.py \
  --results homework-2/results/final-matrix --output "$review_dir/matrix.json"
python homework-2/scripts/analyze_npu_traces.py \
  --output "$review_dir/npu-communication-analysis.json"
```

[Full offline commands](docs/final-validation.md#4-reproduction-and-validation) reproduce27B performance and explicit-directory HBM summaries; [4B commands](docs/4B-discrepancy.md#5-reproduce-and-test) reproduce the diagnosis and fixed comparison. The numerical trace analyzer requires NumPy; the other summaries use the standard library. Validators reject provenance/config mismatches and check every measured output; missing memory remains unknown. Agreement, repeatability and performance are distinct claims.

## Server reproduction templates

Use the course environment and HW1 source. Inspect fresh `npu-smi` and choose a permitted **same-card pair**; adjacency alone does not prove topology. Do not kill existing jobs or bypass memory-balance checks. Keep runs sequential and output directories new; capture complete stdout/stderr including both rank logs.

```bash
# Start at this repository/package root. The retained server checkout was ~/MLSys-HW.
ROOT="$PWD"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd "$ROOT"
export PYTHONPATH="$PWD/homework-1/tokenspeed/python:$PWD/homework-1/tokenspeed/tokenspeed-kernel/python:$PWD/homework-1/tokenspeed/tokenspeed-kernel-npu/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Job-local device-socket range, distinct from rendezvous --port:
export HCCL_NPU_SOCKET_PORT_RANGE=41000-41031
npu-smi info
: "${PAIR:?select permitted same-card physical IDs after fresh inventory (retained retries used 4,5)}"
: "${CHIP:?select one physical ID from that pair}"
PY=homework-1/.venv/bin/python
"$PY" homework-2/scripts/hccl_smoke.py \
  --physical-devices "$PAIR" --port auto --output-dir homework-2/results/hccl-rerun

# 0.8B: TP1/2 × c1/c2; correctness before benchmark, fresh directories.
for TP in 1 2; do
  DEVICES="$CHIP"; [ "$TP" = 1 ] || DEVICES="$PAIR"
  for C in 1 2; do
    for MODE in correctness benchmark; do
      REPEATS=1; [ "$MODE" = correctness ] || REPEATS=3
      timeout --signal=TERM --kill-after=20s 600s "$PY" homework-2/scripts/run_model.py \
        --tp "$TP" --physical-devices "$DEVICES" --model homework-1/models/Qwen3.5-0.8B \
        --dtype bfloat16 \
        --cases homework-2/cases.json --output-dir "homework-2/results/0.8B-${MODE}-tp${TP}-c${C}-rerun" \
        --memory-fraction 0.12 --concurrency "$C" --mode "$MODE" --port 31889 --repeats "$REPEATS" \
        > "homework-2/results/0.8B-${MODE}-tp${TP}-c${C}-rerun.log" 2>&1 || exit $?
    done
  done
done
for C in 1 2; do
  "$PY" homework-2/scripts/compare_tp.py \
    --tp1 "homework-2/results/0.8B-correctness-tp1-c${C}-rerun/result.json" \
    --tp2 "homework-2/results/0.8B-correctness-tp2-c${C}-rerun/result.json" \
    --atol 0.15 --output "homework-2/results/0.8B-comparison-c${C}-rerun.json" || exit $?
done

# Larger model, now with a feasible matching TP1 budget; use fresh output paths.
timeout --signal=TERM --kill-after=20s 600s "$PY" homework-2/scripts/run_model.py \
  --tp 2 --physical-devices "$PAIR" --model homework-2/models/Qwen3.5-27B \
  --dtype bfloat16 \
  --cases homework-2/cases.json --output-dir homework-2/results/27B-benchmark-tp2-c1-rerun \
  --memory-fraction 0.9 --concurrency 1 --mode benchmark --port 31889 --repeats 3 \
  > homework-2/results/27B-benchmark-tp2-c1-rerun.log 2>&1
```

For matched27B correctness, run both `--tp 1 --physical-devices "$CHIP"` and `--tp 2 --physical-devices "$PAIR"` with `--mode correctness --repeats 1`, fraction0.9 and fresh output paths before the benchmark. Latest27B used physical5 and4,5. For4B/9B use their model paths, fractions0.25/0.45 and **`--dtype float16`** (matching FP16 KV). **For4B additionally pass `--lm-head-compute-dtype float32` to BOTH TP1 andTP2**; omit this override for the other recorded configurations. Fractions are allocator budgets, not HBM measurements or reservations. Full4B confirmation launch logs/configs are [retained](results/4B-fp32-head-final-run.txt).

Weight acquisition uses retained `metadata/MODEL/source-manifest.json` with HW1's SHA256-verifying downloader, not a newly resolved mutable revision. Each model is compared to its **own** outputs. Templates create new evidence; summary CLIs require explicit run selection rather than silently picking newest files.

### Incremental patch application

Only on a **separate clean pinned checkout**, not the current adapted submodule or the already-patched source in the submission ZIP:

```bash
# CLEAN must be a separate checkout at 52bdb9763bb2da3d61336398072671885cc0a33e.
: "${CLEAN:?path to separate clean pinned TokenSpeed checkout}"
ROOT="$PWD"
git -C "$CLEAN" apply --check "$ROOT/homework-1/patches/ascend-qwen3.5.patch"
git -C "$CLEAN" apply "$ROOT/homework-1/patches/ascend-qwen3.5.patch"
git -C "$CLEAN" apply --check "$ROOT/homework-2/patches/hccl-torch29.patch"
git -C "$CLEAN" apply "$ROOT/homework-2/patches/hccl-torch29.patch"
```

[Verification](patches/patch-verification.json) checks **2,972 files** byte/mode-exact, retaining executable `logits_processor.py`, `env.py` and `server_args.py`; frozen HW1 remains unchanged. Current HW2 SHA256: `76071724b56e1f7922dd3fd9a4c2c5b359f3422252ac10c74daa5e5d0f2350f6`. Previous3/6/11-file patches are [archived](patches/archive/README.md). Packaging verification is not an additional hardware correctness claim.

## Remaining work

**No unresolved technical blocker remains for the recorded assignment tests.** The [requirements checklist](REQUIREMENTS.md) links each deliverable. Precision choices are per model, not a topology-invariance guarantee; profile windows are short, shared-server timing is descriptive and HBM values are snapshots. The reports and attachments are prepared for the corresponding assignment entries. No report has been submitted.
