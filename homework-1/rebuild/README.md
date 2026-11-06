# Steven's HW1 rebuild

This is a separate reproduction/rebuild record. The group's original HW1 report,
patch and saved experiments remain unchanged. HW2 review and a PR follow after
HW1 runs correctly under Steven's account.

## Verified checkpoint — November 6, 2026

- Started from upstream commit `52bdb9763bb2da3d61336398072671885cc0a33e`.
- Created a private environment inheriting the course CANN / PyTorch base.
- Downloaded the pinned Qwen3.5-0.8B checkpoint and verified every manifest hash.
- Ran the unchanged group Transformers reference harness under Steven's account
  on physical chip 1, after confirming no other listed process on that chip.
- Generated 16 tokens for each of 5-, 32- and 256-token inputs. Tokens and selected
  log probabilities exactly match the retained reference for these three cases.
  This is reference reproduction, not validation of our TokenSpeed execution.
- Clean upstream TokenSpeed failed during model initialization because it created
  a CUDA stream on Ascend. A new regression using the real minimal model on the
  meta device fails before the guard and passes after it.
- Reused the group's remaining frozen HW1 Ascend adapters in the rebuild.
  All 1,058 operator and initialization tests passed on physical chip 1 under
  Steven's account (10.81 seconds; six environment warnings).
  [Run evidence](results/2026-11-06/operator-tests/status.json) and
  [test summary](results/2026-11-06/operator-tests/summary.json) are retained.
  The raw pytest log is retained locally and on the server, outside Git.
- Full TokenSpeed generation validation passed for input lengths 5, 32 and 256:
  all 16 greedy output tokens match the fresh Transformers reference exactly.
  Maximum selected-logprob errors were 0.125555, 0.040811 and 0.002223, below
  the recorded 0.15 tolerance. All three request-state isolation checks passed.
  [Validation evidence](results/2026-11-06/engine-validation/validation.json).
- The warm, uncached single-request benchmark completed all 12 runs: input
  lengths 32/256 × output lengths 8/16 × three repeats. Every run checked
  output tokens against the reference and reported zero cached tokens.
  [Benchmark evidence](results/2026-11-06/benchmark/benchmark.json).
- Jobs were bounded, detached and logged on the server. Before each run,
  physical chip 1 had no listed process and the engine port was free. After
  completion, chip 1 had no listed process and port 31887 was free again.

Evidence: [checkpoint](results/2026-11-06/checkpoint.json),
[fresh reference](results/2026-11-06/reference.json),
[provenance](results/2026-11-06/reference-provenance.json),
[reference run status](results/2026-11-06/reference-run-status.json).

## Contribution and source boundaries

The model implementation, original Ascend adapters, and reference harness come
from the existing group project. The stream guard also exists in the frozen
HW1 patch. Steven's AI-assisted work here is fresh setup and execution, checking
reproduction, adding a focused initialization regression, and documenting the
failure/fix. It is not a claim to have independently authored all adapters.

[Initialization patch](patches/01-device-initialization.patch) applies to the
pinned clean upstream tree. To reconstruct the current rebuild, apply it, then
apply the original HW1 patch while excluding
`python/tokenspeed/runtime/models/qwen3_5.py`. No HW2 incremental patch belongs
in this HW1 baseline. The local live submodule still contains the older HW1+HW2
reconstruction; it is not the source used for these new server runs.

## Environment notes

Runtime: Python 3.11.16, torch 2.9.0+cpu plus torch-npu 2.9.0,
Transformers 5.12.0, triton-ascend 3.2.1, NumPy 2.4.6, pytest 8.4.2,
tokenspeed-scheduler 0.1.42. The private overlay matches the recorded runtime.
Shared packages were not removed or replaced.

The retained private package list is not a resolver-compatible lock. Automatic
resolution first conflicts over pytest and then over the generic `triton`
dependency, which the NPU constraints intentionally exclude. The private
packages were installed with `--no-deps` over the course base. Imports and the
reference execution passed, but `pip check` reports vendor-stack missing and
mismatched dependencies. Do not describe this environment as dependency-clean
or install CUDA Triton merely to silence that report. GitHub was unreachable
from the server; a hash-verified Git bundle supplied the exact upstream commit.

## Measured baseline

Median client-observed times in milliseconds; three repeats per row:

| Input tokens | Output tokens | First token | Per subsequent token | Total |
|---:|---:|---:|---:|---:|
| 32 | 8 | 266.38 | 73.71 | 782.89 |
| 32 | 16 | 260.96 | 70.33 | 1315.98 |
| 256 | 8 | 997.49 | 64.79 | 1451.01 |
| 256 | 16 | 1000.40 | 66.91 | 1994.15 |

These timings include scheduling and IPC. They are a reproduced baseline, not
a speedup claim or pure kernel timings. Other chips on the shared server were
in use. Raw logs remain local and on the server; tracked hash manifests cover
the copied evidence. Original group evidence was not overwritten.

## Next

HW1 single-chip execution and the required reproduction matrix are complete.
Return to HW2 verification and prepare the requested PR. Recheck available
chips before each future run; never stop another user's processes. No PR,
merge, or submission has been performed as part of this rebuild.
