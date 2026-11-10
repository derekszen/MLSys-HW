# Offline reproduction — Steven Walxim (2501112130)

This contribution adds a standard-library HW1 evidence checker and fixes two
macOS path-comparison failures in the HW2 tests. The existing model adaptations,
frozen patches, hardware observations, and Zhuoxi Zeng's attribution are preserved.

## What the new checker establishes

`verify_hw1_evidence.py` reads the saved Transformers reference, engine validation,
and benchmark JSON. It recomputes exact generated IDs, all 16 selected-logprob
differences, the three required state-isolation cases, and the complete
32/256-input × 8/16-output × three-repeat benchmark grid. It checks uncached
outputs, streaming arrival order, TTFT/TPOT arithmetic, and summary statistics.
Missing observations cannot pass merely because a stored `passed` flag is true.

Eighteen regression tests include missing batch results, truncated reference
logprobs, token/logprob misalignment, nonfinite values, changed configurations,
duplicate benchmark samples, and altered timing summaries. The CLI records input
SHA256 values, returns a nonzero exit status on failure, and refuses to overwrite
an existing output file.

The frozen HW1 records do not include checkpoint/source fingerprints. Matching
paths and settings plus recomputed arithmetic do not independently establish
their provenance, full-logit accuracy, or new hardware performance. These limits
are included in the checker's output.

## Run locally

Use Python 3.10 or later for the full HW2 suite. From the repository root, create
an isolated environment and install `homework-2/requirements-analysis.txt` when
running HW2 analysis. The HW1 checker itself needs only the standard library.

```bash
python homework-2/scripts/verify_hw1_evidence.py
python -m unittest discover -s homework-2/tests -p test_hw1_evidence.py -v
```

For the complete HW2 suite, first initialize the pinned TokenSpeed submodule and
apply the two existing patches in the order shown in the root README. Do not
apply them twice. Retain the compressed retry diagnostic arrays for the optional
real-evidence test; otherwise that test reports a skip.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s homework-2/tests -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s homework-1/tests -p test_profile_summary.py -v
```

The two macOS fixes compare the resolved model/archive paths used by the
application. They retain the same checks on engine configuration and compressed
provenance; they do not weaken numerical acceptance criteria.

The patch test creates a temporary local clone. A partial source clone that also
advertises unrelated, incompletely downloaded branches/tags can fail during that
clone before testing any code. Use a complete source checkout or a checkout
containing only the required pinned revision. This is distinct from a patch
reconstruction failure.

## Recomputed saved results

The [local verification record](../results/steven-offline-review.json) identifies
the source revision, verifier hash, environment, input hashes, and results.
The existing commands in [final validation](final-validation.md#4-reproduction-and-validation)
were used to recompute the 0.8B benchmark matrix, 27B correctness/benchmark results,
and HBM attribution. Explicit pairs were also checked for final-source 0.8B and
9B and the independent 4B FP32-head confirmation. The bounded communication trace
analysis was regenerated with `analyze_npu_traces.py`.

No course-server login, model-weight download, fresh inference, or new performance
measurement was performed in this contribution. The next hardware step is to
reproduce the 0.8B check under account `u2501112130` after obtaining the course
hostname/port and selecting a permitted device.
