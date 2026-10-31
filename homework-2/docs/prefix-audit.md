# Bounded reference-prefix numerical audit

`../scripts/audit_prefix.py` provides a **separate numerical audit**, not a replacement for strict TP/decode/reference comparisons. Actual server measurements for 4B/9B/27B are complete; old verdicts remain unchanged.

## Retained server results

All comparisons cover **16 identical reference prefixes of `length_32`**, each independently **full-prefilled → 1 generated token**, at unchanged **atol=0.15**. Coverage is the required candidate set, not the full vocabulary or the recurrent decode path.

| Comparison | Verdict | Maximum candidate error | Retained comparison |
| --- | --- | --- | --- |
| 4B TP1 vs TP2 | **Pass** | **0.12479639053344727** | [JSON](../results/4B-prefix-tp-comparison.json) |
| 9B TP1 vs TP2 | **Fail at generated step 8 (index 7)** | **0.181732177734375** | [JSON](../results/9B-prefix-tp-comparison.json) |
| 27B TP2 vs independent Transformers fresh-prefix forward | **Fail at step 6 (index 5)** | **0.18715286254882812** | [JSON](../results/27B-prefix-reference-comparison.json) |

The [access-resume check](../results/access-resume-check.txt) succeeded. Physical 0/1 then became busy/unbalanced: the [first 4B TP2 attempt](../results/4B-prefix-tp2/result.json) failed the memory guard ([log](../results/4B-prefix-audit-run.txt)). [Fresh inventory](../results/prefix-retry-inventory.txt) established **physical 4/5 on card 2** for TP2 retries. No guard was bypassed or unrelated job terminated.

- **4B:** [TP1, physical 0](../results/4B-prefix-tp1/result.json) vs [TP2, physical 4/5](../results/4B-prefix-tp2-card2/result.json), memory fraction 0.25; [retry log](../results/4B-prefix-retry-run.txt). Placement differs and shared occupancy is not controlled.
- **9B:** [TP1, physical 5](../results/9B-prefix-tp1-card2/result.json) vs [TP2, physical 4/5](../results/9B-prefix-tp2-card2/result.json), fraction 0.45; [run log](../results/9B-prefix-audit-run.txt).
- **27B:** [TP2, physical 4/5](../results/27B-prefix-tp2-card2/result.json), fraction 0.8, vs [Transformers fresh-prefix retry, physical 1](../results/27B-prefix-transformers-retry/result.json). The [first reference attempt](../results/27B-prefix-transformers/result.json) OOMed under occupancy; [retry log](../results/27B-prefix-reference-retry-run.txt) records completion when room was available. This is not a successful TokenSpeed TP1 baseline.

The 4B bounded pass **does not reverse its strict greedy failure**. These historical BF16 9B/27B comparisons exceed tolerance even with identical full-prefill histories. **Later results are separate:** [latest validation](final-validation.md) has a passing 9B FP16 direct pair and a passing 27B BF16 direct pair at fraction0.9; none rewrites the failed prefix verdicts here. The later [4B recurrent-decode diagnosis and explicit FP32-head fix](4B-discrepancy.md) now pass its full-generation checks twice. That is separate from this historical fresh-prefix audit; no old verdict or tolerance was changed.

## Reproduce retained comparisons offline

From the repository root, no NPU needed. Output paths must be new; failing comparisons exit nonzero, so retain each verdict separately:

```bash
review_dir=$(mktemp -d)
AUDIT=homework-2/scripts/audit_prefix.py
RESULTS=homework-2/results
PYTHONDONTWRITEBYTECODE=1 python "$AUDIT" compare \
  --left "$RESULTS/4B-prefix-tp1/result.json" \
  --right "$RESULTS/4B-prefix-tp2-card2/result.json" --atol 0.15 \
  --output "$review_dir/4B.json"
PYTHONDONTWRITEBYTECODE=1 python "$AUDIT" compare \
  --left "$RESULTS/9B-prefix-tp1-card2/result.json" \
  --right "$RESULTS/9B-prefix-tp2-card2/result.json" --atol 0.15 \
  --output "$review_dir/9B.json"  # expected exit 1: retained numerical failure
PYTHONDONTWRITEBYTECODE=1 python "$AUDIT" compare \
  --left "$RESULTS/27B-prefix-tp2-card2/result.json" \
  --right "$RESULTS/27B-prefix-transformers-retry/result.json" --atol 0.15 \
  --output "$review_dir/27B.json"  # expected exit 1
```

For a new measurement, inspect `audit_prefix.py plan --help` and `run --help` in the prepared course environment. Use a **new plan/output directory**, fresh same-card inventory and bounded sequential commands; historical physical IDs are not an allocation. Keep all non-TP TokenSpeed configuration identical between compared runs (including port and memory fraction). Transformers requires actual full-model free memory; the TokenSpeed memory fraction does not constrain its allocator. Do not rerun or overwrite failures merely to obtain a pass.

## Meaning and limits

- `plan` reads explicit canonical cases, saved Transformers reference, saved TP2,
  and optional saved TP1. It selects **only `length_32`, all 16 positions**. Prefix
  `i` is the exact canonical input IDs plus reference output IDs before `i`.
  Each saved trajectory's first divergence from the reference is recorded; its
  common prefix is already among these 16, so no duplicate request is needed.
  Other canonical cases and alternate-history prefixes are outside this audit.
- Required candidates are reference top5 plus the reference target and TP1/TP2
  output tokens **only when their entire preceding output history matches the
  reference**. A later matching token does not restore history eligibility.
- `run --backend tokenspeed` makes one request per prefix, greedy,
  `max_new_tokens=1`, `return_logprob=True`, explicit top10 or top20. It reads
  `meta_info.output_top_logprobs[0]` entries `[logprob, tokenID, None]`.
  **No `token_ids_logprob` calls, recurrent decode replay, or model changes.**
  Prefix caching is disabled, and metadata must show zero cached tokens.
- `run --backend transformers` loads the unmodified
  `Qwen3_5ForConditionalGeneration` locally in BF16/eager on one NPU. Each prefix
  gets a fresh `model(input_ids=..., use_cache=False)` forward, with float32
  last-position `log_softmax`. It stores topK **and every candidate's direct
  logprob**, including candidates outside topK. No quantization/offload/patching.
  Full-model single-device memory must fit; `--memory-fraction` is validated but
  applies only to TokenSpeed, not to the Transformers allocator.
- `compare` requires identical plan/model fingerprints, reference prefixes,
  candidate IDs, and shared unmodified BF16/eager precision. All candidate values
  must be present and finite. Passing means the **maximum absolute logprob error
  over every required candidate and all 16 prefixes is at most `--atol`**, with
  `0 <= atol <= 0.15`. Argmax equality, exact top ties, and top1/top2 margins are
  separate diagnostics, not pass criteria. A pass proves neither the same
  recurrent decode path nor full-vocabulary equality, and cannot reverse a
  previous strict failure.

A completed measurement is `audit_completed_not_compared`, **not a pass**.
Missing TokenSpeed candidates produce `audit_coverage_failed` with explicit
missing IDs and no fabricated values (exit 1). Invalid payloads/runtime errors
produce `failed`; comparisons reject either. Try explicit `--top-k 20` in a new
output directory if top10 coverage fails; top20 still cannot guarantee coverage.

## Provenance and comparison rules

Plans retain exact source paths/SHA256s, expected resolved model path, config and
manifest fingerprints, reference history/top5, trajectory IDs, and planner/helper
script hashes. Runs re-read and hash all source files and reconstruct the plan
before loading a model. They verify local `config.json` against
`source-manifest.json`, then verify both hashes against the plan; invalid IDs and
contexts exceeding 512 fail closed. The plan must be made on the audited host:
absolute source/model paths are intentional, not silently remapped.

**The old Transformers reference lacks config/manifest fingerprints.** Expected
model fingerprints come from saved TP2, checked against TP1 when supplied. This
cannot establish historical reference checkpoint identity. Manifest/config hashes
also do not re-hash every weight shard; they bind metadata, not independently
attest all current weight bytes. Hashes are integrity links, not signatures.

Results embed the plan and retain its canonical content hash and exact file-byte
hash, source hashes, planner/runner/helper hashes, runtime versions, HBM snapshots,
and runner/recursive-child process evidence. TokenSpeed also records fingerprints
of its three Python source trees. Existing helpers may report unavailable process
metadata; inspect these fields rather than inferring rank ownership from names.

TokenSpeed-vs-TokenSpeed comparisons allow **only the five TP-degree config
fields** to differ; all other config values (including memory fraction and port),
source fingerprints, and runtime versions must match. Run sequentially with the
same port. Physical placement is recorded separately. TopK may differ provided
all required candidates are covered. Transformers comparisons validate each
backend's own config and the shared BF16/unmodified/eager subset; **TP2 is not
required** for the other measurement. Backend-specific recurrent/cache/state
implementations are not asserted identical.

## CPU verification

```bash
python -m unittest discover -s homework-2/tests -p test_prefix_audit.py -v
```

Tests parse the actual saved 4B/9B/27B schemas (including the failed optional 27B
TP1), exercise the CLI using temporary files, and test history/provenance/ID
validation, manifest integrity, topK coverage, NaN/Inf/missing candidates,
non-argmax errors, TP-only config differences, Transformers config subsets, and
tie diagnostics. Numerical fixtures are pure data, not mocked accelerator
choreography and not evidence of actual TokenSpeed/Transformers agreement.
