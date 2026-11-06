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
- Reused the group's remaining frozen HW1 Ascend adapters in the rebuild. Their
  new operator test run has not started: permission-review timeout followed by
  loss of VPN connectivity blocked the launch. Full generation remains pending.

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

## Next

1. Re-establish the VPN link and check for an existing test job before launching.
2. Recheck device availability; never stop another user's processes.
3. Run the five Ascend operator test files plus the new initialization test.
4. Run TokenSpeed validation against the fresh reference, including state
   isolation, then the required benchmark matrix into new output paths.
5. Preserve new logs and source checkpoints, then return to HW2 and its PR.
