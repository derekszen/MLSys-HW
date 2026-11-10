# MLSys-HW

**Confirmed HW1/HW2 contributors: Zhuoxi Zeng (2501213387), Steven Walxim (2501112130), and 陈子睿 (2601112045).** Both group reports record all three members' respective contributions. Report sources and package-building tools are retained; generated PDFs/ZIPs are local outputs, not GitHub downloads. No teaching-platform submission has been performed by this project. Start with the [submission guide](SUBMISSION.md) and [generated-file instructions](submissions/README.md).

- [Homework 1: single-chip Qwen3.5-0.8B adaptation](homework-1/README.md) — single-chip inference, reference checks and the 2×2 performance baseline completed. The report includes a self-contained operator matrix and the required contribution statement.
- [Steven's HW1 reproduction report](homework-1/rebuild/report.md) — fresh single-chip correctness and timing results, with separate attribution for the group's existing implementation and historical memory/profile evidence.
- [Homework 2: TP=2 and communication analysis](homework-2/README.md) — **all four models pass matched TP1/TP2 checks.** The 4B precision fix is confirmed twice. The report includes typical-layer sharding, performance/HBM matrices, larger-model results, overlap-aware timing and the batch experiment. The 15-file incremental patch is verified; frozen HW1 is unchanged.

Original assignment files are preserved under each homework directory. See the [HW1 requirements](homework-1/REQUIREMENTS.md) and [HW2 requirements](homework-2/REQUIREMENTS.md) for explicit evidence mapping. Final student review must confirm all group members, the factual contribution statement, and the platform's upload rules. No grade is guaranteed.

## Prepare or check submission files

For Steven's offline reproduction checks and the independent HW1 evidence checker,
see [offline reproduction](homework-2/docs/offline-reproduction.md). These run on a
Mac or Linux CPU and validate saved evidence; they do not execute NPU inference.

```bash
python scripts/build_submissions.py
(cd submissions && sha256sum -c SHA256SUMS)
```

The builder creates Chinese-font PDFs and ZIPs from separate reconstructed source stages; it does not modify the live TokenSpeed source, re-export the HW1 patch, or contact the teaching platform. Full source and hashed evidence are included; weights, private environments and credentials are not.

## Initialize a Git checkout

The public repository stores the course fork as a pinned submodule and our work as
separate, verified patches. The adapted local submodule can therefore appear dirty;
its changes are preserved in the outer repository's patches, not in an unpublished
submodule commit.

```bash
git clone --recurse-submodules https://github.com/derekszen/MLSys-HW.git
cd MLSys-HW
# On a clean pinned checkout only; do not apply twice.
git -C homework-1/tokenspeed apply --check ../patches/ascend-qwen3.5.patch
git -C homework-1/tokenspeed apply ../patches/ascend-qwen3.5.patch
# Stop here for the frozen HW1 runtime. Apply the incremental patch for HW2:
git -C homework-1/tokenspeed apply --check ../../homework-2/patches/hccl-torch29.patch
git -C homework-1/tokenspeed apply ../../homework-2/patches/hccl-torch29.patch
```

For an existing clone, initialize with `git submodule update --init --recursive`
before applying patches. Apply **both** patches before running the HW2 suite or
submission builder. The builder then reconstructs separate HW1-only and HW2 source
stages without changing the working source.

## Public repository contents

Tracked: reports, original assignments, reproduction scripts/tests, model metadata,
verified patches, retained experimental results and compressed diagnostic traces.
Generated PDFs/ZIPs stay local and can be rebuilt with the command above; they are
not Git-hosted downloads. Model weights, environments, credentials, personal VPN/WoL
notes and duplicate raw captures are excluded by `.gitignore`. Names/student IDs and
course-server runtime metadata in the reports/evidence are intentionally public.

Upstream TokenSpeed and its included source notices remain under their existing
MIT license; see the [pinned upstream license](https://github.com/lightseekorg/tokenspeed/blob/52bdb9763bb2da3d61336398072671885cc0a33e/LICENSE).
