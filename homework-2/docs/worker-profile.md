# Owned-worker dynamic profiling — failed historical route

**The PID-scoped `msprof` attachment attempt failed. Successful bounded two-rank captures now use the [in-process explicit NPU adapter](npu-profile-run.md), not this route.** See [actual communication results](communication-results.md) for the validated short-window analysis.

## Retained attempt and ownership boundary

[`profile_workers.py`](../scripts/profile_workers.py) warmed the 0.8B TP2 engine, selected exactly two recursive child schedulers by PID/PPID/name/creation time, and attempted separate collectors of this form:

```text
msprof --dynamic=on --pid=<ownedpid> --duration=<8..20> --output=<newdir> --task-time=on --hccl=on --runtime-api=off --ai-core=off --storage-limit=200MB --export=off
```

The [actual run log](../results/0.8B-dynamic-profile-c1-run.txt) retains failure: the vendor rejected `--pid` with no valid PID even for owned workers; a collector exited **255 before the workload completed**. Advertised CLI options and owned PIDs did not establish supported dynamic capture. No usable two-rank device timeline resulted from this attempt.

The harness has **no whole-application, global/system-wide, device-wide or alternate-profiler fallback**. Collector cleanup is limited to verified owned `Popen` process groups; it does not signal scheduler PIDs, kill by name, or terminate unrelated jobs. Fresh inventory and authorized same-card selection remain mandatory; visibility is not reservation.

## Superseding evidence and remaining limits

Explicit `activities=["NPU"]` through the running workers' profile API now captures real compute/HCCL events on **physical 4/5, card 2**, input **32 → 2 output tokens**, both ranks at c1/c2. That evidence supersedes the old missing-device status **only within this bounded scope**. Long 16-token and 256-input exports still fail the vendor parser; neither successful start nor export exit code alone proves usable device events.

The original whole-application `msprof` timeout/host-only rank0 artifacts and this failed dynamic attempt remain historical evidence, not current “no profile yet” conclusions. The short-window analysis does not certify full-model numerics or benchmark latency. The [latest validation](final-validation.md), including the [4B fix](4B-discrepancy.md), now closes the numerical blockers within the recorded scope; another profiling attempt is not a missing primary deliverable.

## Focused offline tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s homework-2/tests -p test_worker_profile.py -v
```

These tests cover record selection, ambiguity rejection, PID-scoped command construction, output validation and metadata-only filesystem gates. They do not prove vendor attachment support or simulate accelerator/collector lifecycle success. The current full local suite's **178 passes** are recorded separately in [local-tests-head-fix.txt](../results/local-tests-head-fix.txt).
