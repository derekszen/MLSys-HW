# Steven HW2 streaming evidence — 2026-11-09

Five bounded TP1 jobs completed on physical chip 1: smoke, correctness c1/c2,
and streaming c1/c2 with three repeats. Hardware verification and raw event JSON
are retained; logs use deterministic lossless gzip. The manifest records both
compressed and original log hashes. No new TP2 measurements were made.

**Supervisor correction:** `_job-supervisor.py` is the exact historical script
used for these runs. Its occupancy regex omitted a table separator and must not
be reused. The defect was caught during the final availability check before any
TP2 launch. `occupancy-validation.json` independently reparses all five saved
pre-run inventories and the final inventory: chip 1 was free in every snapshot;
every same-card pair had another workload. `_job-supervisor-corrected.py` uses
the tested `homework-2/scripts/npu_occupancy.py` parser; the active server copy
was updated. Snapshot checks cannot eliminate a concurrent-allocation race.

`steven-streaming-local-validation.json` in the parent results directory is the
earlier 221-test checkpoint. The later occupancy fix adds five tests, bringing
the current full HW2 suite to 226. The streaming engine code and its 25 tests
were unchanged by this correction. Real measurements remain attributable to
their saved runner/source hashes.
