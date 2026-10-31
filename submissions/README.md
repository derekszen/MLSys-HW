# Prepared submission files

Student: **Zhuoxi Zeng — 2501213387**.

This directory contains the generated per-homework report PDFs and code/evidence ZIPs. Read [the submission guide](../SUBMISSION.md) for the exact file-to-assignment mapping and remaining student actions. **Nothing has been uploaded.**

- `2501213387_Zhuoxi_Zeng_HW1_Report.pdf`
- `2501213387_Zhuoxi_Zeng_HW1_Submission.zip`
- `2501213387_Zhuoxi_Zeng_HW2_Report.pdf`
- `2501213387_Zhuoxi_Zeng_HW2_Submission.zip`

`manifest.json` records generated sizes, hashes, PDF text checks and ZIP verification. `SHA256SUMS` checks the four deliverable files. PDFs/ZIPs are rebuildable outputs; report sources and build/verification scripts remain in the repository.

```bash
# From repository root:
python scripts/build_submissions.py
(cd submissions && sha256sum -c SHA256SUMS)
```

The source in HW1's ZIP is pinned base + frozen HW1 patch only. HW2's ZIP contains pinned base + HW1 + HW2 patches, plus the HW1 prerequisite evidence. Packages are separate so later TP changes cannot silently contaminate the first homework.
