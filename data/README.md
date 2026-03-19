# Benchmark Data

This directory is the local data root for LMCache benchmark workloads.

Expected layout:

```text
data/
  processed/
    amem/
    memoryos/
    memos/
    <new_workload>/
```

The benchmark scripts under `benchmarks/blend_ttft` default to this directory.

Notes:
- Dataset contents are intentionally ignored by git.
- Keep only small documentation files in this directory under version control.
- Add new workloads as siblings under `data/processed/`.
