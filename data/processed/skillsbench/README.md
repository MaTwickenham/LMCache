# SkillsBench Processed Workload

This directory contains curated file-level fragment datasets derived from
the canonical SkillsBench `tasks/` tree for LMCache fragment-reuse benchmarks.

Generated variants:
- `skillsbench_dense_full`: largest connected component under curated exact-fragment overlap
- `skillsbench_dense_core`: `dense_full` minus edge task `pedestrian-traffic-counting`
- Each query now starts with a short synthetic request brief to separate
  query-specific intent from reusable skill/support fragments

Curated fragment policy:
- Keep `SKILL.md`
- Keep useful `scripts/*` text files (`.py`, `.js`, `.sh`, `.md`, `.txt`) up to 40 KB
- Keep `references/*` text files and common direct references up to 50 KB
- Drop schema blobs, licenses, lockfiles, and other low-value prompt baggage

Dense full summary:
- tasks: 21
- unique fragments: 93
- avg query bytes: 84988.0
- avg shared bytes: 50018.9
- avg prefix bytes: 20687.5

Dense core summary:
- tasks: 20
- unique fragments: 90
- avg query bytes: 86542.4
- avg shared bytes: 51118.2
- avg prefix bytes: 21721.9

Component overview:
- component 0: 21 tasks, 93 unique fragments, pair overlap 0.5286
- component 1: 3 tasks, 9 unique fragments, pair overlap 1.0
- component 2: 2 tasks, 5 unique fragments, pair overlap 1.0
- component 3: 2 tasks, 1 unique fragments, pair overlap 1.0
- component 4: 2 tasks, 6 unique fragments, pair overlap 1.0
- isolated tasks: 57

Primary files:
- `skillsbench_dense_full_manifest.json`
- `skillsbench_dense_full_tasks.json`
- `skillsbench_dense_full_queries.jsonl`
- `skillsbench_dense_full_fragments.jsonl`
- `skillsbench_dense_core_manifest.json`
- `skillsbench_dense_core_tasks.json`
- `skillsbench_dense_core_queries.jsonl`
- `skillsbench_dense_core_fragments.jsonl`
- `skillsbench_component_summary.json`
- `generate_skillsbench_dense.py`
