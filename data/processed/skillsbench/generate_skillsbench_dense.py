#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path(
    "/home/mahaoran/research/compoundai/CacheBlend/skillsbench/tasks"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
SCRIPT_SUFFIXES = {".js", ".md", ".py", ".sh", ".txt"}
REFERENCE_SUFFIXES = {".csv", ".json", ".md", ".toml", ".txt", ".yaml", ".yml"}
DIRECT_REFERENCE_FILES = {"reference.md", "forms.md", "html2pptx.md", "ooxml.md", "docx-js.md"}
CORE_EXCLUDED_TASKS = {"pedestrian-traffic-counting"}


@dataclass(frozen=True)
class SourceFile:
    task_id: str
    skill_name: str
    relative_path: str
    source_path: str
    fragment_id: str
    sha256: str
    kind: str
    size_bytes: int
    size_chars: int
    line_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate curated SkillsBench dense-workload artifacts for LMCache "
            "fragment-reuse benchmarks."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--variant",
        choices=("all", "dense_full", "dense_core"),
        default="all",
        help="Generate one variant or all available variants.",
    )
    return parser.parse_args()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def stable_jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            handle.write("\n")


def include_skill_file(file_path: Path, skill_root: Path) -> bool:
    rel = file_path.relative_to(skill_root).as_posix()
    name = file_path.name
    suffix = file_path.suffix.lower()

    if name == "SKILL.md":
        return True
    if rel.startswith("scripts/"):
        if "schemas/" in rel:
            return False
        return suffix in SCRIPT_SUFFIXES and file_path.stat().st_size <= 40_000
    if rel.startswith("references/") or rel in DIRECT_REFERENCE_FILES:
        return suffix in REFERENCE_SUFFIXES and file_path.stat().st_size <= 50_000
    return False


def classify_kind(file_path: Path, skill_root: Path) -> str:
    rel = file_path.relative_to(skill_root).as_posix()
    if file_path.name == "SKILL.md":
        return "skill_md"
    if rel.startswith("scripts/"):
        return "script"
    return "reference"


def file_order_key(file_path: Path, skill_root: Path) -> tuple[int, str]:
    rel = file_path.relative_to(skill_root).as_posix()
    if file_path.name == "SKILL.md":
        return (0, rel)
    if rel.startswith("references/") or rel in DIRECT_REFERENCE_FILES:
        return (1, rel)
    if rel.startswith("scripts/"):
        return (2, rel)
    return (3, rel)


def read_text_and_stats(path: Path) -> tuple[str, int, int]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    return text, len(raw), text.count("\n") + (1 if text else 0)


def load_task_metadata(task_dir: Path) -> dict[str, Any]:
    task_toml = task_dir / "task.toml"
    metadata: dict[str, Any] = {}
    if task_toml.exists():
        loaded = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        metadata = loaded.get("metadata") or {}
    return metadata


def build_task_inventory(source_root: Path) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[SourceFile]],
]:
    tasks: dict[str, dict[str, Any]] = {}
    fragments: dict[str, dict[str, Any]] = {}
    task_files: dict[str, list[SourceFile]] = defaultdict(list)

    for task_dir in sorted([path for path in source_root.iterdir() if path.is_dir()]):
        task_id = task_dir.name
        metadata = load_task_metadata(task_dir)
        instruction_path = task_dir / "instruction.md"
        instruction = instruction_path.read_text(encoding="utf-8").strip()

        tasks[task_id] = {
            "task_id": task_id,
            "source_task_dir": str(task_dir),
            "instruction_path": str(instruction_path),
            "instruction": instruction,
            "instruction_chars": len(instruction),
            "instruction_bytes": len(instruction.encode("utf-8")),
            "category": metadata.get("category"),
            "difficulty": metadata.get("difficulty"),
            "tags": list(metadata.get("tags") or []),
        }

        skills_root = task_dir / "environment" / "skills"
        if not skills_root.exists():
            continue

        for skill_dir in sorted([path for path in skills_root.iterdir() if path.is_dir()]):
            candidate_files = [
                path
                for path in skill_dir.rglob("*")
                if path.is_file() and include_skill_file(path, skill_dir)
            ]
            for file_path in sorted(candidate_files, key=lambda path: file_order_key(path, skill_dir)):
                text, size_bytes, line_count = read_text_and_stats(file_path)
                sha256 = sha256_hex(text.encode("utf-8"))
                fragment_id = f"skillsbench:frag:{sha256[:16]}"
                kind = classify_kind(file_path, skill_dir)
                source_file = SourceFile(
                    task_id=task_id,
                    skill_name=skill_dir.name,
                    relative_path=file_path.relative_to(skill_dir).as_posix(),
                    source_path=str(file_path),
                    fragment_id=fragment_id,
                    sha256=sha256,
                    kind=kind,
                    size_bytes=size_bytes,
                    size_chars=len(text),
                    line_count=line_count,
                )
                task_files[task_id].append(source_file)

                if fragment_id not in fragments:
                    fragments[fragment_id] = {
                        "fragment_id": fragment_id,
                        "sha256": sha256,
                        "text": text,
                        "kind": kind,
                        "size_bytes": size_bytes,
                        "size_chars": len(text),
                        "line_count": line_count,
                        "source_files": [],
                        "source_tasks": set(),
                        "skill_names": set(),
                    }

                fragments[fragment_id]["source_files"].append(
                    {
                        "task_id": task_id,
                        "skill_name": skill_dir.name,
                        "relative_path": source_file.relative_path,
                        "source_path": source_file.source_path,
                    }
                )
                fragments[fragment_id]["source_tasks"].add(task_id)
                fragments[fragment_id]["skill_names"].add(skill_dir.name)

    for fragment in fragments.values():
        fragment["source_tasks"] = sorted(fragment["source_tasks"])
        fragment["skill_names"] = sorted(fragment["skill_names"])
        fragment["task_count"] = len(fragment["source_tasks"])
        fragment["source_file_count"] = len(fragment["source_files"])

    return tasks, fragments, task_files


def ordered_unique_fragment_ids(files: list[SourceFile]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for source_file in files:
        if source_file.fragment_id in seen:
            continue
        seen.add(source_file.fragment_id)
        ordered.append(source_file.fragment_id)
    return ordered


def build_overlap_graph(task_files: dict[str, list[SourceFile]]) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, list[str]]],
]:
    task_to_fragments = {
        task_id: ordered_unique_fragment_ids(files)
        for task_id, files in task_files.items()
    }
    fragment_to_tasks: dict[str, set[str]] = defaultdict(set)
    for task_id, fragment_ids in task_to_fragments.items():
        for fragment_id in fragment_ids:
            fragment_to_tasks[fragment_id].add(task_id)

    adjacency: dict[str, set[str]] = defaultdict(set)
    shared_fragments: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for fragment_id, task_ids in fragment_to_tasks.items():
        task_list = sorted(task_ids)
        if len(task_list) < 2:
            continue
        for index, left in enumerate(task_list):
            for right in task_list[index + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
                shared_fragments[left][right].append(fragment_id)
                shared_fragments[right][left].append(fragment_id)

    return adjacency, shared_fragments


def connected_components(adjacency: dict[str, set[str]], all_tasks: list[str]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for task_id in all_tasks:
        if task_id in seen or task_id not in adjacency:
            continue
        stack = [task_id]
        component: list[str] = []
        seen.add(task_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda comp: (-len(comp), comp))


def summarize_component(
    component_tasks: list[str],
    tasks: dict[str, dict[str, Any]],
    task_files: dict[str, list[SourceFile]],
    fragments: dict[str, dict[str, Any]],
    shared_fragments: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    component_set = set(component_tasks)
    query_fragment_ids = {
        task_id: ordered_unique_fragment_ids(task_files[task_id])
        for task_id in component_tasks
    }
    unique_fragment_ids = sorted(
        {fragment_id for fragment_ids in query_fragment_ids.values() for fragment_id in fragment_ids}
    )

    query_sizes = [
        sum(fragments[fragment_id]["size_bytes"] for fragment_id in fragment_ids)
        for fragment_ids in query_fragment_ids.values()
    ]
    overlap_pairs = 0
    total_pairs = 0
    for index, left in enumerate(component_tasks):
        for right in component_tasks[index + 1 :]:
            total_pairs += 1
            if shared_fragments.get(left, {}).get(right):
                overlap_pairs += 1

    shared_fragment_ids = [
        fragment_id
        for fragment_id in unique_fragment_ids
        if len(component_set.intersection(fragments[fragment_id]["source_tasks"])) > 1
    ]
    shared_bytes_by_task = []
    prefix_bytes_by_task = []
    for task_id in component_tasks:
        fragment_ids = query_fragment_ids[task_id]
        shared_ids = [
            fragment_id
            for fragment_id in fragment_ids
            if len(component_set.intersection(fragments[fragment_id]["source_tasks"])) > 1
        ]
        shared_bytes_by_task.append(
            sum(fragments[fragment_id]["size_bytes"] for fragment_id in shared_ids)
        )

        best_prefix_bytes = 0
        for other_task_id in component_tasks:
            if other_task_id == task_id:
                continue
            other_fragment_ids = query_fragment_ids[other_task_id]
            prefix_bytes = 0
            for left_fragment_id, right_fragment_id in zip(fragment_ids, other_fragment_ids):
                if left_fragment_id != right_fragment_id:
                    break
                prefix_bytes += fragments[left_fragment_id]["size_bytes"]
            best_prefix_bytes = max(best_prefix_bytes, prefix_bytes)
        prefix_bytes_by_task.append(best_prefix_bytes)

    return {
        "task_count": len(component_tasks),
        "tasks": component_tasks,
        "unique_fragment_count": len(unique_fragment_ids),
        "shared_fragment_count": len(shared_fragment_ids),
        "unique_fragment_bytes": sum(
            fragments[fragment_id]["size_bytes"] for fragment_id in unique_fragment_ids
        ),
        "avg_fragments_per_query": round(
            statistics.mean(len(fragment_ids) for fragment_ids in query_fragment_ids.values()), 2
        ),
        "avg_query_bytes": round(statistics.mean(query_sizes), 1) if query_sizes else 0.0,
        "avg_shared_bytes": round(statistics.mean(shared_bytes_by_task), 1)
        if shared_bytes_by_task
        else 0.0,
        "avg_prefix_bytes": round(statistics.mean(prefix_bytes_by_task), 1)
        if prefix_bytes_by_task
        else 0.0,
        "avg_blend_only_bytes": round(
            statistics.mean(
                max(0, shared - prefix)
                for shared, prefix in zip(shared_bytes_by_task, prefix_bytes_by_task)
            ),
            1,
        )
        if shared_bytes_by_task
        else 0.0,
        "pair_overlap_ratio": round(overlap_pairs / total_pairs, 4) if total_pairs else 0.0,
        "categories": dict(
            Counter(tasks[task_id].get("category") or "unknown" for task_id in component_tasks)
        ),
        "difficulties": dict(
            Counter(tasks[task_id].get("difficulty") or "unknown" for task_id in component_tasks)
        ),
    }


def build_query_rows(
    *,
    variant_name: str,
    task_ids: list[str],
    tasks: dict[str, dict[str, Any]],
    task_files: dict[str, list[SourceFile]],
    fragments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        fragment_ids = ordered_unique_fragment_ids(task_files[task_id])
        per_fragment = []
        query_bytes = 0
        for position, fragment_id in enumerate(fragment_ids):
            fragment = fragments[fragment_id]
            source_match = next(
                source_file for source_file in task_files[task_id] if source_file.fragment_id == fragment_id
            )
            query_bytes += fragment["size_bytes"]
            per_fragment.append(
                {
                    "position": position,
                    "fragment_id": fragment_id,
                    "kind": fragment["kind"],
                    "skill_name": source_match.skill_name,
                    "relative_path": source_match.relative_path,
                    "size_bytes": fragment["size_bytes"],
                    "size_chars": fragment["size_chars"],
                    "task_frequency": fragment["task_count"],
                }
            )

        task_meta = tasks[task_id]
        rows.append(
            {
                "query_id": f"{variant_name}:{task_id}",
                "dataset": variant_name,
                "task_id": task_id,
                "question": task_meta["instruction"],
                "category": task_meta["category"],
                "difficulty": task_meta["difficulty"],
                "tags": task_meta["tags"],
                "source_task_dir": task_meta["source_task_dir"],
                "instruction_path": task_meta["instruction_path"],
                "instruction_bytes": task_meta["instruction_bytes"],
                "fragment_ids": fragment_ids,
                "fragment_count": len(fragment_ids),
                "fragment_bytes": query_bytes,
                "fragments": per_fragment,
            }
        )
    return rows


def build_fragment_rows(
    *,
    task_ids: list[str],
    fragments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    task_id_set = set(task_ids)
    rows: list[dict[str, Any]] = []
    for fragment_id in sorted(fragments):
        fragment = fragments[fragment_id]
        surviving_files = [
            source_file
            for source_file in fragment["source_files"]
            if source_file["task_id"] in task_id_set
        ]
        surviving_tasks = sorted({source_file["task_id"] for source_file in surviving_files})
        if not surviving_tasks:
            continue
        rows.append(
            {
                "fragment_id": fragment["fragment_id"],
                "sha256": fragment["sha256"],
                "kind": fragment["kind"],
                "size_bytes": fragment["size_bytes"],
                "size_chars": fragment["size_chars"],
                "line_count": fragment["line_count"],
                "task_count": len(surviving_tasks),
                "source_file_count": len(surviving_files),
                "source_tasks": surviving_tasks,
                "skill_names": sorted({source_file["skill_name"] for source_file in surviving_files}),
                "source_files": surviving_files,
                "text": fragment["text"],
            }
        )
    return rows


def component_summary_rows(
    components: list[list[str]],
    tasks: dict[str, dict[str, Any]],
    task_files: dict[str, list[SourceFile]],
    fragments: dict[str, dict[str, Any]],
    shared_fragments: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_tasks = sorted(tasks)
    isolated = [task_id for task_id in all_tasks if task_id not in {t for comp in components for t in comp}]
    for index, component_tasks in enumerate(components):
        summary = summarize_component(
            component_tasks=component_tasks,
            tasks=tasks,
            task_files=task_files,
            fragments=fragments,
            shared_fragments=shared_fragments,
        )
        summary["component_id"] = index
        rows.append(summary)
    rows.append(
        {
            "component_id": "isolated",
            "task_count": len(isolated),
            "tasks": isolated,
        }
    )
    return rows


def write_variant(
    *,
    output_root: Path,
    variant_name: str,
    description: str,
    task_ids: list[str],
    tasks: dict[str, dict[str, Any]],
    task_files: dict[str, list[SourceFile]],
    fragments: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    source_root: Path,
) -> None:
    query_rows = build_query_rows(
        variant_name=variant_name,
        task_ids=task_ids,
        tasks=tasks,
        task_files=task_files,
        fragments=fragments,
    )
    query_index = {row["task_id"]: row for row in query_rows}
    fragment_rows = build_fragment_rows(task_ids=task_ids, fragments=fragments)
    task_rows = [
        {
            **tasks[task_id],
            "query_id": f"{variant_name}:{task_id}",
            "fragment_ids": query_index[task_id]["fragment_ids"],
            "fragment_count": query_index[task_id]["fragment_count"],
            "fragment_bytes": query_index[task_id]["fragment_bytes"],
        }
        for task_id in task_ids
    ]

    manifest = {
        "dataset_name": variant_name,
        "description": description,
        "source_root": str(source_root),
        "task_count": len(task_ids),
        "files": {
            "manifest": f"{variant_name}_manifest.json",
            "tasks": f"{variant_name}_tasks.json",
            "queries": f"{variant_name}_queries.jsonl",
            "fragments": f"{variant_name}_fragments.jsonl",
        },
        "summary": summary,
    }

    stable_json_dump(output_root / f"{variant_name}_manifest.json", manifest)
    stable_json_dump(output_root / f"{variant_name}_tasks.json", task_rows)
    stable_jsonl_dump(output_root / f"{variant_name}_queries.jsonl", query_rows)
    stable_jsonl_dump(output_root / f"{variant_name}_fragments.jsonl", fragment_rows)


def generate_readme(
    *,
    output_root: Path,
    dense_full_summary: dict[str, Any],
    dense_core_summary: dict[str, Any],
    component_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# SkillsBench Processed Workload",
        "",
        "This directory contains curated file-level fragment datasets derived from",
        "the canonical SkillsBench `tasks/` tree for LMCache fragment-reuse benchmarks.",
        "",
        "Generated variants:",
        "- `skillsbench_dense_full`: largest connected component under curated exact-fragment overlap",
        "- `skillsbench_dense_core`: `dense_full` minus edge task `pedestrian-traffic-counting`",
        "",
        "Curated fragment policy:",
        "- Keep `SKILL.md`",
        "- Keep useful `scripts/*` text files (`.py`, `.js`, `.sh`, `.md`, `.txt`) up to 40 KB",
        "- Keep `references/*` text files and common direct references up to 50 KB",
        "- Drop schema blobs, licenses, lockfiles, and other low-value prompt baggage",
        "",
        "Dense full summary:",
        f"- tasks: {dense_full_summary['task_count']}",
        f"- unique fragments: {dense_full_summary['unique_fragment_count']}",
        f"- avg query bytes: {dense_full_summary['avg_query_bytes']}",
        f"- avg shared bytes: {dense_full_summary['avg_shared_bytes']}",
        f"- avg prefix bytes: {dense_full_summary['avg_prefix_bytes']}",
        "",
        "Dense core summary:",
        f"- tasks: {dense_core_summary['task_count']}",
        f"- unique fragments: {dense_core_summary['unique_fragment_count']}",
        f"- avg query bytes: {dense_core_summary['avg_query_bytes']}",
        f"- avg shared bytes: {dense_core_summary['avg_shared_bytes']}",
        f"- avg prefix bytes: {dense_core_summary['avg_prefix_bytes']}",
        "",
        "Component overview:",
    ]
    for row in component_rows:
        component_id = row["component_id"]
        task_count = row["task_count"]
        if component_id == "isolated":
            lines.append(f"- isolated tasks: {task_count}")
            continue
        lines.append(
            f"- component {component_id}: {task_count} tasks, "
            f"{row['unique_fragment_count']} unique fragments, "
            f"pair overlap {row['pair_overlap_ratio']}"
        )

    lines.extend(
        [
            "",
            "Primary files:",
            "- `skillsbench_dense_full_manifest.json`",
            "- `skillsbench_dense_full_tasks.json`",
            "- `skillsbench_dense_full_queries.jsonl`",
            "- `skillsbench_dense_full_fragments.jsonl`",
            "- `skillsbench_dense_core_manifest.json`",
            "- `skillsbench_dense_core_tasks.json`",
            "- `skillsbench_dense_core_queries.jsonl`",
            "- `skillsbench_dense_core_fragments.jsonl`",
            "- `skillsbench_component_summary.json`",
            "- `generate_skillsbench_dense.py`",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    generator_copy_path = output_root / "generate_skillsbench_dense.py"
    generator_copy_path.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    tasks, fragments, task_files = build_task_inventory(args.source_root)
    all_task_ids = sorted(tasks)
    adjacency, shared_fragments = build_overlap_graph(task_files)
    components = connected_components(adjacency, all_task_ids)
    if not components:
        raise RuntimeError("No connected components found in curated SkillsBench overlap graph.")

    dense_full_tasks = components[0]
    dense_core_tasks = [task_id for task_id in dense_full_tasks if task_id not in CORE_EXCLUDED_TASKS]

    component_rows = component_summary_rows(
        components=components,
        tasks=tasks,
        task_files=task_files,
        fragments=fragments,
        shared_fragments=shared_fragments,
    )
    stable_json_dump(output_root / "skillsbench_component_summary.json", component_rows)

    dense_full_summary = summarize_component(
        component_tasks=dense_full_tasks,
        tasks=tasks,
        task_files=task_files,
        fragments=fragments,
        shared_fragments=shared_fragments,
    )
    dense_core_summary = summarize_component(
        component_tasks=dense_core_tasks,
        tasks=tasks,
        task_files=task_files,
        fragments=fragments,
        shared_fragments=shared_fragments,
    )

    if args.variant in ("all", "dense_full"):
        write_variant(
            output_root=output_root,
            variant_name="skillsbench_dense_full",
            description=(
                "Largest connected component under curated exact fragment overlap on "
                "SkillsBench canonical tasks."
            ),
            task_ids=dense_full_tasks,
            tasks=tasks,
            task_files=task_files,
            fragments=fragments,
            summary=dense_full_summary,
            source_root=args.source_root,
        )

    if args.variant in ("all", "dense_core"):
        write_variant(
            output_root=output_root,
            variant_name="skillsbench_dense_core",
            description=(
                "Dense full workload with edge task `pedestrian-traffic-counting` "
                "removed to keep the core document/office cluster tighter."
            ),
            task_ids=dense_core_tasks,
            tasks=tasks,
            task_files=task_files,
            fragments=fragments,
            summary=dense_core_summary,
            source_root=args.source_root,
        )

    generate_readme(
        output_root=output_root,
        dense_full_summary=dense_full_summary,
        dense_core_summary=dense_core_summary,
        component_rows=component_rows,
    )

    print(
        json.dumps(
            {
                "source_root": str(args.source_root),
                "output_root": str(output_root),
                "dense_full_tasks": len(dense_full_tasks),
                "dense_core_tasks": len(dense_core_tasks),
                "dense_full_fragments": dense_full_summary["unique_fragment_count"],
                "dense_core_fragments": dense_core_summary["unique_fragment_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
