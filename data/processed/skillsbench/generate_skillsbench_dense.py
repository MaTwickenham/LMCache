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
ANCHOR_SKILL_NAME = "request_anchor"
ANCHOR_RELATIVE_PATH = "generated/request_brief.md"
ANCHOR_FOCUS_LINES = (
    "adapt reusable skills to the current request",
    "validate edge cases before producing the final answer",
    "prioritize output formatting and completion checks",
    "cross-check constraints against the attached skill manuals",
    "compress shared procedures into a concise execution plan",
    "focus on failure recovery and fallback handling",
)
ANCHOR_REVIEW_LINES = (
    "surface the most task-specific constraints first",
    "keep broad support material behind the request brief",
    "treat the downstream skill context as reusable support",
    "separate ephemeral request state from reusable memory",
)


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


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def slugify_label(value: object) -> str:
    raw = str(value or "").strip().lower()
    cleaned = [
        char if char.isalnum() else "_"
        for char in raw
    ]
    label = "".join(cleaned).strip("_")
    while "__" in label:
        label = label.replace("__", "_")
    return label or "unknown"


def role_priority(role: str) -> int:
    return {
        "core_support": 3,
        "primary": 2,
        "secondary_support": 1,
    }.get(role, 0)


def choose_dominant_role(role_counts: Counter[str]) -> str:
    if not role_counts:
        return "primary"
    return max(
        role_counts.items(),
        key=lambda item: (item[1], role_priority(item[0]), item[0]),
    )[0]


def infer_attachment_role(
    *,
    component_size: int,
    fragment_task_count: int,
    skill_task_count: int,
) -> str:
    share_count = max(int(fragment_task_count), int(skill_task_count))
    if share_count <= 1:
        return "primary"
    if component_size <= 0:
        return "secondary_support"
    share_ratio = float(share_count) / float(component_size)
    if share_ratio >= 0.5 or share_count >= 4:
        return "core_support"
    return "secondary_support"


def infer_attachment_reason(role: str) -> str:
    if role == "core_support":
        return "widely reused support skill shared across related tasks"
    if role == "secondary_support":
        return "partially shared helper skill reused across a subset of tasks"
    return "task-specific instruction or narrow skill tied to the current request"


def build_component_lookup(
    *,
    components: list[list[str]],
    all_tasks: list[str],
) -> tuple[dict[str, int], dict[int, set[str]]]:
    task_to_component = {task_id: -1 for task_id in all_tasks}
    component_members: dict[int, set[str]] = {}
    for component_id, component_tasks in enumerate(components):
        members = set(component_tasks)
        component_members[component_id] = members
        for task_id in component_tasks:
            task_to_component[task_id] = component_id
    return task_to_component, component_members


def annotate_fragment_semantics(
    *,
    tasks: dict[str, dict[str, Any]],
    task_files: dict[str, list[SourceFile]],
    fragments: dict[str, dict[str, Any]],
    components: list[list[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    all_tasks = sorted(tasks)
    task_to_component, component_members = build_component_lookup(
        components=components,
        all_tasks=all_tasks,
    )
    skill_tasks_by_component: dict[tuple[int, str], set[str]] = defaultdict(set)
    for task_id, files in task_files.items():
        component_id = task_to_component.get(task_id, -1)
        for source_file in files:
            skill_tasks_by_component[(component_id, source_file.skill_name)].add(task_id)

    for fragment in fragments.values():
        fragment["semantic_groups"] = set(fragment.get("semantic_groups") or [])
        fragment["role_counts"] = Counter(fragment.get("role_counts") or {})
        fragment["primary_task_ids"] = set(fragment.get("primary_task_ids") or [])
        fragment["core_support_task_ids"] = set(fragment.get("core_support_task_ids") or [])
        fragment["secondary_task_ids"] = set(fragment.get("secondary_task_ids") or [])
        fragment["group_frequency"] = int(fragment.get("task_count", 1) or 1)

    task_fragment_meta: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for task_id in all_tasks:
        component_id = task_to_component.get(task_id, -1)
        component_size = len(component_members.get(component_id, {task_id}))
        seen_fragment_ids: set[str] = set()
        for source_file in task_files.get(task_id, []):
            fragment_id = source_file.fragment_id
            if fragment_id in seen_fragment_ids:
                continue
            seen_fragment_ids.add(fragment_id)

            fragment = fragments[fragment_id]
            fragment_task_count = sum(
                1
                for candidate_task in fragment["source_tasks"]
                if task_to_component.get(candidate_task, -1) == component_id
            )
            fragment_task_count = max(fragment_task_count, 1)
            skill_task_count = max(
                len(skill_tasks_by_component[(component_id, source_file.skill_name)]),
                1,
            )
            group_frequency = max(fragment_task_count, skill_task_count)
            role = infer_attachment_role(
                component_size=component_size,
                fragment_task_count=fragment_task_count,
                skill_task_count=skill_task_count,
            )
            reason = infer_attachment_reason(role)
            semantic_groups = {
                f"component::{component_id if component_id >= 0 else 'isolated'}",
                f"skill::{slugify_label(source_file.skill_name)}",
            }
            category = tasks[task_id].get("category")
            if category:
                semantic_groups.add(f"category::{slugify_label(category)}")

            task_fragment_meta[task_id][fragment_id] = {
                "skill_name": source_file.skill_name,
                "relative_path": source_file.relative_path,
                "attachment_role": role,
                "attachment_reason": reason,
                "group_frequency": int(group_frequency),
                "semantic_groups": sorted(semantic_groups),
            }

            for source_record in fragment["source_files"]:
                if (
                    source_record["task_id"] == task_id
                    and source_record["skill_name"] == source_file.skill_name
                    and source_record["relative_path"] == source_file.relative_path
                ):
                    source_record["attachment_role"] = role
                    source_record["attachment_reason"] = reason
                    source_record["group_frequency"] = int(group_frequency)
                    source_record["semantic_groups"] = sorted(semantic_groups)

            fragment["semantic_groups"].update(semantic_groups)
            fragment["role_counts"][role] += 1
            fragment["group_frequency"] = max(
                int(fragment["group_frequency"]),
                int(group_frequency),
            )
            if role == "primary":
                fragment["primary_task_ids"].add(task_id)
            elif role == "core_support":
                fragment["core_support_task_ids"].add(task_id)
            elif role == "secondary_support":
                fragment["secondary_task_ids"].add(task_id)

    for fragment in fragments.values():
        role_counts = Counter(fragment["role_counts"])
        fragment["semantic_groups"] = sorted(fragment["semantic_groups"])
        fragment["usage_roles"] = sorted(role_counts)
        fragment["dominant_role"] = choose_dominant_role(role_counts)
        fragment["role_counts"] = dict(sorted(role_counts.items()))
        fragment["group_frequency"] = max(
            int(fragment.get("group_frequency", fragment["task_count"]) or fragment["task_count"]),
            int(fragment["task_count"]),
        )
        fragment["primary_task_ids"] = sorted(fragment["primary_task_ids"])
        fragment["core_support_task_ids"] = sorted(fragment["core_support_task_ids"])
        fragment["secondary_task_ids"] = sorted(fragment["secondary_task_ids"])

    return {
        task_id: dict(fragment_meta)
        for task_id, fragment_meta in task_fragment_meta.items()
    }


def ordered_semantic_fragment_items(
    *,
    task_id: str,
    task_files: dict[str, list[SourceFile]],
    task_fragment_meta: dict[str, dict[str, dict[str, Any]]],
) -> list[tuple[SourceFile, dict[str, Any]]]:
    seen_fragment_ids: set[str] = set()
    items: list[tuple[SourceFile, dict[str, Any], int]] = []
    for source_index, source_file in enumerate(task_files[task_id]):
        fragment_id = source_file.fragment_id
        if fragment_id in seen_fragment_ids:
            continue
        seen_fragment_ids.add(fragment_id)
        meta = dict(task_fragment_meta.get(task_id, {}).get(fragment_id) or {})
        items.append((source_file, meta, source_index))

    items.sort(
        key=lambda item: (
            {
                "primary": 0,
                "core_support": 1,
                "secondary_support": 2,
            }.get(str(item[1].get("attachment_role") or ""), 3),
            -int(item[1].get("group_frequency", 1) or 1),
            item[2],
            item[0].fragment_id,
        )
    )
    return [(source_file, meta) for source_file, meta, _ in items]


def build_request_anchor_fragment(
    *,
    task_id: str,
    task_meta: dict[str, Any],
    ordered_skill_names: list[str],
    semantic_groups: list[str],
    occurrence_tag: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    focus_seed = stable_int(f"{task_id}::{occurrence_tag}")
    focus_line = ANCHOR_FOCUS_LINES[focus_seed % len(ANCHOR_FOCUS_LINES)]
    review_line = ANCHOR_REVIEW_LINES[(focus_seed // len(ANCHOR_FOCUS_LINES)) % len(ANCHOR_REVIEW_LINES)]
    instruction = " ".join(str(task_meta.get("instruction") or "").split())
    instruction = instruction[:280]
    skill_preview = ", ".join(ordered_skill_names[:4]) if ordered_skill_names else "general_support"
    anchor_text = "\n".join(
        [
            "Request Brief",
            f"Instance: {occurrence_tag}",
            f"Task: {task_id}",
            f"Category: {task_meta.get('category') or 'unknown'}",
            f"Difficulty: {task_meta.get('difficulty') or 'unknown'}",
            f"Immediate focus: {focus_line}.",
            f"Review rule: {review_line}.",
            f"Reusable skills to consult: {skill_preview}",
            f"Current request: {instruction}",
        ]
    )
    anchor_bytes = anchor_text.encode("utf-8")
    fragment_id = f"skillsbench:anchor:{sha256_hex(anchor_bytes)[:16]}"
    fragment_row = {
        "fragment_id": fragment_id,
        "sha256": sha256_hex(anchor_bytes),
        "kind": "task_brief",
        "size_bytes": len(anchor_bytes),
        "size_chars": len(anchor_text),
        "line_count": anchor_text.count("\n") + 1,
        "task_count": 1,
        "source_file_count": 1,
        "source_tasks": [task_id],
        "skill_names": [ANCHOR_SKILL_NAME],
        "source_files": [
            {
                "task_id": task_id,
                "skill_name": ANCHOR_SKILL_NAME,
                "relative_path": ANCHOR_RELATIVE_PATH,
                "source_path": f"synthetic://skillsbench/{task_id}/{occurrence_tag}",
                "attachment_role": "primary",
                "attachment_reason": "query-specific working context that should not be reused across requests",
                "group_frequency": 1,
                "semantic_groups": sorted(set(semantic_groups) | {"ephemeral_request"}),
            }
        ],
        "usage_roles": ["primary"],
        "role_counts": {"primary": 1},
        "dominant_role": "primary",
        "semantic_groups": sorted(set(semantic_groups) | {"ephemeral_request"}),
        "group_frequency": 1,
        "primary_task_ids": [task_id],
        "core_support_task_ids": [],
        "secondary_task_ids": [],
        "text": anchor_text,
    }
    query_fragment = {
        "position": 0,
        "fragment_id": fragment_id,
        "kind": "task_brief",
        "skill_name": ANCHOR_SKILL_NAME,
        "relative_path": ANCHOR_RELATIVE_PATH,
        "size_bytes": len(anchor_bytes),
        "size_chars": len(anchor_text),
        "task_frequency": 1,
        "group_frequency": 1,
        "attachment_role": "primary",
        "attachment_reason": "query-specific working context that separates ephemeral intent from reusable skill memory",
        "semantic_groups": sorted(set(semantic_groups) | {"ephemeral_request"}),
    }
    return fragment_row, query_fragment


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
    task_fragment_meta: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    anchor_fragments: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        fragment_items = ordered_semantic_fragment_items(
            task_id=task_id,
            task_files=task_files,
            task_fragment_meta=task_fragment_meta,
        )
        base_skills = [
            str(meta.get("skill_name") or source_file.skill_name)
            for source_file, meta in fragment_items
        ]
        semantic_groups = sorted(
            {
                group
                for _source_file, meta in fragment_items
                for group in meta.get("semantic_groups") or []
            }
        )
        anchor_row, anchor_query_fragment = build_request_anchor_fragment(
            task_id=task_id,
            task_meta=tasks[task_id],
            ordered_skill_names=list(dict.fromkeys(base_skills)),
            semantic_groups=semantic_groups,
            occurrence_tag=variant_name,
        )
        anchor_fragments[anchor_row["fragment_id"]] = anchor_row

        fragment_ids = [anchor_row["fragment_id"]]
        per_fragment = [anchor_query_fragment]
        query_bytes = int(anchor_row["size_bytes"])
        for position, (source_match, meta) in enumerate(fragment_items, start=1):
            fragment_id = source_match.fragment_id
            fragment = fragments[fragment_id]
            fragment_ids.append(fragment_id)
            query_bytes += fragment["size_bytes"]
            per_fragment.append(
                {
                    "position": position,
                    "fragment_id": fragment_id,
                    "kind": fragment["kind"],
                    "skill_name": str(meta.get("skill_name") or source_match.skill_name),
                    "relative_path": str(meta.get("relative_path") or source_match.relative_path),
                    "size_bytes": fragment["size_bytes"],
                    "size_chars": fragment["size_chars"],
                    "task_frequency": fragment["task_count"],
                    "group_frequency": int(
                        meta.get("group_frequency", fragment.get("group_frequency", fragment["task_count"]))
                    ),
                    "attachment_role": str(meta.get("attachment_role") or fragment.get("dominant_role") or "primary"),
                    "attachment_reason": str(
                        meta.get("attachment_reason")
                        or "semantic role inferred from task-local and component-level reuse"
                    ),
                    "semantic_groups": list(
                        meta.get("semantic_groups")
                        or fragment.get("semantic_groups")
                        or []
                    ),
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
    return rows, sorted(anchor_fragments.values(), key=lambda row: row["fragment_id"])


def build_fragment_rows(
    *,
    task_ids: list[str],
    fragments: dict[str, dict[str, Any]],
    extra_fragments: list[dict[str, Any]] | None = None,
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
                "usage_roles": list(fragment.get("usage_roles") or []),
                "role_counts": dict(fragment.get("role_counts") or {}),
                "dominant_role": str(fragment.get("dominant_role") or "primary"),
                "semantic_groups": list(fragment.get("semantic_groups") or []),
                "group_frequency": int(
                    fragment.get("group_frequency", len(surviving_tasks)) or len(surviving_tasks)
                ),
                "primary_task_ids": list(fragment.get("primary_task_ids") or []),
                "core_support_task_ids": list(fragment.get("core_support_task_ids") or []),
                "secondary_task_ids": list(fragment.get("secondary_task_ids") or []),
                "text": fragment["text"],
            }
        )
    for fragment in sorted(extra_fragments or [], key=lambda row: row["fragment_id"]):
        rows.append(fragment)
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
    task_fragment_meta: dict[str, dict[str, dict[str, Any]]],
    summary: dict[str, Any],
    source_root: Path,
) -> None:
    query_rows, anchor_fragments = build_query_rows(
        variant_name=variant_name,
        task_ids=task_ids,
        tasks=tasks,
        task_files=task_files,
        fragments=fragments,
        task_fragment_meta=task_fragment_meta,
    )
    query_index = {row["task_id"]: row for row in query_rows}
    fragment_rows = build_fragment_rows(
        task_ids=task_ids,
        fragments=fragments,
        extra_fragments=anchor_fragments,
    )
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
        "- Each query now starts with a short synthetic request brief to separate",
        "  query-specific intent from reusable skill/support fragments",
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
    task_fragment_meta = annotate_fragment_semantics(
        tasks=tasks,
        task_files=task_files,
        fragments=fragments,
        components=components,
    )

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
            task_fragment_meta=task_fragment_meta,
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
            task_fragment_meta=task_fragment_meta,
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
