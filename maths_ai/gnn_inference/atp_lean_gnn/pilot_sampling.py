"""Deterministic theorem-level sampling for raw/model S-expression ablations."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .dataset import DATASET_NAME, DatasetRow, iter_dataset_rows
from .labels import label_example
from .preparation import SExprCache
from .state import parse_state


PILOT_SCHEMA_VERSION = 1


def load_selection_manifest(path: Path) -> dict[str, object]:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        detail = "empty" if path.stat().st_size == 0 else "invalid JSON"
        raise ValueError(
            f"Pilot selection manifest is {detail}: {path}"
        ) from exc
    if payload.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported pilot selection manifest: {path}")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"Pilot selection manifest has no splits: {path}")
    return payload


def selected_row_indices(
    path: Path, split: str, *, dataset_name: str | None = None
) -> set[int]:
    payload = load_selection_manifest(path)
    if dataset_name is not None and payload.get("dataset") != dataset_name:
        raise ValueError(
            f"Pilot selection dataset {payload.get('dataset')!r} does not match "
            f"{dataset_name!r}."
        )
    split_payload = payload["splits"].get(split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"Pilot selection manifest has no '{split}' split.")
    row_indices = split_payload.get("row_indices")
    if not isinstance(row_indices, list) or not all(
        isinstance(value, int) for value in row_indices
    ):
        raise ValueError(f"Pilot selection for '{split}' has invalid row indices.")
    if len(row_indices) != len(set(row_indices)):
        raise ValueError(f"Pilot selection for '{split}' contains duplicate rows.")
    return set(row_indices)


def selected_extraction_row_indices(
    path: Path, source_split: str, *, dataset_name: str | None = None
) -> set[int]:
    """Return all logical partition rows backed by one dataset split."""
    payload = load_selection_manifest(path)
    if dataset_name is not None and payload.get("dataset") != dataset_name:
        raise ValueError(
            f"Pilot selection dataset {payload.get('dataset')!r} does not match "
            f"{dataset_name!r}."
        )
    selected: set[int] = set()
    for logical_split, split_payload in payload["splits"].items():
        if not isinstance(split_payload, dict):
            raise ValueError(f"Pilot selection for '{logical_split}' is invalid.")
        if split_payload.get("source_split", logical_split) != source_split:
            continue
        row_indices = split_payload.get("row_indices")
        if not isinstance(row_indices, list) or not all(
            isinstance(value, int) for value in row_indices
        ):
            raise ValueError(
                f"Pilot selection for '{logical_split}' has invalid row indices."
            )
        overlap = selected.intersection(row_indices)
        if overlap:
            raise ValueError(
                "Logical pilot partitions overlap within source split "
                f"'{source_split}'."
            )
        selected.update(row_indices)
    return selected


def _tactic_name(row: DatasetRow) -> str:
    try:
        return str(label_example(row.tactic)["tactic_name"])
    except Exception:
        return "<UNKNOWN>"


def _frequency_buckets(rows: list[DatasetRow]) -> tuple[dict[str, str], Counter[str]]:
    counts = Counter(_tactic_name(row) for row in rows)
    ordered = [
        tactic
        for tactic, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    buckets: dict[str, str] = {}
    for rank, tactic in enumerate(ordered):
        buckets[tactic] = "head" if rank < 30 else "medium" if rank < 90 else "tail"
    return buckets, counts


def _quantile_cutoffs(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    return (
        ordered[min(len(ordered) - 1, len(ordered) // 3)],
        ordered[min(len(ordered) - 1, (2 * len(ordered)) // 3)],
    )


def _size_bucket(value: float, cutoffs: tuple[float, float]) -> str:
    return "small" if value <= cutoffs[0] else "medium" if value <= cutoffs[1] else "large"


def _length_bucket(length: int) -> str:
    return "short" if length <= 2 else "medium" if length <= 8 else "long"


def _context_bucket(mean_hypotheses: float) -> str:
    if mean_hypotheses == 0:
        return "empty"
    return "light" if mean_hypotheses <= 4 else "heavy"


def _dominant_tactic_bucket(
    rows: list[DatasetRow], tactic_buckets: dict[str, str]
) -> str:
    theorem_tactics = Counter(_tactic_name(row) for row in rows)
    bucket_counts = {
        bucket: sum(
            count
            for tactic, count in theorem_tactics.items()
            if tactic_buckets.get(tactic) == bucket
        )
        for bucket in ("head", "medium", "tail")
    }
    return max(
        ("head", "medium", "tail"),
        key=lambda bucket: (
            bucket_counts[bucket],
            {"head": 0, "medium": 1, "tail": 2}[bucket],
        ),
    )


def _select_clustered_evaluation_rows(
    rows: list[DatasetRow],
    *,
    tactic_buckets: dict[str, str],
    target_rows: int,
    seed: int,
) -> tuple[list[DatasetRow], set[str], set[str]]:
    if target_rows >= len(rows):
        return rows, {row.theorem for row in rows}, {row.file_path for row in rows}

    theorem_rows: dict[tuple[str, str], list[DatasetRow]] = defaultdict(list)
    for row in rows:
        theorem_rows[(row.file_path, row.theorem)].append(row)
    groups: list[dict[str, object]] = []
    for group_id, grouped_rows in theorem_rows.items():
        file_path, theorem = group_id
        grouped_rows.sort(key=lambda row: row.row_index)
        groups.append(
            {
                "group_id": group_id,
                "theorem": theorem,
                "rows": grouped_rows,
                "row_count": len(grouped_rows),
                "file_path": file_path,
                "stratum": (
                    _dominant_tactic_bucket(grouped_rows, tactic_buckets),
                    _length_bucket(len(grouped_rows)),
                ),
            }
        )

    source_files = sorted({str(group["file_path"]) for group in groups})
    random.Random(seed).shuffle(source_files)
    file_priority = {file_path: rank for rank, file_path in enumerate(source_files)}
    strata: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for group in groups:
        strata[group["stratum"]].append(group)

    selected: list[dict[str, object]] = []
    selected_group_ids: set[tuple[str, str]] = set()
    for key, stratum_groups in sorted(strata.items()):
        stratum_rows = sum(int(group["row_count"]) for group in stratum_groups)
        quota = max(1, round(target_rows * stratum_rows / len(rows)))
        candidates = list(stratum_groups)
        random.Random(f"{seed}:{key}").shuffle(candidates)
        candidates.sort(key=lambda group: file_priority[str(group["file_path"])])
        accumulated = 0
        for group in candidates:
            if accumulated >= quota:
                break
            selected.append(group)
            selected_group_ids.add(group["group_id"])
            accumulated += int(group["row_count"])

    selected_count = sum(int(group["row_count"]) for group in selected)
    if selected_count < target_rows:
        remaining = [
            group
            for group in groups
            if group["group_id"] not in selected_group_ids
        ]
        random.Random(seed).shuffle(remaining)
        remaining.sort(key=lambda group: file_priority[str(group["file_path"])])
        for group in remaining:
            if selected_count >= target_rows:
                break
            selected.append(group)
            selected_group_ids.add(group["group_id"])
            selected_count += int(group["row_count"])

    selected_rows = sorted(
        (row for group in selected for row in group["rows"]),
        key=lambda row: row.row_index,
    )
    selected_files = {str(group["file_path"]) for group in selected}
    selected_theorems = {
        f"{file_path}::{theorem}" for file_path, theorem in selected_group_ids
    }
    return selected_rows, selected_theorems, selected_files


def _select_balanced_holdout_rows(
    rows: list[DatasetRow],
    *,
    target_rows: int,
    seed: int,
) -> tuple[list[DatasetRow], set[str], set[str]]:
    """Greedily match exact tactic frequencies using whole theorem groups."""
    theorem_rows: dict[tuple[str, str], list[DatasetRow]] = defaultdict(list)
    for row in rows:
        theorem_rows[(row.file_path, row.theorem)].append(row)
    groups: list[dict[str, object]] = []
    for group_id, grouped_rows in theorem_rows.items():
        grouped_rows.sort(key=lambda row: row.row_index)
        groups.append(
            {
                "group_id": group_id,
                "rows": grouped_rows,
                "row_count": len(grouped_rows),
                "tactics": Counter(_tactic_name(row) for row in grouped_rows),
            }
        )
    if target_rows >= len(rows):
        selected_ids = set(theorem_rows)
        return rows, {
            f"{file_path}::{theorem}" for file_path, theorem in selected_ids
        }, {row.file_path for row in rows}

    full_counts = Counter(_tactic_name(row) for row in rows)
    desired = {
        tactic: target_rows * count / len(rows)
        for tactic, count in full_counts.items()
    }
    random.Random(seed).shuffle(groups)
    selected: list[dict[str, object]] = []
    selected_counts: Counter[str] = Counter()
    selected_row_count = 0
    while selected_row_count < target_rows and groups:
        remaining_target = target_rows - selected_row_count

        def score(group: dict[str, object]) -> tuple[float, float]:
            group_tactics = group["tactics"]
            group_size = int(group["row_count"])
            balance = sum(
                count
                * max(desired.get(tactic, 0.0) - selected_counts[tactic], 0.0)
                / max(desired.get(tactic, 0.0), 1.0)
                for tactic, count in group_tactics.items()
            ) / group_size
            size_fit = -abs(remaining_target - group_size) / target_rows
            return balance, size_fit

        best_index = max(range(len(groups)), key=lambda index: score(groups[index]))
        group = groups.pop(best_index)
        selected.append(group)
        selected_row_count += int(group["row_count"])
        selected_counts.update(group["tactics"])

    selected_rows = sorted(
        (row for group in selected for row in group["rows"]),
        key=lambda row: row.row_index,
    )
    selected_ids = {group["group_id"] for group in selected}
    selected_theorems = {
        f"{file_path}::{theorem}" for file_path, theorem in selected_ids
    }
    return selected_rows, selected_theorems, {
        row.file_path for row in selected_rows
    }


def _distribution(rows: Iterable[DatasetRow]) -> dict[str, int]:
    return dict(sorted(Counter(_tactic_name(row) for row in rows).items()))


def _total_variation(
    full_counts: dict[str, int], selected_counts: dict[str, int]
) -> float:
    full_total = sum(full_counts.values())
    selected_total = sum(selected_counts.values())
    if not full_total or not selected_total:
        return 1.0
    labels = set(full_counts) | set(selected_counts)
    return 0.5 * sum(
        abs(
            full_counts.get(label, 0) / full_total
            - selected_counts.get(label, 0) / selected_total
        )
        for label in labels
    )


def build_pilot_selection(
    *,
    prepared_root: Path,
    output_path: Path,
    dataset_name: str = DATASET_NAME,
    target_train_rows: int = 30_000,
    target_val_rows: int = 2_000,
    target_test_rows: int = 2_000,
    seed: int = 42,
    require_cached_train: bool = False,
    evaluation_from_train: bool = False,
) -> dict[str, object]:
    """Select complete train theorems before any S-expression extraction.

    Validation and test remain complete. Train strata combine tactic-frequency,
    proof-state size, proof length, and local-context size. A shared randomized
    source-file priority preserves proportional strata while clustering work
    into fewer Mathlib compilations. Selection occurs at theorem granularity,
    preventing partial proof traces. Exact graph and instance-binder statistics
    are measured after extraction.
    """
    if target_train_rows < 1:
        raise ValueError("target_train_rows must be positive.")
    if target_val_rows < 1 or target_test_rows < 1:
        raise ValueError("target_val_rows and target_test_rows must be positive.")
    if evaluation_from_train and not require_cached_train:
        raise ValueError("evaluation_from_train requires require_cached_train.")

    cache = (
        SExprCache(prepared_root, project_path="", enabled=True)
        if require_cached_train
        else None
    )
    split_rows = {
        split: list(iter_dataset_rows(dataset_name=dataset_name, split=split))
        for split in ("train", "val", "test")
    }
    train_rows = split_rows["train"]
    tactic_buckets, tactic_counts = _frequency_buckets(train_rows)

    rows_by_theorem: dict[tuple[str, str], list[DatasetRow]] = defaultdict(list)
    for row in train_rows:
        rows_by_theorem[(row.file_path, row.theorem)].append(row)

    eligible_groups: list[dict[str, object]] = []
    for group_id, theorem_rows in rows_by_theorem.items():
        file_path, theorem = group_id
        theorem_rows.sort(key=lambda row: row.row_index)
        if cache is not None and any(
            cache.load_for_row(
                row, extractor_version=SExprCache.EXTRACTOR_VERSION
            )
            is None
            for row in theorem_rows
        ):
            continue
        states = [parse_state(row.state) for row in theorem_rows]
        state_characters = sum(len(row.state) for row in theorem_rows)
        hypothesis_count = sum(len(state.hypotheses) for state in states)
        eligible_groups.append(
            {
                "group_id": group_id,
                "theorem": theorem,
                "rows": theorem_rows,
                "row_count": len(theorem_rows),
                "file_path": file_path,
                "mean_state_characters": state_characters / len(theorem_rows),
                "mean_hypothesis_count": hypothesis_count / len(theorem_rows),
                "tactic_bucket": _dominant_tactic_bucket(
                    theorem_rows, tactic_buckets
                ),
            }
        )

    eligible_rows = sum(int(group["row_count"]) for group in eligible_groups)
    if eligible_rows < target_train_rows:
        raise RuntimeError(
            f"Only {eligible_rows} eligible theorem-complete rows are available; "
            f"cannot select {target_train_rows}."
        )

    cutoffs = _quantile_cutoffs(
        [float(group["mean_state_characters"]) for group in eligible_groups]
    )
    source_files = sorted({str(group["file_path"]) for group in eligible_groups})
    random.Random(seed).shuffle(source_files)
    file_priority = {file_path: rank for rank, file_path in enumerate(source_files)}
    strata: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for group in eligible_groups:
        key = (
            str(group["tactic_bucket"]),
            _size_bucket(float(group["mean_state_characters"]), cutoffs),
            _length_bucket(int(group["row_count"])),
            _context_bucket(float(group["mean_hypothesis_count"])),
        )
        strata[key].append(group)

    selected: list[dict[str, object]] = []
    selected_group_ids: set[tuple[str, str]] = set()
    for key, groups in sorted(strata.items()):
        stratum_rows = sum(int(group["row_count"]) for group in groups)
        quota = max(1, round(target_train_rows * stratum_rows / eligible_rows))
        shuffled = list(groups)
        random.Random(f"{seed}:{key}").shuffle(shuffled)
        shuffled.sort(key=lambda group: file_priority[str(group["file_path"])])
        accumulated = 0
        for group in shuffled:
            if accumulated >= quota:
                break
            selected.append(group)
            selected_group_ids.add(group["group_id"])
            accumulated += int(group["row_count"])

    selected_count = sum(int(group["row_count"]) for group in selected)
    if selected_count < target_train_rows:
        remaining = [
            group
            for group in eligible_groups
            if group["group_id"] not in selected_group_ids
        ]
        random.Random(seed).shuffle(remaining)
        remaining.sort(key=lambda group: file_priority[str(group["file_path"])])
        for group in remaining:
            if selected_count >= target_train_rows:
                break
            selected.append(group)
            selected_group_ids.add(group["group_id"])
            selected_count += int(group["row_count"])

    selected_train_rows = sorted(
        (row for group in selected for row in group["rows"]),
        key=lambda row: row.row_index,
    )
    eligible_distribution = _distribution(
        row
        for group in eligible_groups
        for row in group["rows"]
    )
    full_distribution = _distribution(train_rows)
    selected_distribution = _distribution(selected_train_rows)
    selected_theorems = {
        f"{file_path}::{theorem}" for file_path, theorem in selected_group_ids
    }

    if evaluation_from_train:
        holdout_val, val_theorems, val_files = _select_balanced_holdout_rows(
            selected_train_rows,
            target_rows=target_val_rows,
            seed=seed + 1,
        )
        val_indices = {row.row_index for row in holdout_val}
        after_val = [
            row for row in selected_train_rows if row.row_index not in val_indices
        ]
        holdout_test, test_theorems, test_files = (
            _select_balanced_holdout_rows(
                after_val,
                target_rows=target_test_rows,
                seed=seed + 2,
            )
        )
        test_indices = {row.row_index for row in holdout_test}
        logical_train = [
            row for row in after_val if row.row_index not in test_indices
        ]
        train_theorems = {
            f"{row.file_path}::{row.theorem}" for row in logical_train
        }
        splits: dict[str, object] = {}
        for logical_split, logical_rows, theorem_ids, files in (
            (
                "train",
                logical_train,
                train_theorems,
                {row.file_path for row in logical_train},
            ),
            ("val", holdout_val, val_theorems, val_files),
            ("test", holdout_test, test_theorems, test_files),
        ):
            distribution = _distribution(logical_rows)
            splits[logical_split] = {
                "source_split": "train",
                "internal_holdout": logical_split != "train",
                "row_indices": [row.row_index for row in logical_rows],
                "theorems": sorted(theorem_ids),
                "row_count": len(logical_rows),
                "theorem_count": len(theorem_ids),
                "source_file_count": len(files),
                "tactic_distribution": distribution,
                "full_tactic_distribution": full_distribution,
                "tactic_total_variation": _total_variation(
                    full_distribution, distribution
                ),
            }
        selected_distribution = _distribution(logical_train)
    else:
        splits = {
            "train": {
                "source_split": "train",
                "internal_holdout": False,
                "row_indices": [row.row_index for row in selected_train_rows],
                "theorems": sorted(selected_theorems),
                "row_count": len(selected_train_rows),
                "theorem_count": len(selected_theorems),
                "tactic_distribution": selected_distribution,
                "eligible_tactic_distribution": eligible_distribution,
                "full_tactic_distribution": full_distribution,
                "tactic_total_variation": _total_variation(
                    full_distribution, selected_distribution
                ),
            }
        }
        for split in ("val", "test"):
            rows = split_rows[split]
            target_rows = target_val_rows if split == "val" else target_test_rows
            selected_rows, selected_eval_theorems, selected_files = (
                _select_clustered_evaluation_rows(
                    rows,
                    tactic_buckets=tactic_buckets,
                    target_rows=target_rows,
                    seed=seed + (1 if split == "val" else 2),
                )
            )
            splits[split] = {
                "source_split": split,
                "internal_holdout": False,
                "row_indices": [row.row_index for row in selected_rows],
                "theorems": sorted(selected_eval_theorems),
                "row_count": len(selected_rows),
                "theorem_count": len(selected_eval_theorems),
                "source_file_count": len(selected_files),
                "full_split_row_count": len(rows),
                "tactic_distribution": _distribution(selected_rows),
                "full_tactic_distribution": _distribution(rows),
                "tactic_total_variation": _total_variation(
                    _distribution(rows), _distribution(selected_rows)
                ),
            }

    selected_source_files = {
        str(group["file_path"]) for group in selected
    }
    manifest: dict[str, object] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "selection_basis": "dataset-metadata-file-clustered-v1",
        "train_cache_required": require_cached_train,
        "evaluation_from_train": evaluation_from_train,
        "dataset": dataset_name,
        "seed": seed,
        "target_train_rows": target_train_rows,
        "target_val_rows": target_val_rows,
        "target_test_rows": target_test_rows,
        "selected_source_train_rows": len(selected_train_rows),
        "selected_train_rows": int(splits["train"]["row_count"]),
        "eligible_train_rows": eligible_rows,
        "eligible_train_fraction": (
            eligible_rows / len(train_rows) if train_rows else 0.0
        ),
        "selected_train_source_files": len(selected_source_files),
        "total_train_source_files": len(source_files),
        "state_character_cutoffs": {
            "small_medium": cutoffs[0],
            "medium_large": cutoffs[1],
        },
        "stratum_count": len(strata),
        "splits": splits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return manifest
