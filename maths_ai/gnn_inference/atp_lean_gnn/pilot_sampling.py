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


PILOT_SCHEMA_VERSION = 1


def load_selection_manifest(path: Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
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
    seed: int = 42,
    minimum_train_raw_coverage: float = 0.8,
    require_complete_eval: bool = True,
) -> dict[str, object]:
    """Select complete train theorems proportionally within structural strata.

    Validation and test remain complete. Train strata combine tactic-frequency,
    graph-size proxy, proof-length, and typeclass-context presence. Selection
    occurs at theorem granularity, preventing partial proof traces.
    """
    if target_train_rows < 1:
        raise ValueError("target_train_rows must be positive.")

    cache = SExprCache(prepared_root, project_path="", enabled=True)
    split_rows = {
        split: list(iter_dataset_rows(dataset_name=dataset_name, split=split))
        for split in ("train", "val", "test")
    }
    train_rows = split_rows["train"]
    tactic_buckets, tactic_counts = _frequency_buckets(train_rows)

    rows_by_theorem: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in train_rows:
        rows_by_theorem[row.theorem].append(row)

    eligible_groups: list[dict[str, object]] = []
    covered_train_rows = 0
    for theorem, theorem_rows in rows_by_theorem.items():
        theorem_rows.sort(key=lambda row: row.row_index)
        raw_records = [
            cache.load_for_row(row, extractor_version=SExprCache.EXTRACTOR_VERSION)
            for row in theorem_rows
        ]
        covered_train_rows += sum(record is not None for record in raw_records)
        if any(record is None for record in raw_records):
            continue
        records = [record for record in raw_records if record is not None]
        goal_characters = sum(len(str(record["goal_sexp"])) for record in records)
        theorem_tactics = Counter(_tactic_name(row) for row in theorem_rows)
        bucket_counts = Counter(
            {
                bucket: sum(
                    count
                    for tactic, count in theorem_tactics.items()
                    if tactic_buckets.get(tactic) == bucket
                )
                for bucket in ("head", "medium", "tail")
            }
        )
        tactic_bucket = max(
            ("head", "medium", "tail"),
            key=lambda bucket: (bucket_counts[bucket], {"head": 0, "medium": 1, "tail": 2}[bucket]),
        )
        has_instances = any(
            any(
                str(hypothesis.get("name", "")).startswith("inst")
                for hypothesis in record.get("hyp_sexps", [])
                if isinstance(hypothesis, dict)
            )
            for record in records
        )
        eligible_groups.append(
            {
                "theorem": theorem,
                "rows": theorem_rows,
                "row_count": len(theorem_rows),
                "mean_goal_characters": goal_characters / len(records),
                "tactic_bucket": tactic_bucket,
                "has_instances": has_instances,
            }
        )

    raw_coverage = covered_train_rows / len(train_rows) if train_rows else 0.0
    if raw_coverage < minimum_train_raw_coverage:
        raise RuntimeError(
            "Raw train coverage is too low for a representative pilot: "
            f"{raw_coverage:.2%} < {minimum_train_raw_coverage:.2%}."
        )
    eligible_rows = sum(int(group["row_count"]) for group in eligible_groups)
    if eligible_rows < target_train_rows:
        raise RuntimeError(
            f"Only {eligible_rows} rows belong to fully cached theorems; "
            f"cannot select {target_train_rows}."
        )

    cutoffs = _quantile_cutoffs(
        [float(group["mean_goal_characters"]) for group in eligible_groups]
    )
    strata: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for group in eligible_groups:
        key = (
            str(group["tactic_bucket"]),
            _size_bucket(float(group["mean_goal_characters"]), cutoffs),
            _length_bucket(int(group["row_count"])),
            "instances" if group["has_instances"] else "no-instances",
        )
        strata[key].append(group)

    selected: list[dict[str, object]] = []
    selected_theorems: set[str] = set()
    for key, groups in sorted(strata.items()):
        stratum_rows = sum(int(group["row_count"]) for group in groups)
        quota = max(1, round(target_train_rows * stratum_rows / eligible_rows))
        shuffled = list(groups)
        random.Random(f"{seed}:{key}").shuffle(shuffled)
        accumulated = 0
        for group in shuffled:
            if accumulated >= quota:
                break
            selected.append(group)
            selected_theorems.add(str(group["theorem"]))
            accumulated += int(group["row_count"])

    selected_count = sum(int(group["row_count"]) for group in selected)
    if selected_count < target_train_rows:
        remaining = [
            group
            for group in eligible_groups
            if str(group["theorem"]) not in selected_theorems
        ]
        random.Random(seed).shuffle(remaining)
        for group in remaining:
            if selected_count >= target_train_rows:
                break
            selected.append(group)
            selected_theorems.add(str(group["theorem"]))
            selected_count += int(group["row_count"])

    selected_train_rows = sorted(
        (row for group in selected for row in group["rows"]),
        key=lambda row: row.row_index,
    )
    full_distribution = _distribution(
        row
        for group in eligible_groups
        for row in group["rows"]
    )
    selected_distribution = _distribution(selected_train_rows)

    splits: dict[str, object] = {
        "train": {
            "row_indices": [row.row_index for row in selected_train_rows],
            "theorems": sorted(selected_theorems),
            "row_count": len(selected_train_rows),
            "theorem_count": len(selected_theorems),
            "tactic_distribution": selected_distribution,
            "eligible_tactic_distribution": full_distribution,
            "tactic_total_variation": _total_variation(
                full_distribution, selected_distribution
            ),
        }
    }
    for split in ("val", "test"):
        rows = split_rows[split]
        missing = [
            row.row_index
            for row in rows
            if cache.load_for_row(
                row, extractor_version=SExprCache.EXTRACTOR_VERSION
            )
            is None
        ]
        if missing and require_complete_eval:
            raise RuntimeError(
                f"Raw {split} cache is incomplete: {len(missing)} missing rows."
            )
        missing_set = set(missing)
        covered = [row for row in rows if row.row_index not in missing_set]
        splits[split] = {
            "row_indices": [row.row_index for row in covered],
            "theorems": sorted({row.theorem for row in covered}),
            "row_count": len(covered),
            "theorem_count": len({row.theorem for row in covered}),
            "missing_raw_rows": missing,
        }

    manifest: dict[str, object] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "dataset": dataset_name,
        "seed": seed,
        "target_train_rows": target_train_rows,
        "selected_train_rows": len(selected_train_rows),
        "train_raw_coverage": raw_coverage,
        "fully_cached_train_theorem_rows": eligible_rows,
        "goal_character_cutoffs": {
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
