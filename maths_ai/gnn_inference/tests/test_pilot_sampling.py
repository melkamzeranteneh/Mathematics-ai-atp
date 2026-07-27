from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.pilot_sampling import (
    build_pilot_selection,
    selected_row_indices,
)
from maths_ai.gnn_inference.atp_lean_gnn.preprocess import _load_rows


def _row(split: str, index: int, theorem: str, tactic: str) -> DatasetRow:
    return DatasetRow(
        state=f"state {index}",
        target_state=f"target {index}",
        theorem=theorem,
        tactic=tactic,
        split=split,
        row_index=index,
        dataset_name="fake/dataset",
    )


def _rows() -> dict[str, list[DatasetRow]]:
    train: list[DatasetRow] = []
    index = 0
    tactics = ("rw", "simp", "exact", "intro")
    for theorem_index, length in enumerate((1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7)):
        for step in range(length):
            train.append(
                _row(
                    "train",
                    index,
                    f"Demo.theorem{theorem_index}",
                    tactics[(theorem_index + step) % len(tactics)],
                )
            )
            index += 1
    return {
        "train": train,
        "val": [_row("val", index, "Demo.val", "rw") for index in range(5)],
        "test": [_row("test", index, "Demo.test", "simp") for index in range(4)],
    }


def _raw_record(row: DatasetRow) -> dict[str, object]:
    hypotheses = (
        [{"name": "instDemo", "sexp": "(:c Demo)"}]
        if row.row_index % 2 == 0
        else [{"name": "x", "sexp": "(:c Nat)"}]
    )
    return {
        "goal_sexp": "(:app (:c Goal) " + "x" * (row.row_index % 7) + ")",
        "hyp_sexps": hypotheses,
    }


def test_pilot_is_deterministic_and_selects_whole_theorems(tmp_path: Path):
    rows = _rows()

    def iter_rows(*, split: str, **_kwargs):
        return iter(rows[split])

    def load_raw(_cache, row: DatasetRow, **_kwargs):
        return _raw_record(row)

    with (
        patch(
            "maths_ai.gnn_inference.atp_lean_gnn.pilot_sampling.iter_dataset_rows",
            side_effect=iter_rows,
        ),
        patch(
            "maths_ai.gnn_inference.atp_lean_gnn.pilot_sampling.SExprCache.load_for_row",
            new=load_raw,
        ),
    ):
        first = build_pilot_selection(
            prepared_root=tmp_path,
            output_path=tmp_path / "first.json",
            dataset_name="fake/dataset",
            target_train_rows=18,
            seed=17,
        )
        second = build_pilot_selection(
            prepared_root=tmp_path,
            output_path=tmp_path / "second.json",
            dataset_name="fake/dataset",
            target_train_rows=18,
            seed=17,
        )

    selected = set(first["splits"]["train"]["row_indices"])
    assert selected == set(second["splits"]["train"]["row_indices"])
    assert len(selected) >= 18
    for theorem in {row.theorem for row in rows["train"]}:
        theorem_indices = {
            row.row_index for row in rows["train"] if row.theorem == theorem
        }
        assert not (selected & theorem_indices) or theorem_indices <= selected
    assert first["splits"]["val"]["row_count"] == len(rows["val"])
    assert first["splits"]["test"]["row_count"] == len(rows["test"])
    assert first["stratum_count"] > 1


def test_selection_manifest_filters_exact_rows(tmp_path: Path):
    rows = _rows()
    manifest = tmp_path / "pilot.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "fake/dataset",
                "splits": {
                    "train": {"row_indices": [1, 3]},
                    "val": {"row_indices": []},
                    "test": {"row_indices": []},
                },
            }
        ),
        encoding="utf-8",
    )
    with patch(
        "maths_ai.gnn_inference.atp_lean_gnn.preprocess.iter_dataset_rows",
        return_value=iter(rows["train"]),
    ):
        filtered = _load_rows(
            "fake/dataset", "train", selection_manifest=manifest
        )
    assert [row.row_index for row in filtered] == [1, 3]
    assert selected_row_indices(
        manifest, "train", dataset_name="fake/dataset"
    ) == {1, 3}


def test_selection_manifest_rejects_sample_limit(tmp_path: Path):
    manifest = tmp_path / "pilot.json"
    manifest.write_text(
        '{"schema_version": 1, "dataset": "fake/dataset", '
        '"splits": {"train": {"row_indices": []}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        _load_rows(
            "fake/dataset",
            "train",
            sample_limit=1,
            selection_manifest=manifest,
        )
