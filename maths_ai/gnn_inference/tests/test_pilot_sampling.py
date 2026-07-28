from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.pilot_sampling import (
    build_pilot_selection,
    selected_row_indices,
)
from maths_ai.gnn_inference.atp_lean_gnn.preprocess import (
    _build_sexpr_map,
    _load_rows,
)


def _row(
    split: str,
    index: int,
    theorem: str,
    tactic: str,
    *,
    file_path: str | None = None,
) -> DatasetRow:
    return DatasetRow(
        state=f"state {index}",
        target_state=f"target {index}",
        theorem=theorem,
        tactic=tactic,
        split=split,
        row_index=index,
        dataset_name="fake/dataset",
        file_path=file_path or f"Mathlib/{theorem}.lean",
    )


def _rows() -> dict[str, list[DatasetRow]]:
    train: list[DatasetRow] = []
    index = 0
    tactics = ("rw", "simp", "exact", "intro")
    for theorem_index, length in enumerate((1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7)):
        theorem = "aux" if theorem_index < 2 else f"Demo.theorem{theorem_index}"
        for step in range(length):
            train.append(
                _row(
                    "train",
                    index,
                    theorem,
                    tactics[(theorem_index + step) % len(tactics)],
                    file_path=f"Mathlib/File{theorem_index}.lean",
                )
            )
            index += 1
    return {
        "train": train,
        "val": [
            _row("val", index, f"Demo.val{index // 2}", "rw")
            for index in range(5)
        ],
        "test": [
            _row("test", index, f"Demo.test{index // 2}", "simp")
            for index in range(4)
        ],
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

    with patch(
            "maths_ai.gnn_inference.atp_lean_gnn.pilot_sampling.iter_dataset_rows",
            side_effect=iter_rows,
        ):
        first = build_pilot_selection(
            prepared_root=tmp_path,
            output_path=tmp_path / "first.json",
            dataset_name="fake/dataset",
            target_train_rows=18,
            target_val_rows=3,
            target_test_rows=2,
            seed=17,
        )
        second = build_pilot_selection(
            prepared_root=tmp_path,
            output_path=tmp_path / "second.json",
            dataset_name="fake/dataset",
            target_train_rows=18,
            target_val_rows=3,
            target_test_rows=2,
            seed=17,
        )

    selected = set(first["splits"]["train"]["row_indices"])
    assert selected == set(second["splits"]["train"]["row_indices"])
    assert len(selected) >= 18
    for theorem_key in {(row.file_path, row.theorem) for row in rows["train"]}:
        theorem_indices = {
            row.row_index
            for row in rows["train"]
            if (row.file_path, row.theorem) == theorem_key
        }
        assert not (selected & theorem_indices) or theorem_indices <= selected
    assert 3 <= first["splits"]["val"]["row_count"] < len(rows["val"])
    assert 2 <= first["splits"]["test"]["row_count"] <= len(rows["test"])
    assert first["stratum_count"] > 1
    assert first["selection_basis"] == "dataset-metadata-file-clustered-v1"


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


def test_controlled_ablation_rejects_missing_sidecars(tmp_path: Path):
    rows = _rows()["train"][:2]
    cache = SimpleNamespace(
        output_root=tmp_path,
        load_for_row=Mock(side_effect=lambda row, **_kwargs: _raw_record(row)),
    )
    with patch(
        "maths_ai.gnn_inference.atp_lean_gnn.preprocess.ModelSExprCache.load_for_raw_record",
        return_value=None,
    ):
        with pytest.raises(RuntimeError, match="mixed-representation"):
            _build_sexpr_map(
                rows,
                project_path="",
                use_sexpr=True,
                sexpr_cache=cache,
                split_label="train",
                sexpr_variant="model",
                require_complete=True,
            )
