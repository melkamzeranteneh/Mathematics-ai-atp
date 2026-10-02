"""Prepare and launch text-only Laya retraining jobs.

This module deliberately does not build PyG graphs. It translates Lean proof
states into Laya-readable text before creating choice-training records. A
remote server can supply its installed Laya training callable through
``trainer=`` or the CLI's ``--trainer module:function`` option.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maths_ai.gnn_inference.atp_lean_gnn.labels import normalize_tactic

from .laya_adapter import (
    choice_question,
    translate_lean_state,
    translate_tactic_step,
)
from .tactic_choose import validate_tactic_names

LAYA_RETRAIN_INSTRUCTIONS = "Which Lean 4 tactic family best matches this proof step?"


@dataclass(frozen=True)
class RetrainConfig:
    dataset_path: Path
    output_path: Path
    candidates: tuple[str, ...] = ()
    state_column: str = "text_state"
    tactic_column: str = "tactic"
    candidate_column: str | None = "candidates"
    instructions: str = LAYA_RETRAIN_INSTRUCTIONS
    split_name: str = "train"


def _parse_candidates(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [item.strip() for item in text.split(",") if item.strip()]
        value = parsed
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    raise ValueError("candidate options must be a list or a JSON/comma-separated string")


def _row_candidates(
    row: Mapping[str, Any],
    *,
    target: str,
    configured: Sequence[str],
    candidate_column: str | None,
) -> tuple[str, ...]:
    candidates = _parse_candidates(row.get(candidate_column)) if candidate_column else None
    selected = candidates or list(configured)
    if not selected:
        raise ValueError(
            "no candidate pool was supplied; add a candidates column or pass an explicit candidate list"
        )
    selected = list(validate_tactic_names(selected))
    if target not in selected:
        selected.insert(0, target)
    return tuple(selected)


def build_training_record(
    row: Mapping[str, Any],
    *,
    candidates: Sequence[str] = (),
    state_column: str = "text_state",
    tactic_column: str = "tactic",
    candidate_column: str | None = "candidates",
    instructions: str = LAYA_RETRAIN_INSTRUCTIONS,
    split_name: str = "train",
) -> dict[str, Any]:
    """Translate one dataset row into an official-style Laya choice record."""
    raw_state = str(row.get(state_column, "")).strip()
    if not raw_state:
        raise ValueError(f"dataset row is missing a non-empty {state_column!r}")
    target = normalize_tactic(str(row.get(tactic_column, "")))
    if not target or target.startswith("<"):
        raise ValueError(f"dataset row has no usable tactic in {tactic_column!r}")
    option_list = _row_candidates(
        row,
        target=target,
        configured=candidates,
        candidate_column=candidate_column,
    )
    translated_state = translate_lean_state(raw_state)
    return {
        "state": translated_state,
        "raw_state": raw_state,
        "questions": {"tactic": choice_question(option_list, instructions)["tactic"]},
        "expected": {
            "tactic": target,
            "description": translate_tactic_step(target),
        },
        "tags": ["atp", "laya-retrain", split_name],
        "row_index": row.get("row_index"),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, JSON, CSV, or parquet rows without converting to graphs."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas and pyarrow to read parquet training data") from exc
        return pd.read_parquet(path).to_dict("records")
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON training data must contain a list of rows")
        return payload
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def prepare_training_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: RetrainConfig,
) -> list[dict[str, Any]]:
    records = []
    for row_number, row in enumerate(rows):
        try:
            records.append(
                build_training_record(
                    row,
                    candidates=config.candidates,
                    state_column=config.state_column,
                    tactic_column=config.tactic_column,
                    candidate_column=config.candidate_column,
                    instructions=config.instructions,
                    split_name=config.split_name,
                )
            )
        except ValueError as exc:
            raise ValueError(f"invalid training row {row_number}: {exc}") from exc
    if not records:
        raise ValueError("the training dataset produced no records")
    return records


def write_training_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def prepare_dataset(config: RetrainConfig) -> Path:
    """Translate a dataset and write the text-only records consumed by Laya."""
    records = prepare_training_records(load_rows(config.dataset_path), config=config)
    return write_training_jsonl(records, config.output_path)


def load_callable(spec: str) -> Callable[..., Any]:
    """Load a remote-server trainer as ``module:function``."""
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("trainer must have the form module:function")
    trainer = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(trainer):
        raise TypeError(f"trainer {spec!r} is not callable")
    return trainer


def retrain_laya(
    config: RetrainConfig,
    *,
    trainer: Callable[..., Any] | None = None,
    trainer_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Prepare records, then optionally hand them to the server's Laya trainer.

    The callable receives ``records`` and keyword arguments. This keeps the
    data preparation stable across Laya releases whose fine-tuning APIs differ.
    With no callable, this function only writes the translated JSONL dataset.
    """
    records = prepare_training_records(load_rows(config.dataset_path), config=config)
    output_path = write_training_jsonl(records, config.output_path)
    if trainer is None:
        return output_path
    kwargs = dict(trainer_kwargs or {})
    return trainer(records=records, output_path=str(output_path), **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate Lean dataset rows for Laya retraining")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidates", help="comma-separated candidate tactic families")
    parser.add_argument("--state-column", default="text_state")
    parser.add_argument("--tactic-column", default="tactic")
    parser.add_argument("--candidate-column", default="candidates")
    parser.add_argument("--split", default="train")
    parser.add_argument("--trainer", help="optional remote trainer as module:function")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidates = tuple(_parse_candidates(args.candidates) or ())
    config = RetrainConfig(
        dataset_path=args.dataset,
        output_path=args.output,
        candidates=validate_tactic_names(list(candidates)),
        state_column=args.state_column,
        tactic_column=args.tactic_column,
        candidate_column=args.candidate_column or None,
        split_name=args.split,
    )
    trainer = load_callable(args.trainer) if args.trainer else None
    result = retrain_laya(config, trainer=trainer)
    print(f"prepared translated Laya records: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
