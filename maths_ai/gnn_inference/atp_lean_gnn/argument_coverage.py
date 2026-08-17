from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch

from .dataset import canonicalize_split_name
from .labels import get_tactic_arity, parse_tactic_arguments
from .lemma_corpus import load_lemma_corpus
from .reporting import console_print
from .training import load_prepared_metadata


ALL_FRESH_IDENTIFIER_TACTICS = {
    "intro",
    "intros",
    "rintro",
    "rename_i",
}
FIRST_FRESH_IDENTIFIER_TACTICS = {"generalize", "let", "set"}
_LITERAL_RE = re.compile(
    r"^(?:[+-]?[0-9]+(?:\.[0-9]+)?|true|false|True|False|none|some)$"
)


@dataclass(frozen=True)
class ArgumentCoverageConfig:
    prepared_root: Path
    output_dir: Path
    splits: tuple[str, ...] = ("train", "val", "test")
    lemma_corpus: Path | None = None
    lemma_index: Path | None = None
    max_items_per_split: int | None = None
    sample_limit_per_category: int = 20
    force: bool = False

    def normalized(self) -> "ArgumentCoverageConfig":
        splits = tuple(canonicalize_split_name(split) for split in self.splits)
        if not splits:
            raise ValueError("At least one split is required.")
        if self.max_items_per_split is not None and self.max_items_per_split < 1:
            raise ValueError("max_items_per_split must be positive when provided.")
        if self.sample_limit_per_category < 0:
            raise ValueError("sample_limit_per_category cannot be negative.")
        if self.lemma_corpus is not None and self.lemma_index is not None:
            raise ValueError("Specify either lemma_corpus or lemma_index, not both.")
        return ArgumentCoverageConfig(
            prepared_root=self.prepared_root.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            splits=splits,
            lemma_corpus=(
                self.lemma_corpus.expanduser().resolve()
                if self.lemma_corpus is not None
                else None
            ),
            lemma_index=(
                self.lemma_index.expanduser().resolve()
                if self.lemma_index is not None
                else None
            ),
            max_items_per_split=self.max_items_per_split,
            sample_limit_per_category=self.sample_limit_per_category,
            force=self.force,
        )


def _tensor_values(data, field: str) -> list[int]:
    value = getattr(data, field, None)
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(item) for item in value.view(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _parse_shape(expected_arity: int, parsed_count: int) -> str:
    if parsed_count == expected_arity:
        return "exact"
    if parsed_count < expected_arity:
        return "missing_tokens"
    if expected_arity == 0:
        return "registry_zero_but_parser_found_tokens"
    return "excess_tokens_or_compound_expression"


def _is_fresh_identifier_slot(tactic_name: str, step: int) -> bool:
    return tactic_name in ALL_FRESH_IDENTIFIER_TACTICS or (
        tactic_name in FIRST_FRESH_IDENTIFIER_TACTICS and step == 0
    )


def classify_argument_slots(
    data,
    *,
    split: str,
    lemma_names: set[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Classify each pointer-expected argument slot in one prepared example."""
    tactic_raw = str(getattr(data, "tactic_raw", ""))
    stored_tactic_name = str(getattr(data, "tactic_name", ""))
    parsed_tactic_name, tokens = parse_tactic_arguments(tactic_raw)
    tactic_name = stored_tactic_name or parsed_tactic_name
    expected_arity = get_tactic_arity(tactic_name)
    parse_shape = _parse_shape(expected_arity, len(tokens))
    node_indices = _tensor_values(data, "arg_node_indices")
    lemma_ids = _tensor_values(data, "arg_lemma_ids")
    premise_mask = getattr(data, "premise_mask", None)
    if torch.is_tensor(premise_mask):
        selectable = [bool(item) for item in premise_mask.view(-1).tolist()]
    elif isinstance(premise_mask, (list, tuple)):
        selectable = [bool(item) for item in premise_mask]
    else:
        selectable = []

    row_index = int(getattr(data, "row_index", -1))
    theorem = str(getattr(data, "theorem", ""))
    records: list[dict[str, object]] = []
    ambiguous_parse = parse_shape in {
        "excess_tokens_or_compound_expression",
        "registry_zero_but_parser_found_tokens",
    }

    for step in range(expected_arity):
        token = tokens[step] if step < len(tokens) else None
        node_id = node_indices[step] if step < len(node_indices) else -1
        lemma_id = lemma_ids[step] if step < len(lemma_ids) else -1

        if token is None:
            category = "parser_missing_argument"
        elif node_id >= 0:
            if node_id >= len(selectable):
                category = "invalid_cached_node_index"
            elif selectable[node_id]:
                category = "local_selectable"
            else:
                category = "local_present_but_masked"
        elif lemma_id >= 0 or (lemma_names is not None and token in lemma_names):
            category = "global_library_lemma"
        elif _is_fresh_identifier_slot(tactic_name, step):
            category = "fresh_identifier"
        elif _LITERAL_RE.fullmatch(token):
            category = "literal"
        elif "." in token:
            category = "unresolved_qualified_name"
        else:
            category = "unmapped_token"

        records.append(
            {
                "split": split,
                "row_index": row_index,
                "theorem": theorem,
                "tactic_name": tactic_name,
                "tactic_raw": tactic_raw,
                "argument_step": step,
                "argument_token": token,
                "category": category,
                "node_id": node_id,
                "lemma_id": lemma_id,
                "expected_arity": expected_arity,
                "parsed_argument_count": len(tokens),
                "parse_shape": parse_shape,
                "semantically_ambiguous_parse": ambiguous_parse,
            }
        )

    row_info = {
        "tactic_name": tactic_name,
        "expected_arity": expected_arity,
        "parsed_argument_count": len(tokens),
        "ignored_parsed_token_count": max(0, len(tokens) - expected_arity),
        "parse_shape": parse_shape,
    }
    return records, row_info


def _iter_packed_graphs(prepared_root: Path, split: str) -> Iterator[object]:
    manifest_path = prepared_root / "packed" / "bidirectional" / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_payload = dict(manifest.get("splits", {})).get(split)
    if not isinstance(split_payload, dict):
        return
    chunk_names = split_payload.get("chunks", [])
    if not isinstance(chunk_names, list) or not chunk_names:
        return
    packed_root = manifest_path.parent / split
    for chunk_name in chunk_names:
        chunk_path = packed_root / str(chunk_name)
        if not chunk_path.exists():
            raise FileNotFoundError(f"Packed graph chunk does not exist: {chunk_path}")
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if not isinstance(chunk, list):
            raise ValueError(f"Packed graph chunk must contain a list: {chunk_path}")
        yield from chunk


def _iter_graphs(prepared_root: Path, split: str) -> tuple[Iterable[object], str]:
    packed_manifest = prepared_root / "packed" / "bidirectional" / "manifest.json"
    if packed_manifest.exists():
        manifest = json.loads(packed_manifest.read_text(encoding="utf-8"))
        split_payload = dict(manifest.get("splits", {})).get(split)
        if isinstance(split_payload, dict) and split_payload.get("chunks"):
            return _iter_packed_graphs(prepared_root, split), "packed"

    metadata = load_prepared_metadata(prepared_root)
    pyg_dir = metadata.split_pyg_dir(split)
    files = sorted(pyg_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No prepared PyG artifacts found under {pyg_dir}")

    def individual_graphs() -> Iterator[object]:
        for path in files:
            yield torch.load(path, map_location="cpu", weights_only=False)

    return individual_graphs(), "individual"


def _new_tactic_stats() -> dict[str, object]:
    return {
        "row_count": 0,
        "expected_argument_slots": 0,
        "parsed_argument_tokens": 0,
        "ignored_parsed_tokens": 0,
        "categories": Counter(),
        "parse_shapes": Counter(),
    }


def _finalize_stats(stats: dict[str, object]) -> dict[str, object]:
    categories = Counter(stats["categories"])
    expected = int(stats["expected_argument_slots"])
    selectable = int(categories["local_selectable"])
    unambiguous_selectable = int(stats.get("unambiguous_selectable_slots", 0))
    return {
        "row_count": int(stats["row_count"]),
        "expected_argument_slots": expected,
        "parsed_argument_tokens": int(stats["parsed_argument_tokens"]),
        "ignored_parsed_tokens": int(stats["ignored_parsed_tokens"]),
        "categories": dict(sorted(categories.items())),
        "parse_shapes": dict(sorted(Counter(stats["parse_shapes"]).items())),
        "local_selectable_coverage": selectable / max(expected, 1),
        "conservative_unambiguous_local_coverage": (
            unambiguous_selectable / max(expected, 1)
        ),
    }


def _render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Argument Coverage Audit",
        "",
        f"- Prepared root: `{summary['prepared_root']}`",
        f"- Lemma corpus: `{summary.get('lemma_corpus') or 'not supplied'}`",
        f"- Lemma index: `{summary.get('lemma_index') or 'not supplied'}`",
        "",
    ]
    for split, payload in dict(summary["splits"]).items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Rows: `{payload['row_count']}`",
                f"- Expected argument slots: `{payload['expected_argument_slots']}`",
                f"- Local selectable coverage: `{payload['local_selectable_coverage']:.4%}`",
                "- Conservative unambiguous local coverage: "
                f"`{payload['conservative_unambiguous_local_coverage']:.4%}`",
                f"- Ignored parsed tokens: `{payload['ignored_parsed_tokens']}`",
                "",
                "| Category | Slots | Share |",
                "| --- | ---: | ---: |",
            ]
        )
        expected = max(int(payload["expected_argument_slots"]), 1)
        for category, count in dict(payload["categories"]).items():
            lines.append(f"| {category} | {count} | {count / expected:.2%} |")
        lines.extend(["", "### Parser shape", "", "| Shape | Rows |", "| --- | ---: |"])
        for shape, count in dict(payload["parse_shapes"]).items():
            lines.append(f"| {shape} | {count} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_argument_coverage_audit(config: ArgumentCoverageConfig) -> dict[str, object]:
    config = config.normalized()
    load_prepared_metadata(config.prepared_root)
    if config.output_dir.exists() and any(config.output_dir.iterdir()) and not config.force:
        raise FileExistsError(
            f"Output directory is not empty: {config.output_dir}. Use --force to overwrite reports."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    lemma_names: set[str] | None = None
    if config.lemma_corpus is not None:
        lemma_names = {record.name for record in load_lemma_corpus(config.lemma_corpus)}
    elif config.lemma_index is not None:
        names_path = (
            config.lemma_index
            if config.lemma_index.is_file()
            else config.lemma_index / "lemma_names.json"
        )
        if not names_path.exists():
            raise FileNotFoundError(
                f"Lemma index names file does not exist: {names_path}"
            )
        names_payload = json.loads(names_path.read_text(encoding="utf-8"))
        if not isinstance(names_payload, list) or not all(
            isinstance(name, str) for name in names_payload
        ):
            raise ValueError(f"Lemma names file must contain a JSON string list: {names_path}")
        lemma_names = set(names_payload)

    summary: dict[str, object] = {
        "prepared_root": str(config.prepared_root),
        "lemma_corpus": str(config.lemma_corpus) if config.lemma_corpus else None,
        "lemma_index": str(config.lemma_index) if config.lemma_index else None,
        "splits": {},
    }
    records_path = config.output_dir / "argument_samples.jsonl"
    sample_counts: Counter[tuple[str, str]] = Counter()

    with records_path.open("w", encoding="utf-8") as records_handle:
        for split in config.splits:
            graphs, source = _iter_graphs(config.prepared_root, split)
            stats = _new_tactic_stats()
            per_tactic: defaultdict[str, dict[str, object]] = defaultdict(_new_tactic_stats)
            processed = 0
            for data in graphs:
                if config.max_items_per_split is not None and processed >= config.max_items_per_split:
                    break
                processed += 1
                records, row = classify_argument_slots(
                    data,
                    split=split,
                    lemma_names=lemma_names,
                )
                tactic_name = str(row["tactic_name"])
                for target in (stats, per_tactic[tactic_name]):
                    target["row_count"] = int(target["row_count"]) + 1
                    target["expected_argument_slots"] = (
                        int(target["expected_argument_slots"])
                        + int(row["expected_arity"])
                    )
                    target["parsed_argument_tokens"] = (
                        int(target["parsed_argument_tokens"])
                        + int(row["parsed_argument_count"])
                    )
                    target["ignored_parsed_tokens"] = (
                        int(target["ignored_parsed_tokens"])
                        + int(row["ignored_parsed_token_count"])
                    )
                    parse_shapes = target["parse_shapes"]
                    if not isinstance(parse_shapes, Counter):
                        raise TypeError("Internal audit parse-shape counter is invalid.")
                    parse_shapes[str(row["parse_shape"])] += 1

                for record in records:
                    category = str(record["category"])
                    for target in (stats, per_tactic[tactic_name]):
                        categories = target["categories"]
                        if not isinstance(categories, Counter):
                            raise TypeError("Internal audit category counter is invalid.")
                        categories[category] += 1
                        if (
                            category == "local_selectable"
                            and not bool(record["semantically_ambiguous_parse"])
                        ):
                            target["unambiguous_selectable_slots"] = (
                                int(target.get("unambiguous_selectable_slots", 0)) + 1
                            )
                    sample_key = (split, category)
                    if sample_counts[sample_key] < config.sample_limit_per_category:
                        records_handle.write(
                            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        sample_counts[sample_key] += 1

                if processed == 1 or processed % 10000 == 0:
                    console_print(f"  [{split}] audited {processed} rows ({source})")

            split_summary = _finalize_stats(stats)
            split_summary["artifact_source"] = source
            split_summary["per_tactic"] = {
                tactic: _finalize_stats(tactic_stats)
                for tactic, tactic_stats in sorted(per_tactic.items())
            }
            summary_splits = summary["splits"]
            if not isinstance(summary_splits, dict):
                raise TypeError("Internal audit split summary is invalid.")
            summary_splits[split] = split_summary
            console_print(
                f"  [{split}] coverage={split_summary['local_selectable_coverage']:.2%} "
                f"expected_slots={split_summary['expected_argument_slots']}"
            )

    summary_path = config.output_dir / "summary.json"
    markdown_path = config.output_dir / "summary.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    console_print(f"  Wrote summary : {summary_path}")
    console_print(f"  Wrote report  : {markdown_path}")
    console_print(f"  Wrote samples : {records_path}")
    return summary
