#!/usr/bin/env python3
"""Generate the Hugging Face dataset card from extraction manifests.

Every coverage number in the card is read from the extraction manifests and the
pack report rather than typed by hand.  The train split is only partially
extracted and its missing rows are not a uniform random sample, so the card has
to state the per-phase failure taxonomy accurately -- a transcription error here
would misrepresent the dataset to anyone who trains on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reported alongside each split so readers can tell an environment limit from a
# property of the data itself.  Only the first two are worth retrying; the rest
# describe rows this extractor cannot represent.
PHASE_NOTES = {
    "file_compile": (
        "environment",
        "Lean could not elaborate the source file within the timeout, or the "
        "REPL exited while compiling it. Retryable.",
    ),
    "theorem_identity": (
        "environment",
        "The theorem's compilation unit could not be identified in the file. "
        "Retryable.",
    ),
    "goal_cardinality": (
        "extractor limit",
        "The tactic was invoked against multiple pending goals. This extractor "
        "records single-goal states only, so these rows are absent by design.",
    ),
    "model_goal_cardinality": (
        "extractor limit",
        "As goal_cardinality, raised while building the normalized sidecar.",
    ),
    "action_goal_cardinality": (
        "extractor limit",
        "As goal_cardinality, raised while building the action trace.",
    ),
    "invocation_alignment": (
        "row data",
        "The dataset's tactic text could not be matched to any invocation "
        "recorded by Lean in that file.",
    ),
    "ambiguous_invocation": (
        "row data",
        "The tactic text matched more than one invocation, so the proof state "
        "could not be attributed unambiguously.",
    ),
    "sexpr_capture": (
        "row data",
        "Lean returned an invocation without a serialized S-expression.",
    ),
    "model_sexpr_capture": (
        "row data",
        "Lean returned an invocation without a normalized S-expression.",
    ),
    "source_syntax_capture": (
        "row data",
        "Lean returned an invocation without a compact tactic-syntax trace.",
    ),
    "action_trace_capture": (
        "row data",
        "The elaborated tactic terms could not be recorded.",
    ),
    "action_context_metadata": (
        "row data",
        "A local-context entry lacked a stable index or binder role.",
    ),
    "model_context_metadata": (
        "row data",
        "A hypothesis lacked a stable context index or binder role.",
    ),
    "commit_mismatch": (
        "environment",
        "The source checkout was not at the commit recorded in the dataset.",
    ),
    "source_metadata": (
        "environment",
        "The source file could not be read at the expected path.",
    ),
    "unexpected_error": (
        "environment",
        "An unclassified extraction error. Retryable.",
    ),
}

COLUMN_NOTES = [
    ("dataset", "Upstream dataset name the row came from."),
    ("split", "Upstream split: `train`, `val`, or `test`."),
    ("row_index", "Row index within the upstream split. Stable join key."),
    ("theorem", "Fully qualified Mathlib theorem name."),
    ("file_path", "Mathlib source path, relative to the repository root."),
    ("repo_url", "Mathlib repository URL."),
    ("repo_commit", "Mathlib commit the state was extracted at."),
    ("tactic", "The tactic text applied at this step."),
    ("text_state", "Pretty-printed goal before the tactic, as Lean printed it."),
    ("text_target_state", "Pretty-printed goal after the tactic."),
    ("raw_goal_sexp", "Source-faithful S-expression of the goal type."),
    (
        "raw_hyp_sexps",
        "JSON string: list of `{name, sexp}` for each hypothesis, source-faithful.",
    ),
    ("model_goal_sexp", "Normalized S-expression of the goal type."),
    (
        "model_hyp_sexps",
        "JSON string: list of `{name, internal_name, context_index, "
        "binder_role, is_instance, is_let, sexp}`.",
    ),
    (
        "source_syntax",
        "JSON string: the tactic's original syntax tree, with identifier leaves "
        "annotated by Lean-resolved references. This is the generation target.",
    ),
    ("syntax_args", "JSON string: normalized argument slots of the tactic syntax."),
    ("term_ranges", "JSON string: source ranges of elaborated tactic terms."),
    (
        "local_context",
        "JSON string: the local context with stable indices, for pointer-style "
        "argument supervision.",
    ),
    ("unit_index", "Index of the compilation unit within the file."),
    ("invocation_index", "Index of the matched invocation within the unit."),
    ("alignment_kind", "How the dataset tactic text was matched to the invocation."),
    (
        "target_state_matches_invocation",
        "True when Lean's post-tactic goal matches the dataset's target state. "
        "A useful filter for strict agreement with upstream.",
    ),
    ("pending_goal_count", "Pending goals at the invocation. Always 1; see coverage."),
    ("hypothesis_count", "Number of hypotheses in the local context."),
    (
        "hypothesis_names_match",
        "True when Lean's hypothesis names match those parsed from the upstream "
        "pretty-printed state.",
    ),
    ("raw_schema_version", "Schema version of the source-faithful record."),
    ("raw_extractor_version", "Extractor version that produced the raw record."),
    ("model_schema_version", "Schema version of the normalized sidecar."),
    ("model_normalization", "Normalization identifier applied to model S-expressions."),
    ("trace_schema_version", "Schema version of the action trace."),
    ("trace_extractor_version", "Extractor version that produced the action trace."),
    ("pantograph_commit", "Pantograph commit used for raw extraction."),
    ("model_pantograph_commit", "Pantograph commit used for normalization."),
    (
        "raw_record_sha256",
        "Digest binding the sidecars to the raw record. Retained for audit; "
        "consumers need not check it, since all three targets ship in one row.",
    ),
    ("state_sha256", "SHA-256 of the upstream state text."),
    ("tactic_sha256", "SHA-256 of the upstream tactic text."),
    ("target_state_sha256", "SHA-256 of the upstream target-state text."),
]


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.2f}%" if whole else "n/a"


def _frontmatter(
    *,
    license_id: str,
    splits: dict[str, dict],
    pretty_name: str,
) -> str:
    lines = [
        "---",
        f"license: {license_id}",
        "language:",
        "- en",
        f"pretty_name: {pretty_name}",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- lean4",
        "- mathlib",
        "- theorem-proving",
        "- s-expressions",
        "- formal-mathematics",
        "size_categories:",
        "- 100K<n<1M",
        "configs:",
        "- config_name: default",
        "  data_files:",
    ]
    for split in splits:
        lines.append(f"  - split: {split}")
        lines.append(f"    path: {split}/{split}-*.parquet")
    lines.append("dataset_info:")
    lines.append("  splits:")
    for split, info in splits.items():
        lines.append(f"  - name: {split}")
        lines.append(f"    num_examples: {info['rows']}")
    lines.append("---")
    return "\n".join(lines)


def build_card(
    *,
    pack_report: dict,
    manifests: dict[str, dict],
    upstream_sizes: dict[str, int],
    license_id: str,
    repo_id: str,
    pretty_name: str,
) -> str:
    splits = pack_report["splits"]
    packed_total = sum(int(info["rows"]) for info in splits.values())
    upstream_total = sum(upstream_sizes.get(split, 0) for split in splits)

    any_manifest = next(iter(manifests.values()), {})
    source_commit = any_manifest.get("source_commit", "unknown")

    out = [
        _frontmatter(
            license_id=license_id, splits=splits, pretty_name=pretty_name
        ),
        "",
        f"# {pretty_name}",
        "",
        "Lean 4 proof states from Mathlib, paired with the tactic applied at each",
        "step, in three representations extracted directly from the Lean kernel:",
        "",
        "1. **Source-faithful S-expressions** of the goal and every hypothesis, as",
        "   Lean elaborated them.",
        "2. **Normalized S-expressions** of the same state, with stable local-context",
        "   indices suitable for model input.",
        "3. **Annotated tactic syntax** -- the original tactic's syntax tree with",
        "   identifier leaves resolved to the constants and hypotheses they refer to.",
        "",
        "Row identity follows",
        "[`cat-searcher/leandojo-benchmark-4-random`](https://huggingface.co/datasets/cat-searcher/leandojo-benchmark-4-random),",
        "so `split` and `row_index` join directly against it and the `val`/`test`",
        "splits remain comparable to published results on that benchmark.",
        "",
        "Unlike pretty-printed proof states, S-expressions carry the elaborated term",
        "structure: implicit arguments, instance resolution, and binder structure are",
        "all explicit, which is what makes graph- and pointer-based models possible.",
        "",
        "## Loading",
        "",
        "```python",
        "from datasets import load_dataset",
        "import json",
        "",
        f'ds = load_dataset("{repo_id}", split="train")',
        "row = ds[0]",
        'print(row["model_goal_sexp"])',
        "",
        "# Nested payloads are JSON strings; see the schema note below.",
        'syntax_tree = json.loads(row["source_syntax"])',
        'context = json.loads(row["local_context"])',
        "```",
        "",
        "## Coverage",
        "",
        "**Read this before training on the `train` split.**",
        "",
        "| split | rows published | upstream rows | coverage |",
        "|---|---:|---:|---:|",
    ]
    for split, info in splits.items():
        upstream = upstream_sizes.get(split, 0)
        out.append(
            f"| `{split}` | {int(info['rows']):,} | {upstream:,} | "
            f"{_pct(int(info['rows']), upstream)} |"
        )
    out.append(
        f"| **total** | **{packed_total:,}** | **{upstream_total:,}** | "
        f"**{_pct(packed_total, upstream_total)}** |"
    )
    out += [
        "",
        "A row is published only when all three representations were extracted and",
        "validated for it. Rows are missing for two distinct reasons, and the",
        "difference matters:",
        "",
        "- **Environment limits** -- a source file that did not elaborate within the",
        "  timeout. These rows are absent for incidental reasons and are, in",
        "  principle, recoverable. Their absence is close to random.",
        "- **Extractor limits and row data** -- most importantly",
        "  `goal_cardinality`: this extractor records states with exactly one pending",
        "  goal, so every tactic invoked against multiple goals is excluded. Their",
        "  absence is **systematic**.",
        "",
        "### Known bias",
        "",
        "The `train` split is not a uniform random subsample of its upstream split.",
        "Because multi-goal invocations are excluded by construction, single-goal",
        "proof states are over-represented relative to the full benchmark, as are the",
        "tactics that tend to be applied to them. Any comparison against a model",
        "trained on all upstream rows should account for this.",
        "",
        "The `val` and `test` splits are near-complete, so evaluation numbers",
        "computed on them remain comparable with published results.",
        "",
        "### Per-split failure taxonomy",
        "",
    ]

    for split in splits:
        manifest = manifests.get(split)
        if not manifest:
            continue
        attempted = int(manifest.get("attempted_rows", 0))
        upstream = upstream_sizes.get(split, 0)
        if upstream and attempted and attempted != upstream:
            # A manifest is written per extraction run. If it attempted a
            # different number of rows than the split holds, it describes some
            # earlier partial run, and its phase counts are measured on a
            # sample of unknown shape. Publishing them as the split's taxonomy
            # would understate the missing rows by whatever factor the two
            # counts differ by, so the table is withheld instead.
            print(
                f"Warning: manifest for '{split}' attempted {attempted:,} rows "
                f"but the split has {upstream:,}; it is from an earlier run, so "
                "its failure taxonomy is withheld from the card."
            )
            out += [
                f"#### `{split}` -- per-phase breakdown unavailable",
                "",
                f"This split was extracted over several runs, and the manifest "
                f"left on disk covers only {attempted:,} of its {upstream:,} "
                "rows. Rather than report phase counts measured on that "
                "fraction as if they described the split, they are omitted. The "
                "aggregate coverage above is measured from the published rows "
                "themselves and is exact; the causes listed under Coverage all "
                "apply here, `goal_cardinality` and `file_compile` foremost, but "
                "their relative sizes for this split are not established.",
                "",
            ]
            continue
        phases = manifest.get("failure_phases") or {}
        if not phases:
            continue
        failed = int(manifest.get("failed_rows", 0))
        out += [
            f"#### `{split}` -- {failed:,} rows not published",
            "",
            "| phase | rows | kind | reason |",
            "|---|---:|---|---|",
        ]
        for phase, count in sorted(phases.items(), key=lambda kv: -int(kv[1])):
            kind, reason = PHASE_NOTES.get(phase, ("unclassified", ""))
            out.append(f"| `{phase}` | {int(count):,} | {kind} | {reason} |")
        out.append("")

    out += [
        "## Schema",
        "",
        "All three representations for a row ship in that same row, so nothing needs",
        "to be joined across files. Recursive payloads (`source_syntax`,",
        "`local_context`, the hypothesis lists) are stored as **JSON strings**: their",
        "shape varies per tactic, and a typed nested column would force either schema",
        "inference failures or a lowest-common-denominator union. Call `json.loads` on",
        "them.",
        "",
        "| column | description |",
        "|---|---|",
    ]
    for name, note in COLUMN_NOTES:
        out.append(f"| `{name}` | {note} |")

    out += [
        "",
        "## Provenance",
        "",
        f"- Mathlib commit: `{source_commit}`",
        "- Upstream dataset: `cat-searcher/leandojo-benchmark-4-random`",
        "- Extraction: each Mathlib source file is elaborated with a patched",
        "  [Pantograph](https://github.com/lenianiva/Pantograph) REPL, and the proof",
        "  state Lean records at each matched tactic invocation is captured. States",
        "  are read from Lean, never parsed back from pretty-printed text.",
    ]
    for split, manifest in manifests.items():
        extractor = manifest.get("extractor_version")
        if extractor:
            out.append(
                f"- `{split}`: extractor `{extractor}`, "
                f"raw schema v{manifest.get('schema_version')}, "
                f"Pantograph `{manifest.get('pantograph_commit', 'unknown')}`"
            )
    out += [
        "",
        "## Limitations",
        "",
        "- Single-goal states only; see Coverage.",
        "- `train` coverage is partial and biased; `val`/`test` are near-complete.",
        "- S-expressions are large. Expect the goal and hypothesis columns to",
        "  dominate memory; select only the representation you need.",
        "- Normalized and source-faithful S-expressions are *different*",
        "  representations of the same state, not alternatives of equal fidelity.",
        "  Train on `model_*`; use `raw_*` when you need exactly what Lean emitted.",
        "",
        "## License and attribution",
        "",
        f"This dataset is released under `{license_id}`. It is derived work, and",
        "both upstream licenses continue to apply to the material they cover:",
        "",
        "- **Mathlib 4** is licensed Apache-2.0. Theorem names, tactic text, file",
        "  paths, and every elaborated term in the S-expression columns derive from",
        f"  Mathlib at commit `{source_commit}`. See",
        "  <https://github.com/leanprover-community/mathlib4/blob/master/LICENSE>.",
        "- **LeanDojo** is licensed MIT. The row selection and the `split` /",
        "  `row_index` assignment are inherited unchanged from the LeanDojo",
        "  benchmark, so its notice is reproduced below as MIT requires.",
        "",
        "```",
        "MIT License",
        "",
        "Copyright (c) 2023 LeanDojo Team",
        "",
        "Permission is hereby granted, free of charge, to any person obtaining a copy",
        'of this software and associated documentation files (the "Software"), to deal',
        "in the Software without restriction, including without limitation the rights",
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell",
        "copies of the Software, and to permit persons to whom the Software is",
        "furnished to do so, subject to the following conditions:",
        "",
        "The above copyright notice and this permission notice shall be included in all",
        "copies or substantial portions of the Software.",
        "",
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR',
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,",
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE",
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER",
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,",
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE",
        "SOFTWARE.",
        "```",
        "",
        "## Citation",
        "",
        "Please cite both upstreams alongside this dataset: Mathlib for the",
        "mathematical content, and LeanDojo for the benchmark this dataset's row",
        "identity follows.",
        "",
    ]
    return "\n".join(out) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the Hugging Face dataset card, reading every coverage number "
            "from the extraction manifests and the pack report."
        )
    )
    parser.add_argument(
        "--packed-root",
        type=Path,
        required=True,
        help="Output root produced by pack_hf_dataset, holding pack_report.json.",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Prepared root holding sexpr_extraction/manifests/{split}.json.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hub dataset id, for example org/dataset-name.",
    )
    parser.add_argument(
        "--license",
        required=True,
        help=(
            "SPDX license identifier. This is derived work: check what Mathlib "
            "and the upstream benchmark require before choosing."
        ),
    )
    parser.add_argument(
        "--pretty-name",
        default="Mathlib Normalized S-Expressions",
        help="Human-readable dataset name shown on the Hub.",
    )
    parser.add_argument(
        "--upstream-sizes",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping split to upstream row count. Defaults to the "
            "attempted_rows recorded in each extraction manifest."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report_path = args.packed_root / "pack_report.json"
    if not report_path.is_file():
        print(f"No pack report at {report_path}; run pack_hf_dataset first.")
        return 1
    pack_report = _load(report_path)

    manifests: dict[str, dict] = {}
    manifest_dir = args.prepared_root / "sexpr_extraction" / "manifests"
    for split in pack_report["splits"]:
        path = manifest_dir / f"{split}.json"
        if path.is_file():
            manifests[split] = _load(path)
        else:
            print(f"Warning: no manifest for '{split}' at {path}; "
                  "its failure taxonomy will be omitted.")

    if args.upstream_sizes is not None:
        upstream_sizes = {
            str(key): int(value) for key, value in _load(args.upstream_sizes).items()
        }
    else:
        upstream_sizes = {
            split: int(manifest.get("attempted_rows", 0))
            for split, manifest in manifests.items()
        }
    missing = [s for s in pack_report["splits"] if not upstream_sizes.get(s)]
    if missing:
        print(
            "Warning: no upstream row count for "
            f"{', '.join(missing)}; coverage will read 'n/a'. Pass "
            "--upstream-sizes to supply them."
        )

    card = build_card(
        pack_report=pack_report,
        manifests=manifests,
        upstream_sizes=upstream_sizes,
        license_id=args.license,
        repo_id=args.repo_id,
        pretty_name=args.pretty_name,
    )
    card_path = args.packed_root / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"Wrote {card_path} ({len(card):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
