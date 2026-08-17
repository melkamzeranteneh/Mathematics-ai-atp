from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from maths_ai.gnn_inference.atp_lean_gnn.argument_coverage import (
    ArgumentCoverageConfig,
    classify_argument_slots,
)


def _example(
    tactic_raw: str,
    tactic_name: str,
    *,
    node_indices: list[int],
    premise_mask: list[bool],
    lemma_ids: list[int] | None = None,
) -> Data:
    data = Data(
        x=torch.arange(len(premise_mask), dtype=torch.long),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    data.tactic_raw = tactic_raw
    data.tactic_name = tactic_name
    data.arg_node_indices = torch.tensor(node_indices, dtype=torch.long)
    data.arg_lemma_ids = torch.tensor(lemma_ids or [], dtype=torch.long)
    data.arg_count = len(node_indices)
    data.premise_mask = torch.tensor(premise_mask, dtype=torch.bool)
    data.row_index = 17
    data.theorem = "Demo.theorem"
    return data


class ArgumentCoverageClassificationTests(unittest.TestCase):
    def test_classifies_selectable_and_masked_local_targets(self) -> None:
        selectable, _ = classify_argument_slots(
            _example(
                "apply h",
                "apply",
                node_indices=[1],
                premise_mask=[False, True],
            ),
            split="val",
        )
        masked, _ = classify_argument_slots(
            _example(
                "apply h",
                "apply",
                node_indices=[1],
                premise_mask=[False, False],
            ),
            split="val",
        )

        self.assertEqual(selectable[0]["category"], "local_selectable")
        self.assertEqual(masked[0]["category"], "local_present_but_masked")

    def test_recovers_global_lemma_from_cached_id_or_corpus(self) -> None:
        cached, _ = classify_argument_slots(
            _example(
                "exact Mathlib.result",
                "exact",
                node_indices=[-1],
                lemma_ids=[42],
                premise_mask=[False],
            ),
            split="test",
        )
        recovered, _ = classify_argument_slots(
            _example(
                "exact Mathlib.result",
                "exact",
                node_indices=[-1],
                premise_mask=[False],
            ),
            split="test",
            lemma_names={"Mathlib.result"},
        )

        self.assertEqual(cached[0]["category"], "global_library_lemma")
        self.assertEqual(recovered[0]["category"], "global_library_lemma")

    def test_classifies_missing_fresh_literal_and_qualified_targets(self) -> None:
        cases = [
            ("apply", "apply", "parser_missing_argument"),
            ("intro h", "intro", "fresh_identifier"),
            ("exact 42", "exact", "literal"),
            ("apply Foo.bar", "apply", "unresolved_qualified_name"),
        ]
        for tactic_raw, tactic_name, expected in cases:
            with self.subTest(tactic=tactic_raw):
                records, _ = classify_argument_slots(
                    _example(
                        tactic_raw,
                        tactic_name,
                        node_indices=[-1],
                        premise_mask=[False],
                    ),
                    split="train",
                )
                self.assertEqual(records[0]["category"], expected)

    def test_marks_excess_parser_tokens_as_semantically_ambiguous(self) -> None:
        records, row = classify_argument_slots(
            _example(
                "exact f x",
                "exact",
                node_indices=[0, -1],
                premise_mask=[True],
            ),
            split="val",
        )

        self.assertEqual(row["expected_arity"], 1)
        self.assertEqual(row["parsed_argument_count"], 2)
        self.assertEqual(row["ignored_parsed_token_count"], 1)
        self.assertEqual(row["parse_shape"], "excess_tokens_or_compound_expression")
        self.assertTrue(records[0]["semantically_ambiguous_parse"])

    def test_only_first_let_slot_is_a_fresh_identifier(self) -> None:
        records, _ = classify_argument_slots(
            _example(
                "let result value",
                "let",
                node_indices=[-1, -1],
                premise_mask=[False],
            ),
            split="train",
        )

        self.assertEqual(
            [record["category"] for record in records],
            ["fresh_identifier", "unmapped_token"],
        )

    def test_rejects_two_lemma_name_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "either lemma_corpus or lemma_index"):
            ArgumentCoverageConfig(
                prepared_root=Path("prepared"),
                output_dir=Path("output"),
                lemma_corpus=Path("lemmas.jsonl"),
                lemma_index=Path("index"),
            ).normalized()


if __name__ == "__main__":
    unittest.main()
