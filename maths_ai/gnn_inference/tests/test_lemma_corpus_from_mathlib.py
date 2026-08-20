"""Tests for the Mathlib-environment lemma corpus extractor."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Iterator, Sequence
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus
from maths_ai.gnn_inference.scripts.extract_lemma_corpus_from_mathlib import (
    extract_corpus,
    select_catalog_names,
)


class _FakeEnvironmentClient:
    """Stand-in for the Pantograph REPL.

    ``inspect_many`` yields one reply per requested name in request order,
    matching the real client's only means of pairing a reply with its request,
    and records what it was asked so tests can assert that ordering.
    """

    def __init__(
        self,
        catalog: Sequence[str],
        replies: dict[str, dict[str, object]],
    ) -> None:
        self._catalog = list(catalog)
        self._replies = dict(replies)
        self.requested: list[str] = []
        self.started = False
        self.closed = False

    def start(self) -> "_FakeEnvironmentClient":
        self.started = True
        return self

    def catalog(self) -> list[str]:
        return list(self._catalog)

    def inspect_many(self, names: Sequence[str]) -> Iterator[dict[str, object]]:
        for name in names:
            self.requested.append(name)
            yield self._replies.get(name, {"error": "unknown constant"})

    def close(self) -> None:
        self.closed = True


def _theorem(pp: str, module: str, **extra: object) -> dict[str, object]:
    return {"type": {"pp": pp}, "module": module, "isUnsafe": False, **extra}


class SelectCatalogNamesTests(unittest.TestCase):
    def test_kind_tag_is_stripped_and_filtered(self) -> None:
        catalog = ["tmul_comm", "dFunction.comp", "cOr.inl"]

        self.assertEqual(select_catalog_names(catalog, "t"), ["mul_comm"])
        self.assertEqual(
            select_catalog_names(catalog, "tc"), ["mul_comm", "Or.inl"]
        )

    def test_duplicate_names_across_tags_are_emitted_once(self) -> None:
        # The corpus is keyed by name, so a constant listed under two tags must
        # not produce two records competing for the same key.
        self.assertEqual(select_catalog_names(["tfoo", "dfoo"], "td"), ["foo"])

    def test_unknown_kind_tag_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "z"):
            select_catalog_names(["tfoo"], "tz")


class ExtractCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._temporary.name) / "corpus"
        self.catalog = ["tmul_comm", "dFunction.comp", "cOr.inl", "tunsafe_lemma"]
        self.replies = {
            "mul_comm": _theorem(
                "∀ {G : Type u_1} [inst : CommMagma G] (a b : G), a * b = b * a",
                "Mathlib.Algebra.Group.Defs",
            ),
            "Function.comp": _theorem(
                "{α : Sort u} → {β : Sort v} → (β → γ) → (α → β) → α → γ",
                "Init.Prelude",
            ),
            "Or.inl": _theorem(
                "∀ {a b : Prop}, a → a ∨ b",
                "Init.Prelude",
                constructorInfo={"numParams": 2, "numFields": 1, "induct": "Or", "cidx": 0},
            ),
            "unsafe_lemma": {
                "type": {"pp": "True"},
                "module": "Mathlib.Unsafe",
                "isUnsafe": True,
            },
        }

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _run(self, **kwargs: object):
        client = _FakeEnvironmentClient(self.catalog, self.replies)
        result = extract_corpus(
            output_dir=self.output_dir,
            client_factory=lambda: client,
            progress=lambda _message: None,
            **kwargs,
        )
        return result, client

    def test_corpus_records_carry_the_inspected_type_and_module(self) -> None:
        result, client = self._run()

        self.assertTrue(client.started)
        self.assertTrue(client.closed)
        self.assertEqual(result.catalog_size, 4)
        self.assertEqual(result.selected_names, 4)
        self.assertEqual(result.records, 3)

        records = {record.name: record for record in load_lemma_corpus(self.output_dir)}
        self.assertEqual(set(records), {"mul_comm", "Function.comp", "Or.inl"})
        self.assertEqual(
            records["mul_comm"].statement,
            "∀ {G : Type u_1} [inst : CommMagma G] (a b : G), a * b = b * a",
        )
        self.assertEqual(records["mul_comm"].module, "Mathlib.Algebra.Group.Defs")
        self.assertEqual(records["mul_comm"].namespace, "")
        self.assertEqual(records["Or.inl"].namespace, "Or")
        self.assertEqual(
            sorted(record.lemma_id for record in records.values()), [0, 1, 2]
        )

    def test_replies_are_paired_with_requests_by_order(self) -> None:
        _, client = self._run()

        self.assertEqual(
            client.requested,
            ["mul_comm", "Function.comp", "Or.inl", "unsafe_lemma"],
        )

    def test_unsafe_declarations_are_skipped_but_counted(self) -> None:
        result, _ = self._run()

        self.assertEqual(result.unsafe_skipped, 1)
        names = {record.name for record in load_lemma_corpus(self.output_dir)}
        self.assertNotIn("unsafe_lemma", names)

    def test_unsafe_declarations_can_be_kept(self) -> None:
        result, _ = self._run(include_unsafe=True)

        self.assertEqual(result.unsafe_skipped, 0)
        self.assertEqual(result.records, 4)

    def test_kind_filter_restricts_the_corpus(self) -> None:
        result, client = self._run(kinds="c")

        self.assertEqual(result.selected_names, 1)
        self.assertEqual(client.requested, ["Or.inl"])

    def test_unreadable_declaration_is_recorded_rather_than_fatal(self) -> None:
        # A constant the REPL rejects must not abort a run of hundreds of
        # thousands of names, and must not silently vanish either.
        self.replies.pop("Function.comp")

        result, _ = self._run()

        self.assertEqual(result.records, 2)
        self.assertEqual(result.inspect_failures, 1)
        failures = [
            json.loads(line)
            for line in (self.output_dir / "failures.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(
            [(entry["name"], entry["reason"]) for entry in failures],
            [("Function.comp", "inspect_error")],
        )

    def test_names_only_mode_writes_the_index_and_never_inspects(self) -> None:
        result, client = self._run(names_only=True)

        self.assertTrue(result.names_only)
        self.assertEqual(result.records, 0)
        self.assertEqual(client.requested, [])
        self.assertFalse((self.output_dir / "lemmas.jsonl").exists())

        names = json.loads((self.output_dir / "lemma_names.json").read_text(encoding="utf-8"))
        self.assertEqual(
            names, ["mul_comm", "Function.comp", "Or.inl", "unsafe_lemma"]
        )

    def test_existing_corpus_is_not_overwritten_without_force(self) -> None:
        self._run()

        with self.assertRaisesRegex(FileExistsError, "lemmas.jsonl"):
            self._run()

        result, _ = self._run(force=True)
        self.assertEqual(result.records, 3)

    def test_manifest_records_the_provenance_of_the_statements(self) -> None:
        self._run(limit=2)

        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"], "mathlib-environment")
        self.assertEqual(manifest["statement_source"], "env.inspect type.pp")
        self.assertEqual(manifest["limit"], 2)
        self.assertEqual(manifest["selected_names"], 2)
        self.assertEqual(manifest["records"], 2)


if __name__ == "__main__":
    unittest.main()
