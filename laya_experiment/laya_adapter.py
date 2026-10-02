"""Small adapter around the optional Laya decision model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from maths_ai.gnn_inference.atp_lean_gnn.state import parse_state

from .tactic_choose import TACTIC_SET

DEFAULT_GLOSS = "Lean 4 tactic"
TACTIC_GLOSS: dict[str, str] = {
    ".": "move on to the next goal in the list",
    "apply": "apply a lemma or hypothesis to the goal",
    "assumption": "close the goal with a matching hypothesis",
    "cases": "destruct a value into its constructors",
    "constructor": "apply the goal's first constructor",
    "exact": "close the goal with an exact term",
    "intro": "introduce a binder or hypothesis",
    "linarith": "linear arithmetic reasoning",
    "nlinarith": "nonlinear arithmetic reasoning",
    "rfl": "close the goal by reflexivity",
    "rw": "rewrite with an equation",
    "simp": "simplify with a lemma set",
    "simpa": "simplify and then close with exact",
    "ring": "commutative ring normalization",
    "refine": "refine the goal with a partial term",
}


_LEAN_WORDS = {
    "⊢": "the goal is",
    "→": "implies",
    "->": "implies",
    "↔": "if and only if",
    "∧": "and",
    "∨": "or",
    "¬": "not",
    "∀": "for every",
    "∃": "there exists",
    "∈": "is an element of",
    "≤": "is less than or equal to",
    "≥": "is greater than or equal to",
    "≠": "is not equal to",
    "＝": "equals",
    "=": "equals",
}


def translate_lean_statement(statement: str) -> str:
    """Render common Lean symbols as short English phrases for Laya.

    This is intentionally deterministic and conservative: identifiers, theorem
    names, and type names are preserved, while only logical/connective syntax
    is expanded. The original Lean state should still be retained in datasets.
    """
    translated = statement.strip()
    for source, target in sorted(_LEAN_WORDS.items(), key=lambda item: -len(item[0])):
        translated = translated.replace(source, f" {target} ")
    translated = re.sub(r"\s+", " ", translated).strip()
    return translated


def translate_tactic_step(tactic_name: str, arguments: str = "") -> str:
    """Describe a tactic step in natural language for supervised Laya data."""
    tactic = tactic_name.strip()
    if tactic not in TACTIC_SET:
        raise ValueError(f"Unknown tactic family: {tactic_name!r}")
    gloss = TACTIC_GLOSS.get(tactic, f"apply the Lean tactic {tactic}")
    suffix = f" with arguments {arguments.strip()}" if arguments.strip() else ""
    return f"Use the Lean tactic '{tactic}' to {gloss}{suffix}."


def translate_lean_state(text_state: str) -> str:
    """Translate a proof state while keeping its goal before its context."""
    parsed = parse_state(text_state)
    context = "\n".join(
        f"Hypothesis {hypothesis.name}: {translate_lean_statement(hypothesis.type_expr)}"
        for hypothesis in parsed.hypotheses
    )
    goal = translate_lean_statement(parsed.goal)
    return f"GOAL: {goal}\nCONTEXT: {context}"


def laya_state(text_state: str) -> str:
    """Place the goal before local context so right truncation preserves it."""
    return translate_lean_state(text_state)


def choice_question(candidates: Sequence[str], instructions: str) -> dict[str, Any]:
    """Build the documented Laya choice-question payload."""
    if not candidates:
        raise ValueError("choice questions need at least one candidate")
    return {
        "tactic": {
            "type": "choice",
            "instructions": instructions,
            "criteria": {candidate: TACTIC_GLOSS.get(candidate, DEFAULT_GLOSS) for candidate in candidates},
        }
    }


def rotated_choice_question(
    candidates: Sequence[str], instructions: str, rotation: int
) -> dict[str, Any]:
    """Build a rotated question and retain the rotation metadata for auditing."""
    if not candidates:
        raise ValueError("choice questions need at least one candidate")
    offset = rotation % len(candidates)
    order = list(range(offset, len(candidates))) + list(range(offset))
    rotated = [candidates[index] for index in order]
    question = choice_question(rotated, instructions)
    question["tactic"]["option_order"] = order
    return question


class LayaUnavailableError(RuntimeError):
    """Raised when a real Laya run is requested without the optional package."""


class LayaAdapter:
    """Validated interface for real or fake Laya agents."""

    def __init__(self, agent: Any, instructions: str) -> None:
        self.agent = agent
        self.instructions = instructions

    @classmethod
    def load(cls, model_name: str, *, device: str, instructions: str) -> "LayaAdapter":
        try:
            import laya
        except ImportError as exc:
            raise LayaUnavailableError(
                "Laya is not installed. Install laya and the experiment dependencies before running reranking."
            ) from exc
        return cls(laya.load(model_name, device=device), instructions)

    def predict(self, text_state: str, candidates: Sequence[str]) -> dict[str, Any]:
        candidate_list = list(candidates)
        question = choice_question(candidate_list, self.instructions)
        result = self.agent.predict(laya_state(text_state), question)
        try:
            answer = result["answers"]["tactic"]
            probabilities = {str(key): float(value) for key, value in answer["probabilities"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Laya response does not contain tactic probabilities") from exc
        if set(probabilities) != set(candidate_list):
            raise ValueError("Laya probabilities are not exactly a permutation of the candidates")
        order = sorted(candidate_list, key=lambda candidate: (-probabilities[candidate], candidate))
        return {
            "order": order,
            "probabilities": probabilities,
            "choice": answer.get("choice"),
            "answer_confidence": float(answer.get("answer_confidence", 0.0)),
        }

    def balanced_predict(self, text_state: str, candidates: Sequence[str]) -> dict[str, Any]:
        candidate_list = list(candidates)
        totals = {candidate: 0.0 for candidate in candidate_list}
        for rotation in range(len(candidate_list)):
            question = rotated_choice_question(candidate_list, self.instructions, rotation)
            result = self.agent.predict(laya_state(text_state), question)
            probabilities = result["answers"]["tactic"]["probabilities"]
            if set(probabilities) != set(candidate_list):
                raise ValueError("Laya probabilities are not exactly a permutation of the candidates")
            for candidate in candidate_list:
                totals[candidate] += float(probabilities[candidate]) / len(candidate_list)
        order = sorted(candidate_list, key=lambda candidate: (-totals[candidate], candidate))
        return {"order": order, "probabilities": totals}
