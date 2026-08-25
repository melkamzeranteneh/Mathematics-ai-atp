"""Request/response schemas for the maths_ai HTTP API.

The response models deliberately re-export the domain models already used
across the pipeline (``TacticCandidate``, ``STV``, ``RankedSubgoal``) instead
of duplicating them: API consumers see exactly the shapes the inference
components exchange internally.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from maths_ai.data_models.proof_components import STV, TacticCandidate


def _strip_or_raise(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be a non-empty string")
    return stripped


class ProveRequest(BaseModel):
    """Request body for ``POST /prove``."""

    goal: str = Field(
        description="Lean goal expression to prove (the text after `⊢`).",
        examples=["forall (p q : Prop), Or p q -> Or q p"],
    )
    hypotheses: List[str] = Field(
        default_factory=list,
        description="Local context entries, each rendered as `name : type`.",
        examples=[["h : p ∨ q"]],
    )
    max_depth: Optional[int] = Field(
        default=None,
        ge=1,
        description="Per-request override of the search branch-depth bound.",
    )
    max_nodes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Per-request override of the total hypergraph-size bound.",
    )

    @field_validator("goal", mode="after")
    @classmethod
    def _validate_goal(cls, value: str) -> str:
        return _strip_or_raise(value, "goal")

    @field_validator("hypotheses", mode="after")
    @classmethod
    def _validate_hypotheses(cls, value: List[str]) -> List[str]:
        return [_strip_or_raise(hypothesis, "hypotheses") for hypothesis in value]


class ProofTraceNode(BaseModel):
    """One step of a witnessing proof, as a tree.

    Mirrors ``ProofHypergraph.proof_trace()``: each node carries the goal it
    closes and the tactic (with arguments) that closed it; ``subgoals`` holds
    one fully-recursed entry per remaining proof obligation (empty ⇒ the
    tactic fully discharged the goal). The flattened pre-order sequence of
    ``tactic + arguments`` replays as a Lean tactic block, which is what makes
    the trace independently verifiable.
    """

    goal: str
    tactic: str
    arguments: List[str] = Field(default_factory=list)
    subgoals: List["ProofTraceNode"] = Field(default_factory=list)


ProofTraceNode.model_rebuild()


class ProveResponse(BaseModel):
    """Response body for ``POST /prove``.

    ``proof_trace`` is ``None`` unless ``solved`` is true; ``summary``
    carries the full hypergraph dump (nodes/edges with statuses, ranks and
    STVs) for inspection when the search ended unsolved or hit its budget.
    """

    solved: bool
    exhausted: bool
    proof_trace: Optional[ProofTraceNode] = None
    summary: Dict[str, Any]

    @classmethod
    def from_graph(cls, graph: Any) -> "ProveResponse":
        trace = graph.proof_trace()
        return cls(
            solved=graph.is_solved(),
            exhausted=graph.is_exhausted(),
            proof_trace=ProofTraceNode.model_validate(trace) if trace is not None else None,
            summary=graph.summary(),
        )


class PredictTacticRequest(BaseModel):
    """Request body for ``POST /gnn/predict_tactic``."""

    goal: str = Field(description="Lean goal expression to predict tactics for.")
    top_k: int = Field(default=3, ge=1, le=20)

    @field_validator("goal", mode="after")
    @classmethod
    def _validate_goal(cls, value: str) -> str:
        return _strip_or_raise(value, "goal")


class PredictTacticResponse(BaseModel):
    """Up to ``top_k`` ranked tactic candidates from the GNN engine.

    An empty ``candidates`` list means the GNN found no viable tactic for the
    goal — callers must treat that goal as a dead branch, mirroring
    ``HybridReasoner.predict_next_tactic``'s contract.
    """

    goal: str
    candidates: List[TacticCandidate]


class EvaluateRequest(BaseModel):
    """Request body for ``POST /pln/evaluate``."""

    expression: str = Field(description="Lean target formula to score.")
    hypotheses: List[str] = Field(
        default_factory=list,
        description="Local-context formulas asserted into the PLN knowledge base.",
    )

    @field_validator("expression", mode="after")
    @classmethod
    def _validate_expression(cls, value: str) -> str:
        return _strip_or_raise(value, "expression")


class EvaluateResponse(BaseModel):
    """STV score for one expression, from PeTTaChainer/PLN.

    ``score`` is strength × confidence. ``is_fallback`` marks scores that are
    *not* real PLN results (exploration samples) — they must not be read as
    evidence of provability.
    """

    expression: str
    stv: STV
    score: float
    status: str
    is_fallback: bool


class HealthResponse(BaseModel):
    """Liveness/readiness payload for ``GET /health``."""

    status: str
    ready: bool
    error: Optional[str] = None
