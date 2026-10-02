"""Paired ranking metrics for the GNN pool experiment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def rank_of(target: str, ranking: Sequence[str]) -> int | None:
    try:
        return list(ranking).index(target) + 1
    except ValueError:
        return None


def evaluate_rankings(
    rankings: Sequence[Sequence[str]],
    targets: Sequence[str],
    mask: Sequence[bool],
    *,
    ks: Sequence[int] = (10, 5, 1),
    mrr_k: int = 10,
) -> dict[str, Any]:
    indices = [index for index, keep in enumerate(mask) if keep]
    result: dict[str, Any] = {"n": len(indices)}
    if not indices:
        return result
    for k in ks:
        result[f"recall@{k}"] = sum(targets[i] in rankings[i][:k] for i in indices) / len(indices)
    reciprocal_ranks = []
    for index in indices:
        rank = rank_of(targets[index], rankings[index][:mrr_k])
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    result[f"mrr@{mrr_k}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return result


def transition_table(
    reference: Sequence[Sequence[str]],
    candidate: Sequence[Sequence[str]],
    targets: Sequence[str],
    mask: Sequence[bool],
    *,
    k: int,
) -> dict[str, float | int]:
    indices = [index for index, keep in enumerate(mask) if keep]
    total = len(indices)
    if not total:
        return {"both right": 0.0, "only GNN right": 0.0, "only Laya right": 0.0, "neither": 0.0, "n": 0}
    reference_ok = [targets[i] in reference[i][:k] for i in indices]
    candidate_ok = [targets[i] in candidate[i][:k] for i in indices]
    return {
        "both right": sum(a and b for a, b in zip(reference_ok, candidate_ok)) / total,
        "only GNN right": sum(a and not b for a, b in zip(reference_ok, candidate_ok)) / total,
        "only Laya right": sum((not a) and b for a, b in zip(reference_ok, candidate_ok)) / total,
        "neither": sum((not a) and (not b) for a, b in zip(reference_ok, candidate_ok)) / total,
        "n": total,
    }
