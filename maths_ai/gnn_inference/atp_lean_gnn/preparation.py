from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .dataset import DatasetRow
from .graph import DAGBuilder, proof_state_to_dag
from .labels import label_example
from .state import ProofState, parse_state


PREPARATION_PHASES = ("parse_state", "proof_state_to_dag", "label_example")


@dataclass(frozen=True)
class PreparedExample:
    row: DatasetRow
    parsed_state: ProofState
    dag: DAGBuilder
    tactic_name: str


class PreparationPhaseError(Exception):
    def __init__(self, *, phase: str, cause: Exception):
        self.phase = phase
        self.cause = cause
        super().__init__(str(cause))


class SExprCache:
    """Disk cache for S-expressions with on-demand generation via proof replay."""

    def __init__(
        self,
        output_root: Path,
        project_path: str,
        enabled: bool = True,
    ):
        self.output_root = Path(output_root)
        self.project_path = project_path
        self.enabled = enabled

    def _sexpr_dir(self, split: str) -> Path:
        d = self.output_root / split / "sexpr"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load(self, split: str, row_index: int) -> Optional[dict]:
        sexpr_file = self._sexpr_dir(split) / f"{row_index:09d}.json"
        if sexpr_file.exists():
            try:
                with open(sexpr_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def save(self, split: str, row_index: int, data: dict) -> None:
        sexpr_file = self._sexpr_dir(split) / f"{row_index:09d}.json"
        with open(sexpr_file, "w") as f:
            json.dump(data, f)


def _fix_pp_for_goal_start(pp: str) -> str:
    """Preprocess a pretty-printed Lean 4 type so goal_start can parse it.

    Fixes:
    - Universe variables: ``Type u_1`` / ``Type u₁`` → ``Type``
    - Sort expressions: ``Sort (+ u_1 1)`` → ``Type``
    - Grouped binders: ``{a b : T}`` → ``{a : T} {b : T}``
    - Universe-polymorphic instances: ``Category.{v_1, u_1} C`` → ``Category C``
    """
    s = pp.strip()

    # Subscript unicode digits: u₁ → u1
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    s = s.translate(subscript_map)

    # Type u_1, Type u1, Sort u_1 etc → Type
    s = re.sub(r"(?:Type|Sort)\s+u_?\d+", "Type", s)
    s = re.sub(r"Sort\s*\(\+\s*u_?\d+\s+\d+\)", "Type", s)

    # Universe-polymorphic instances: Category.{v_1, u_1} C → Category C
    s = re.sub(r"\.\{[^}]*\}", "", s)

    def _split_grouped(m: re.Match) -> str:
        content = m.group(1).strip()
        close = m.group(0)[0]
        other_close = "}" if close == "{" else ")"
        if " : " in content:
            names_part, type_part = content.split(" : ", 1)
        elif ":" in content:
            names_part, type_part = content.split(":", 1)
        else:
            return m.group(0)
        names = names_part.split()
        if len(names) <= 1:
            return m.group(0)
        return " ".join(
            f"{close}{n} : {type_part.strip()}{other_close}" for n in names
        )

    s = re.sub(r"\{([^{}]+)\}", _split_grouped, s)
    s = re.sub(r"\(([^()]+)\)", _split_grouped, s)
    return s


def _extract_binder_names_from_pp(pp_type: str) -> list[str]:
    """Extract ordered binder names from a pretty-printed Lean 4 forall type.

    Handles implicit ``{x : T}``, explicit ``(x : T)``, and instance
    ``[C]`` binders.  Grouped binders like ``{a b : T}`` must already be
    split (use ``_fix_pp_for_goal_start`` first).
    """
    s = pp_type.strip()
    if s.startswith("∀"):
        s = s[1:].lstrip()
    elif s.startswith("forall"):
        s = s[5:].lstrip()

    names: list[str] = []
    idx = 0
    while idx < len(s):
        c = s[idx]
        if c in ("{", "("):
            close = "}" if c == "{" else ")"
            depth = 1
            start = idx + 1
            idx += 1
            while idx < len(s) and depth > 0:
                if s[idx] == c:
                    depth += 1
                elif s[idx] == close:
                    depth -= 1
                idx += 1
            if depth == 0:
                inner = s[start : idx - 1].strip()
                if inner.startswith("["):
                    names.append("inst")
                elif " : " in inner:
                    before_colon = inner.split(" : ", 1)[0].strip()
                    for n in before_colon.split():
                        n = n.strip()
                        if n:
                            names.append(n)
                elif ":" in inner:
                    before_colon = inner.split(":", 1)[0].strip()
                    for n in before_colon.split():
                        n = n.strip()
                        if n:
                            names.append(n)
        elif c == "[":
            depth = 1
            idx += 1
            while idx < len(s) and depth > 0:
                if s[idx] == "[":
                    depth += 1
                elif s[idx] == "]":
                    depth -= 1
                idx += 1
            names.append("inst")
        elif c in (",", "\u2192", "\u27f6"):
            idx += 1
        elif c in ("\n", " "):
            idx += 1
        else:
            break

    return names


async def _replay_one_theorem(server, full_name: str, tactics: list[str]) -> list[dict]:
    """Replay a single theorem on an existing server (no server create/shutdown)."""
    from .graph import goal_state_to_proof_state

    results: list[dict] = []
    try:
        inspect_result = await server.env_inspect_async(full_name)
        pp_type = inspect_result["type"]["pp"]
        fixed_pp = _fix_pp_for_goal_start(pp_type)
        binder_names = _extract_binder_names_from_pp(fixed_pp)

        gs = await server.goal_start_async(fixed_pp)

        for bname in binder_names:
            try:
                gs = await server.goal_tactic_async(gs, f"intro {bname}")
            except Exception:
                try:
                    gs = await server.goal_tactic_async(gs, "intro")
                except Exception:
                    break

        text, hyp_sexps, goal_sexp = goal_state_to_proof_state(gs)
        results.append({
            "goal_sexp": goal_sexp,
            "hyp_sexps": [{"name": name, "sexp": sexp} for name, sexp in hyp_sexps],
            "text_state": text,
        })

        for tactic in tactics:
            try:
                gs = await server.goal_tactic_async(gs, tactic)
            except Exception:
                break

            text, hyp_sexps, goal_sexp = goal_state_to_proof_state(gs)
            if not goal_sexp and not hyp_sexps:
                break
            results.append({
                "goal_sexp": goal_sexp,
                "hyp_sexps": [{"name": name, "sexp": sexp} for name, sexp in hyp_sexps],
                "text_state": text,
            })
    except Exception:
        pass

    return results


async def _replay_batch(
    project_path: str,
    theorems: dict[str, list[str]],
    imports: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Replay multiple theorems on a SINGLE server.

    Args:
        project_path: Path to the Lean project.
        theorems: Dict mapping theorem name → list of tactics.
        imports: Lean imports (default: ``["Init", "Mathlib"]``).

    Returns:
        Dict mapping theorem name → list of S-expression dicts.
    """
    from pantograph.server import Server
    from .graph import patch_pantograph_for_sexp

    patch_pantograph_for_sexp()

    if imports is None:
        imports = ["Init", "Mathlib"]

    server = await Server.create(
        project_path=project_path,
        imports=imports,
        options={"printExprAST": True},
    )

    all_results: dict[str, list[dict]] = {}
    try:
        for full_name, tactics in theorems.items():
            results = await _replay_one_theorem(server, full_name, tactics)
            if results:
                all_results[full_name] = results
    finally:
        try:
            await server.shutdown_async()
        except Exception:
            pass

    return all_results


def replay_batch_sexpr(
    project_path: str,
    theorems: dict[str, list[str]],
    imports: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Synchronous wrapper: replay multiple theorems on a single server.

    Args:
        project_path: Path to the Lean project.
        theorems: Dict mapping theorem name → list of tactics.

    Returns:
        Dict mapping theorem name → list of S-expression dicts.
    """
    async def _run():
        return await _replay_batch(project_path, theorems, imports)

    return asyncio.run(_run())


def replay_theorem_sexpr(
    project_path: str,
    full_name: str,
    state_tactic_pairs: list[tuple[str, str]],
    imports: list[str] | None = None,
) -> list[dict]:
    """Synchronous wrapper for single theorem replay.

    Args:
        project_path: Path to the Lean project.
        full_name: Fully qualified theorem name.
        state_tactic_pairs: List of ``(state_str, tactic_str)`` pairs.

    Returns:
        List of S-expression dicts, one per proof step.
    """
    tactics = [tactic for _, tactic in state_tactic_pairs]
    result = replay_batch_sexpr(project_path, {full_name: tactics}, imports)
    return result.get(full_name, [])


def prepare_example(
    row: DatasetRow,
    *,
    sexpr_cache: Optional[SExprCache] = None,
    sexpr_data: Optional[dict] = None,
    use_sexpr: bool = True,
) -> tuple:
    """Prepare a single example, returning (dag, tactic_name).

    If ``sexpr_data`` is provided, uses it for S-expression DAG construction.
    Otherwise falls back to the text parser.
    """
    parsed_state = parse_state(row.state)

    goal_sexp = None
    hyp_sexps = None

    if use_sexpr and sexpr_data is not None:
        goal_sexp = sexpr_data.get("goal_sexp")
        hyp_sexps = [
            (item["name"], item["sexp"]) for item in sexpr_data.get("hyp_sexps", [])
        ]
    elif use_sexpr and sexpr_cache is not None and sexpr_cache.enabled:
        cached = sexpr_cache.load(row.split, row.row_index)
        if cached is not None:
            goal_sexp = cached.get("goal_sexp")
            hyp_sexps = [
                (item["name"], item["sexp"]) for item in cached.get("hyp_sexps", [])
            ]

    dag = proof_state_to_dag(
        row.state,
        goal_sexp=goal_sexp,
        hyp_sexps=hyp_sexps,
    )

    label_info = label_example(row.tactic)
    tactic_name = str(label_info["tactic_name"])

    return dag, tactic_name
