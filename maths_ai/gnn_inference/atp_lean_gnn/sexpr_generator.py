from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from maths_ai.gnn_inference.atp_lean_gnn.graph import (
    patch_pantograph_for_sexp,
    goal_state_to_proof_state,
)
from pantograph.server import Server


class SExprGenerator:
    """Generates S-expressions for Lean proof states using Pantograph."""

    def __init__(self, project_path: str = "maths_ai/lean_mathlib", imports: list[str] | None = None):
        self.project_path = project_path
        self.imports = imports or ["Init", "Mathlib"]
        self._server: Server | None = None
        self._server_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the Pantograph server."""
        from pantograph.server import Server
        patch_pantograph_for_sexp()
        self._server = await Server.create(
            project_path=self.project_path,
            imports=self.imports,
            options={"printExprAST": True},
        )

    async def close(self) -> None:
        """Close the Pantograph server."""
        if self._server is not None:
            self._server._close()

    async def generate_sexpr(self, state_str: str) -> dict | None:
        """
        Generate S-expressions for a proof state string.
        
        Args:
            state_str: The proof state string (e.g., "h : P → Q ⊢ Q")
            
        Returns:
            Dictionary with goal_sexp, hyp_sexps, text_state or None if failed
        """
        if self._server is None:
            raise RuntimeError("SExprGenerator not started. Call start() first.")
        
        try:
            # Parse the state string to extract hypotheses and goal
            lines = state_str.split('\n')
            hypotheses = []
            goal = ""
            for line in lines:
                line = line.strip()
                if line.startswith('⊢'):
                    goal = line[1:].strip()
                elif ' : ' in line:
                    name, typ = line.split(':', 1)
                    hypotheses.append((name.strip(), typ.strip()))
                elif ':' in line and ' : ' not in line:
                    name, typ = line.split(':', 1)
                    hypotheses.append((name.strip(), typ.strip()))

            if not goal:
                return None

            # Start with just the goal
            goal_state = await self._server.goal_start_async(goal)

            # Introduce hypotheses
            if hypotheses:
                names = " ".join(name for name, _ in hypotheses)
                goal_state = await self._server.goal_tactic_async(goal_state, f"intro {names}")

            # Extract S-expressions
            text_state, hyp_sexps, goal_sexp = goal_state_to_proof_state(goal_state)

            return {
                "goal_sexp": goal_sexp,
                "hyp_sexps": [{"name": name, "sexp": sexp} for name, sexp in hyp_sexps],
                "text_state": text_state,
            }
        except Exception as e:
            print(f"Failed to generate S-expr: {e}")
            return None


class SExprCache:
    """Manages on-disk caching of S-expressions."""
    
    def __init__(self, prepared_root: Path):
        self.prepared_root = Path(prepared_root)

    def _sexpr_path(self, split: str, row_index: int) -> Path:
        return self.prepared_root / split / "sexpr" / f"{row_index:09d}.json"

    def load(self, split: str, row_index: int) -> dict | None:
        """Load cached S-expressions."""
        sexpr_file = self.prepared_root / split / "sexpr" / f"{row_index:09d}.json"
        if not sexpr_file.exists():
            return None
        try:
            with open(sexpr_file) as f:
                return json.load(f)
        except Exception:
            return None

    def save(self, split: str, row_index: int, data: dict) -> None:
        """Save S-expressions to cache."""
        sexpr_dir = self.prepared_root / split / "sexpr"
        sexpr_dir.mkdir(parents=True, exist_ok=True)
        sexpr_file = sexpr_dir / f"{row_index:09d}.json"
        with open(sexpr_file, "w") as f:
            json.dump(data, f)

    def has(self, split: str, row_index: int) -> bool:
        """Check if S-expressions are cached."""
        return self._sexpr_path(split, row_index).exists()


async def generate_sexpr_for_row(
    row,
    generator: "SExprGenerator",
    cache: "SExprCache",
    split: str,
    row_index: int,
    force: bool = False,
) -> dict | None:
    """Generate S-expressions for a single row, using cache if available."""
    if generator._server is None:
        return None
    
    cache = SExprCache(Path("maths_ai/gnn_inference/artifacts/prepared/v1"))
    
    if not force and cache.has(row.split, row.row_index):
        return cache.load(row.split, row.row_index)
    
    result = await generate_sexpr_for_state(row.state, generator)
    if result is not None:
        cache.save(row.split, row.row_index, result)
    return result


async def generate_sexpr_for_state(state_str: str, generator: "SExprGenerator") -> dict | None:
    """Generate S-expressions for a proof state string."""
    if generator._server is None:
        return None
    
    try:
        # Parse the state string
        lines = state_str.split('\n')
        hypotheses = []
        goal = ""
        for line in lines:
            line = line.strip()
            if line.startswith('⊢'):
                goal = line[1:].strip()
            elif ' : ' in line:
                name, typ = line.split(':', 1)
                hypotheses.append((name.strip(), typ.strip()))
            elif ':' in line and ' : ' not in line:
                name, typ = line.split(':', 1)
                hypotheses.append((name.strip(), typ.strip()))
        
        if not goal:
            return None

        # Start with just the goal
        goal_state = await generator._server.goal_start_async(goal)

        # Introduce hypotheses
        if hypotheses:
            names = " ".join(name for name, _ in hypotheses)
            goal_state = await generator._server.goal_tactic_async(goal_state, f"intro {names}")

        # Extract S-expressions
        from maths_ai.gnn_inference.atp_lean_gnn.graph import goal_state_to_proof_state
        text_state, hyp_sexps, goal_sexp = goal_state_to_proof_state(goal_state)

        return {
            "goal_sexp": goal_sexp,
            "hyp_sexps": [{"name": name, "sexp": sexp} for name, sexp in hyp_sexps],
            "text_state": text_state,
        }
    except Exception as e:
        print(f"Failed to generate S-expr: {e}")
        return None


class SExprManager:
    """High-level manager for S-expression generation with caching."""
    
    def __init__(self, prepared_root: Path, project_path: str = "maths_ai/lean_mathlib"):
        self.prepared_root = Path(prepared_root)
        self.generator = SExprGenerator(project_path=project_path)
        self.cache = SExprCache(Path("maths_ai/gnn_inference/artifacts/prepared/v1"))
    
    async def start(self):
        await self.generator.start()
    
    async def close(self):
        await self.generator.close()
    
    async def get_sexpr_for_row(self, row, split: str, row_index: int, force: bool = False) -> tuple[str | None, list | None, str | None]:
        """Get S-expressions for a row, using cache if available."""
        if self.generator._server is None:
            return None, None, None
        
        cache = SExprCache(Path("maths_ai/gnn_inference/artifacts/prepared/v1"))
        
        if not force and self.cache.has(split, row.row_index):
            cached = self.cache.load(split, row.row_index)
            if cached:
                return cached.get("goal_sexp"), cached.get("hyp_sexps"), cached.get("text_state")
        
        result = await generate_sexpr_for_state(row.state, self.generator)
        if result is not None:
            self.cache.save(split, row.row_index, result)
            return result.get("goal_sexp"), result.get("hyp_sexps"), result.get("text_state")
        return None, None, None
    
    async def close(self):
        await self.generator.close()