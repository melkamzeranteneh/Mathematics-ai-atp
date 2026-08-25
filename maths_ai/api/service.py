"""Service layer for the maths_ai HTTP API.

``InferenceService`` wraps a fully-constructed ``HybridReasoner`` (GNN engine,
PLN chainer and Pantograph executor all loaded once at startup) and adds the
concurrency machinery the HTTP surface needs:

* a single shared Pantograph ``Server`` process — proof searches are
  serialized behind an asyncio lock because the server's goal-state handles
  and the DTS sampler are stateful and not safe to interleave. Individual
  GNN/PLN calls are read-only with respect to that state and run in worker
  threads instead;
* concurrency-safe access to the DTS state file: every persist/load goes
  through its own lock, so concurrent ``/prove`` requests can't corrupt it.

The heavy imports (torch, pantograph) live inside ``build_default_service``
so the rest of the package stays importable on machines without them (tests
inject fakes; docs generation imports the app).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from maths_ai.core.config import settings
from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.pln_inference.model import PLNInference, PLNResult

logger = logging.getLogger(__name__)


@dataclass
class InferenceService:
    """Owns the loaded inference components plus the API-level locks."""

    reasoner: Any  # HybridReasoner; typed loosely so fakes can stand in
    dts_state_path: Path = field(default_factory=lambda: settings.dts_state_file)
    prove_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    dts_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def prove(
        self,
        goal: str,
        hypotheses: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        max_nodes: Optional[int] = None,
    ):
        """Run one full hybrid proof search, serially.

        Per-request depth/node-budget overrides are applied inside the lock
        and restored afterwards — safe because searches never overlap.
        """
        async with self.prove_lock:
            saved = (self.reasoner.max_depth, self.reasoner.max_nodes)
            try:
                if max_depth is not None:
                    self.reasoner.max_depth = max_depth
                if max_nodes is not None:
                    self.reasoner.max_nodes = max_nodes
                graph = await self.reasoner.prove(goal, hypotheses=hypotheses)
            finally:
                self.reasoner.max_depth, self.reasoner.max_nodes = saved
        await self.persist_dts_state()
        return graph

    async def predict_tactics(self, goal: str, top_k: int) -> List[TacticCandidate]:
        return await asyncio.to_thread(self.reasoner.gnn_engine.inference, goal, top_k=top_k)

    async def evaluate(self, expression: str, hypotheses: Optional[List[str]] = None) -> PLNResult:
        return await asyncio.to_thread(self.reasoner.petta_chainer.evaluate, expression, hypotheses=hypotheses)

    async def persist_dts_state(self) -> None:
        """Persist the DTS sampler under the state-file lock.

        Failures are logged but swallowed: losing an exploration-stats update
        must not fail an otherwise-successful proof request.
        """
        sampler = getattr(self.reasoner, "dts_sampler", None)
        if sampler is None:
            return
        async with self.dts_lock:
            try:
                await asyncio.to_thread(self._save_dts_sync)
            except Exception:
                logger.exception("failed to persist DTS state to %s", self.dts_state_path)

    def _save_dts_sync(self) -> None:
        self.dts_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.reasoner.dts_sampler.save_to(str(self.dts_state_path))

    async def shutdown(self) -> None:
        """Flush DTS state and close the Pantograph server process."""
        await self.persist_dts_state()
        server = getattr(self.reasoner, "server", None)
        close = getattr(server, "_close", None)
        if close is not None:
            await asyncio.to_thread(close)


async def build_default_service(
    *,
    config_path: Optional[Path] = None,
    tactic_model_path: Optional[Path] = None,
    argument_model_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    top_k_tactics: int = 3,
    top_k_subgoals: int = 3,
    max_depth: Optional[int] = None,
    max_nodes: int = 500,
    dts_c: Optional[float] = None,
    dts_random_seed: Optional[int] = None,
) -> InferenceService:
    """Construct the real service from configured model paths.

    Loads the GNN checkpoints once, starts one Pantograph/Lean server process
    once, and restores any existing DTS state — everything the batch CLI
    ``joint_inference.main`` does, minus the argparse/goal plumbing.
    Paths default to the env-configurable settings so Docker/deployments can
    redirect them without code changes.
    """
    from pantograph.server import Server

    from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, PantographExecutor
    from maths_ai.pln_inference.metta.translator.translator_modules.runner import DynamicThompsonSampler

    config_path = config_path or settings.gnn_config_path
    tactic_model_path = tactic_model_path or settings.tactic_model_path
    argument_model_path = argument_model_path or settings.argument_model_path
    index_path = index_path if index_path is not None else settings.lemma_index_path
    corpus_path = corpus_path if corpus_path is not None else settings.lemma_corpus_path

    server = await Server.create()
    try:
        dts_input = settings.dts_state_file
        dts_sampler = None
        if dts_input.exists() and dts_input.stat().st_size > 0:
            try:
                dts_sampler = DynamicThompsonSampler.load_from(
                    str(dts_input), C=dts_c if dts_c is not None else settings.dts_default_c
                )
            except Exception as exc:
                logger.warning("could not load DTS state from %s (%s); starting fresh", dts_input, exc)

        reasoner = HybridReasoner(
            config_path=config_path,
            tactic_model_path=tactic_model_path,
            argument_model_path=argument_model_path,
            index_path=index_path,
            corpus_path=corpus_path,
            executor=PantographExecutor(server=server),
            top_k_tactics=top_k_tactics,
            top_k_subgoals=top_k_subgoals,
            max_depth=max_depth if max_depth is not None else settings.proof_depth,
            max_nodes=max_nodes,
            dts_sampler=dts_sampler,
            dts_c=dts_c,
            dts_random_seed=dts_random_seed,
        )
    except Exception:
        server._close()
        raise

    return InferenceService(reasoner=reasoner)
