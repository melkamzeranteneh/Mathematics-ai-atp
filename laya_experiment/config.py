"""Configuration for the notebook-to-script Laya experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    model_repo: str = "jajostrains/Mathlib-Sexpr-GNN"
    dataset_repo: str = "jajostrains/Mathlib-Normalized-Sexpr"
    bundle_name: str = "pointer-gat-gru"
    split: str = "test"
    n_rows: int = 500
    seed: int = 42
    pool_k: int = 10
    select_k: int = 5
    batch_size: int = 64
    laya_model: str = "convaiinnovations/laya"
    laya_instructions: str = "Which Lean 4 tactic family best matches this proof step?"
    checkpoint_every: int = 50
    work_dir: Path = field(default_factory=Path.cwd)
    cache_dir: Path | None = None
    output_dir: Path | None = None
    device: str = "auto"
    laya_device: str = "auto"
    run_b1: bool = False
    run_b2: bool = False
    run_multilingual: bool = False
    export_laya: bool = False

    def __post_init__(self) -> None:
        if self.n_rows < 1:
            raise ValueError("n_rows must be positive")
        if self.pool_k < 1:
            raise ValueError("pool_k must be positive")
        if self.select_k < 1 or self.select_k > self.pool_k:
            raise ValueError("select_k must be between 1 and pool_k")
        if self.batch_size < 1 or self.checkpoint_every < 1:
            raise ValueError("batch_size and checkpoint_every must be positive")

    @property
    def model_dir(self) -> Path:
        return self.work_dir / "weights" / self.model_repo.replace("/", "--")

    @property
    def bundle_dir(self) -> Path:
        return self.model_dir / self.bundle_name

    @property
    def dataset_dir(self) -> Path:
        return self.work_dir / "data" / self.dataset_repo.replace("/", "--")

    @property
    def resolved_cache_dir(self) -> Path:
        return self.cache_dir or self.work_dir / "cache"

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir or self.work_dir / "outputs"

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def resolved_laya_device(self) -> str:
        if self.laya_device != "auto":
            return self.laya_device
        return self.resolved_device

    def prepare_directories(self) -> None:
        for path in (self.model_dir, self.dataset_dir, self.resolved_cache_dir, self.resolved_output_dir):
            path.mkdir(parents=True, exist_ok=True)
