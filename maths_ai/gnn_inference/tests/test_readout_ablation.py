from __future__ import annotations

import contextlib
import io
import shutil
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from maths_ai.gnn_inference.atp_lean_gnn.model import (
    GATv2StateClassifier,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import BaselineConfig
from maths_ai.gnn_inference.scripts.run_readout_ablation import main as ablation_main


class ReadoutAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("maths_ai/gnn_inference/tests/_tmp_readout_ablation")
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def _batch() -> Batch:
        graphs = []
        for offset in (0, 1):
            graphs.append(
                Data(
                    x=torch.tensor([1 + offset, 2, 3], dtype=torch.long),
                    node_type=torch.tensor([0, 1, 1], dtype=torch.long),
                    edge_index=torch.tensor(
                        [[0, 0, 1, 2], [1, 2, 0, 0]],
                        dtype=torch.long,
                    ),
                    state_node_index=torch.tensor([0], dtype=torch.long),
                    is_bound=torch.zeros(3, dtype=torch.long),
                    binder_depth=torch.zeros(3, dtype=torch.long),
                    binder_kind=torch.zeros(3, dtype=torch.long),
                )
            )
        return Batch.from_data_list(graphs)

    def test_all_attention_readouts_produce_graph_logits_and_normalized_weights(self):
        batch = self._batch()
        expected_fusion_widths = {
            "state_mean_attention": 24,
            "state_max_attention": 24,
            "state_mean_max_attention": 32,
        }
        for readout, fusion_width in expected_fusion_widths.items():
            with self.subTest(readout=readout):
                model = GATv2StateClassifier(
                    num_node_labels=8,
                    num_tactics=5,
                    hidden_dim=8,
                    num_layers=1,
                    heads=2,
                    dropout=0.0,
                    readout=readout,
                )
                logits, details = model.forward_with_readout_details(batch)

                self.assertEqual(tuple(logits.shape), (2, 5))
                self.assertEqual(
                    model.global_readout.fusion.in_features,
                    fusion_width,
                )
                weights = details["attention_weights"]
                for graph_index in range(2):
                    graph_sum = weights[batch.batch == graph_index].sum()
                    self.assertTrue(
                        torch.allclose(graph_sum, torch.tensor(1.0), atol=1e-6)
                    )

    def test_training_config_accepts_every_ablation_readout(self):
        for readout in (
            "state_mean_attention",
            "state_max_attention",
            "state_mean_max_attention",
        ):
            with self.subTest(readout=readout):
                config = BaselineConfig.from_dict(
                    {
                        "prepared_root": str(self.root),
                        "gnn_type": "gat",
                        "model": {
                            "hidden_dim": 8,
                            "num_layers": 1,
                            "dropout": 0.0,
                            "heads": 2,
                            "readout": readout,
                        },
                    }
                )
                self.assertEqual(config.model.readout, readout)

    def test_ablation_launcher_dry_run_assigns_requested_gpu(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = ablation_main(
                [
                    "--prepared-root",
                    str(self.root),
                    "--variants",
                    "state_max_attention",
                    "--gpus",
                    "1",
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("state_max_attention", output.getvalue())
        self.assertIn("--device cuda:1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
