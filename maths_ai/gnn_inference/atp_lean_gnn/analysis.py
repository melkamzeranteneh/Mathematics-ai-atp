from __future__ import annotations

import json
import argparse
import csv
import hashlib
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .dataset import canonicalize_split_name
from .labels import UNKNOWN_TACTIC
from .reporting import console_print
from .training import (
    _amp_dtype,
    _load_checkpoint,
    build_baseline_model,
    load_baseline_config,
    load_prepared_metadata,
    resolve_device,
    PreparedGraphDataset,
    REQUIRED_DATA_FIELDS,
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_csv(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(records)
    return path


def _invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {index: token for token, index in vocab.items()}


def _normalize_batch_strings(value, batch_size: int) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value) for _ in range(batch_size)]


def _normalize_batch_ints(value, batch_size: int) -> list[int]:
    if torch.is_tensor(value):
        flattened = value.view(-1).tolist()
        return [int(item) for item in flattened]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value) for _ in range(batch_size)]


def load_run_summary(run_dir: str | Path) -> dict[str, object]:
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run directory '{run_dir}' is missing 'summary.json'.")
    return _read_json(summary_path)


def load_metrics_history(run_dir: str | Path) -> list[dict[str, object]]:
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run directory '{run_dir}' is missing 'metrics.jsonl'.")
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _build_analysis_loader(
    run_dir: Path,
    split: str,
    *,
    prepared_root: str | Path | None = None,
):
    config = load_baseline_config(
        run_dir / "config.json",
        prepared_root_override=prepared_root,
    )
    metadata = load_prepared_metadata(config.prepared_root)
    dataset = PreparedGraphDataset(
        metadata,
        split=split,
        edge_mode=config.edge_mode,
        required_fields=REQUIRED_DATA_FIELDS,
        cache_in_memory=config.training.cache_in_memory,
    )

    # Keep analysis loaders single-process on Windows. These reports are
    # throughput-insensitive, and this avoids multiprocessing permission issues
    # in sandboxed or desktop Python environments.
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return config, metadata, loader


def _tensor_signature(hasher, name: str, value) -> None:
    hasher.update(name.encode("utf-8"))
    hasher.update(b"\0")
    if not torch.is_tensor(value):
        hasher.update(b"<missing>")
        return
    tensor = value.detach().cpu().contiguous()
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(str(tuple(tensor.shape)).encode("ascii"))
    hasher.update(tensor.numpy().tobytes())


def _model_input_signature(data) -> str:
    """Hash exactly the graph fields consumed by the baseline encoder."""
    hasher = hashlib.sha256()
    for field in (
        "x",
        "node_type",
        "is_bound",
        "binder_depth",
        "binder_kind",
        "edge_index",
        "state_node_index",
    ):
        _tensor_signature(hasher, field, getattr(data, field, None))
    return hasher.hexdigest()


def _iter_training_graphs(metadata, *, edge_mode: str):
    packed_manifest_path = metadata.root / "packed" / edge_mode / "manifest.json"
    if packed_manifest_path.exists():
        packed_manifest = _read_json(packed_manifest_path)
        if packed_manifest.get("edge_mode") != edge_mode:
            packed_manifest = {}
        split_payload = dict(packed_manifest.get("splits", {})).get("train")
        if isinstance(split_payload, dict):
            chunk_names = split_payload.get("chunks", [])
            expected_count = int(split_payload.get("count", -1))
            if isinstance(chunk_names, list) and chunk_names and expected_count >= 0:
                packed_root = packed_manifest_path.parent / "train"
                if all((packed_root / str(name)).exists() for name in chunk_names):
                    seen = 0
                    started = time.perf_counter()
                    for chunk_index, chunk_name in enumerate(chunk_names, start=1):
                        chunk_path = packed_root / str(chunk_name)
                        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
                        if not isinstance(chunk, list):
                            raise ValueError(f"Packed graph chunk '{chunk_path}' must contain a list.")
                        for data in chunk:
                            yield data
                        seen += len(chunk)
                        if chunk_index == 1 or chunk_index % 25 == 0 or chunk_index == len(chunk_names):
                            console_print(
                                f"    profiled {seen}/{expected_count} train examples "
                                f"from {chunk_index}/{len(chunk_names)} chunks "
                                f"({time.perf_counter() - started:.1f}s)"
                            )
                    if seen != expected_count:
                        raise ValueError(
                            f"Packed train cache yielded {seen} examples, expected {expected_count}."
                        )
                    return

    dataset = PreparedGraphDataset(metadata, split="train", edge_mode=edge_mode)
    started = time.perf_counter()
    for index, data in enumerate(dataset, start=1):
        yield data
        if index == 1 or index % 25000 == 0 or index == len(dataset):
            console_print(
                f"    profiled {index}/{len(dataset)} individual train examples "
                f"({time.perf_counter() - started:.1f}s)"
            )


def _build_training_profile(metadata, *, edge_mode: str) -> dict[str, object]:
    id_to_tactic = _invert_vocab(metadata.tactic_vocab)
    tactic_counts: Counter[int] = Counter()
    signature_labels: dict[str, Counter[int]] = defaultdict(Counter)
    total_examples = 0

    console_print("  Profiling training labels and duplicate model inputs...")
    for data in _iter_training_graphs(metadata, edge_mode=edge_mode):
        target_id = int(data.y.view(-1)[0].item())
        tactic_counts[target_id] += 1
        signature_labels[_model_input_signature(data)][target_id] += 1
        total_examples += 1

    duplicate_groups = [counts for counts in signature_labels.values() if sum(counts.values()) > 1]
    conflicting_groups = [counts for counts in duplicate_groups if len(counts) > 1]

    def _group_example_count(groups: list[Counter[int]]) -> int:
        return sum(sum(counts.values()) for counts in groups)

    def _group_oracle(groups: list[Counter[int]]) -> float | None:
        support = _group_example_count(groups)
        if support == 0:
            return None
        return sum(max(counts.values()) for counts in groups) / support

    oracle_correct = sum(max(counts.values()) for counts in signature_labels.values())
    counts_by_name = {
        id_to_tactic.get(tactic_id, UNKNOWN_TACTIC): count
        for tactic_id, count in sorted(tactic_counts.items())
    }
    return {
        "profile_version": 1,
        "prepared_root": str(metadata.root),
        "edge_mode": edge_mode,
        "total_examples": total_examples,
        "tactic_counts": counts_by_name,
        "frequency_order": [
            id_to_tactic.get(tactic_id, UNKNOWN_TACTIC)
            for tactic_id, _ in tactic_counts.most_common()
        ],
        "ambiguity": {
            "unique_model_input_count": len(signature_labels),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_example_count": _group_example_count(duplicate_groups),
            "conflicting_group_count": len(conflicting_groups),
            "conflicting_example_count": _group_example_count(conflicting_groups),
            "empirical_deterministic_top1_ceiling": (
                oracle_correct / total_examples if total_examples else None
            ),
            "duplicate_groups_top1_ceiling": _group_oracle(duplicate_groups),
            "conflicting_groups_top1_ceiling": _group_oracle(conflicting_groups),
        },
    }


def _load_or_build_training_profile(
    run_dir: Path,
    metadata,
    *,
    edge_mode: str,
) -> tuple[dict[str, object], Path]:
    path = run_dir / "analysis_training_profile.json"
    expected_count = int(metadata.split_manifest("train").get("success_count", -1))
    if path.exists():
        cached = _read_json(path)
        if (
            int(cached.get("profile_version", 0)) == 1
            and cached.get("prepared_root") == str(metadata.root)
            and cached.get("edge_mode") == edge_mode
            and int(cached.get("total_examples", -2)) == expected_count
        ):
            console_print(f"  Reusing cached training profile: {path}")
            return cached, path
    profile = _build_training_profile(metadata, edge_mode=edge_mode)
    return profile, _write_json(path, profile)


def _build_per_tactic_summary(
    records: list[dict[str, object]],
    *,
    min_support: int,
    training_counts: dict[str, int],
    medium_frequency_min: int,
    head_frequency_min: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if bool(record["is_unknown_target"]):
            continue
        grouped[str(record["true_tactic"])].append(record)

    summaries: list[dict[str, object]] = []
    for tactic_name, tactic_records in grouped.items():
        support = len(tactic_records)
        train_support = int(training_counts.get(tactic_name, 0))
        if train_support >= head_frequency_min:
            frequency_bucket = "head"
        elif train_support >= medium_frequency_min:
            frequency_bucket = "medium"
        elif train_support > 0:
            frequency_bucket = "tail"
        else:
            frequency_bucket = "unseen"
        top1_correct = sum(1 for record in tactic_records if bool(record["correct_top1"]))
        top5_correct = sum(1 for record in tactic_records if bool(record["correct_top5"]))
        summaries.append(
            {
                "tactic_name": tactic_name,
                "train_support": train_support,
                "frequency_bucket": frequency_bucket,
                "support": support,
                "top1_accuracy": top1_correct / support,
                "top5_accuracy": top5_correct / support,
            }
        )

    summaries.sort(key=lambda item: (-int(item["support"]), str(item["tactic_name"])))
    hardest = [
        item
        for item in summaries
        if int(item["support"]) >= min_support
    ]
    hardest.sort(key=lambda item: (float(item["top1_accuracy"]), -int(item["support"]), str(item["tactic_name"])))
    return summaries, hardest[:10]


def _build_frequency_bucket_summary(
    per_tactic: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for bucket in ("head", "medium", "tail", "unseen"):
        members = [item for item in per_tactic if item["frequency_bucket"] == bucket]
        support = sum(int(item["support"]) for item in members)
        if not members:
            continue
        weighted_top1 = sum(
            float(item["top1_accuracy"]) * int(item["support"])
            for item in members
        )
        weighted_top5 = sum(
            float(item["top5_accuracy"]) * int(item["support"])
            for item in members
        )
        summaries.append(
            {
                "bucket": bucket,
                "tactic_count": len(members),
                "support": support,
                "top1_accuracy": weighted_top1 / support if support else 0.0,
                "top5_accuracy": weighted_top5 / support if support else 0.0,
                "macro_top1_recall": sum(float(item["top1_accuracy"]) for item in members) / len(members),
            }
        )
    return summaries


def _build_rank_summary(records: list[dict[str, object]]) -> dict[str, object]:
    known = [record for record in records if not bool(record["is_unknown_target"])]
    if not known:
        return {"mean_reciprocal_rank": 0.0, "topk_accuracy": {}}
    topk_accuracy = {
        str(k): sum(int(record["true_rank"]) <= k for record in known) / len(known)
        for k in (1, 2, 3, 5, 10)
    }
    return {
        "mean_reciprocal_rank": sum(1.0 / int(record["true_rank"]) for record in known) / len(known),
        "mean_true_rank": sum(int(record["true_rank"]) for record in known) / len(known),
        "median_true_rank": float(statistics.median(int(record["true_rank"]) for record in known)),
        "topk_accuracy": topk_accuracy,
    }


def _build_calibration_summary(
    records: list[dict[str, object]],
    *,
    num_bins: int = 10,
) -> dict[str, object]:
    known = [record for record in records if not bool(record["is_unknown_target"])]
    bins: list[dict[str, object]] = []
    ece = 0.0
    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        members = [
            record
            for record in known
            if lower <= float(record["predicted_top1_confidence"])
            and (
                float(record["predicted_top1_confidence"]) < upper
                or (index == num_bins - 1 and float(record["predicted_top1_confidence"]) <= upper)
            )
        ]
        if not members:
            bins.append({"lower": lower, "upper": upper, "count": 0, "accuracy": None, "mean_confidence": None})
            continue
        accuracy = sum(bool(record["correct_top1"]) for record in members) / len(members)
        mean_confidence = sum(float(record["predicted_top1_confidence"]) for record in members) / len(members)
        ece += len(members) / max(len(known), 1) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    mean_confidence = (
        sum(float(record["predicted_top1_confidence"]) for record in known) / len(known)
        if known else 0.0
    )
    return {
        "bin_count": num_bins,
        "expected_calibration_error": ece,
        "mean_top1_confidence": mean_confidence,
        "top1_accuracy": sum(bool(record["correct_top1"]) for record in known) / len(known) if known else 0.0,
        "bins": bins,
    }


def _build_frequency_baselines(
    records: list[dict[str, object]],
    *,
    frequency_order: list[str],
) -> dict[str, object]:
    known = [record for record in records if not bool(record["is_unknown_target"])]
    topk: dict[str, float] = {}
    for k in (1, 2, 3, 5, 10):
        candidates = set(frequency_order[:k])
        topk[str(k)] = (
            sum(str(record["true_tactic"]) in candidates for record in known) / len(known)
            if known else 0.0
        )
    return {
        "ordered_tactics": frequency_order[:10],
        "topk_accuracy": topk,
    }


def _build_confusion_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    confusions: Counter[tuple[str, str]] = Counter()
    for record in records:
        if bool(record["is_unknown_target"]) or bool(record["correct_top1"]):
            continue
        confusions[(str(record["true_tactic"]), str(record["predicted_top1"]))] += 1
    return [
        {"true_tactic": true_tactic, "predicted_tactic": predicted_tactic, "count": count}
        for (true_tactic, predicted_tactic), count in confusions.most_common()
    ]


def _build_error_samples(records: list[dict[str, object]]) -> list[dict[str, object]]:
    errors = [
        record
        for record in records
        if not bool(record["is_unknown_target"]) and not bool(record["correct_top1"])
    ]
    errors.sort(key=lambda item: (-float(item["predicted_top1_confidence"]), str(item["true_tactic"])))
    return errors[:25]


def _render_analysis_markdown(analysis: dict[str, object]) -> str:
    overall = analysis["overall"]
    lines = [
        f"# Run Analysis ({analysis['split']})",
        "",
        f"- run dir: `{analysis['run_dir']}`",
        f"- checkpoint: `{analysis['checkpoint']}`",
        f"- epoch: `{analysis['epoch']}`",
        f"- top-1: `{overall['top1_accuracy']:.4f}`",
        f"- top-5: `{overall['top5_accuracy']:.4f}`",
        f"- top-10: `{analysis['ranking']['topk_accuracy']['10']:.4f}`",
        f"- MRR: `{analysis['ranking']['mean_reciprocal_rank']:.4f}`",
        f"- loss: `{overall['loss']:.4f}`",
        f"- macro top-1 recall: `{overall['macro_top1_recall']:.4f}`",
        f"- mean confidence: `{analysis['calibration']['mean_top1_confidence']:.4f}`",
        f"- calibration error (ECE): `{analysis['calibration']['expected_calibration_error']:.4f}`",
        f"- known labels: `{overall['known_label_count']}`",
        f"- unknown labels excluded: `{overall['unknown_label_excluded_count']}`",
        "",
        "## Frequency Baselines",
        "",
        f"- most-common tactic: `{analysis['frequency_baseline']['ordered_tactics'][0] if analysis['frequency_baseline']['ordered_tactics'] else 'none'}`",
        f"- most-common top-1 baseline: `{analysis['frequency_baseline']['topk_accuracy']['1']:.4f}`",
        f"- five-most-common coverage: `{analysis['frequency_baseline']['topk_accuracy']['5']:.4f}`",
        "",
        "## Training Input Ambiguity",
        "",
        f"- unique model inputs: `{analysis['training_ambiguity']['unique_model_input_count']}`",
        f"- duplicate input groups: `{analysis['training_ambiguity']['duplicate_group_count']}`",
        f"- conflicting-label groups: `{analysis['training_ambiguity']['conflicting_group_count']}`",
        f"- conflicting examples: `{analysis['training_ambiguity']['conflicting_example_count']}`",
        f"- empirical deterministic top-1 ceiling: `{analysis['training_ambiguity']['empirical_deterministic_top1_ceiling']}`",
        "",
        "## Accuracy by Training Frequency",
        "",
        "| Bucket | Tactics | Support | Top-1 | Top-5 | Macro Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in analysis["frequency_buckets"]:
        lines.append(
            f"| {item['bucket']} | {item['tactic_count']} | {item['support']} | "
            f"{item['top1_accuracy']:.4f} | {item['top5_accuracy']:.4f} | "
            f"{item['macro_top1_recall']:.4f} |"
        )

    lines.extend([
        "",
        "## Hardest Tactics",
        "",
        "| Tactic | Train Support | Eval Support | Bucket | Top-1 | Top-5 |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ])

    hardest_tactics = list(analysis["hardest_tactics"])
    if hardest_tactics:
        for item in hardest_tactics:
            lines.append(
                f"| {item['tactic_name']} | {item['train_support']} | {item['support']} | "
                f"{item['frequency_bucket']} | "
                f"{item['top1_accuracy']:.4f} | {item['top5_accuracy']:.4f} |"
            )
    else:
        lines.append("| none | 0 | 0 | none | 0.0000 | 0.0000 |")

    lines.extend(["", "## Common Confusions", ""])
    confusions = list(analysis["common_confusions"])
    if confusions:
        for item in confusions:
            lines.append(
                f"- `{item['true_tactic']}` -> `{item['predicted_tactic']}`: {item['count']}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Sample Errors", ""])
    errors = list(analysis["sample_errors"])
    if errors:
        for record in errors[:10]:
            lines.append(
                f"- `{record['true_tactic']}` predicted as `{record['predicted_top1']}` "
                f"(confidence={record['predicted_top1_confidence']:.4f}) "
                f"at row `{record['row_index']}` in `{record['theorem']}`"
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def analyze_saved_run(
    run_dir: str | Path,
    *,
    split: str,
    top_k: int = 10,
    min_support: int = 20,
    prepared_root: str | Path | None = None,
    medium_frequency_min: int = 100,
    head_frequency_min: int = 1000,
) -> dict[str, object]:
    run_directory = Path(run_dir)
    if not run_directory.exists():
        raise FileNotFoundError(f"Run directory '{run_directory}' does not exist.")

    canonical_split = canonicalize_split_name(split)
    if canonical_split not in {"val", "test"}:
        raise ValueError("Analysis split must be either 'val' or 'test'.")
    if top_k < 1:
        raise ValueError("Analysis parameter 'top_k' must be positive.")
    if min_support < 1:
        raise ValueError("Analysis parameter 'min_support' must be positive.")
    if medium_frequency_min < 1:
        raise ValueError("Analysis parameter 'medium_frequency_min' must be positive.")
    if head_frequency_min <= medium_frequency_min:
        raise ValueError("Analysis parameter 'head_frequency_min' must exceed 'medium_frequency_min'.")

    config, metadata, loader = _build_analysis_loader(
        run_directory,
        canonical_split,
        prepared_root=prepared_root,
    )
    training_profile, training_profile_path = _load_or_build_training_profile(
        run_directory,
        metadata,
        edge_mode=config.edge_mode,
    )
    device = resolve_device(config.device)
    model = build_baseline_model(metadata, config).to(device)
    checkpoint_path = run_directory / "best.pt"
    checkpoint = _load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    amp_dtype = _amp_dtype(device, config)
    use_amp = amp_dtype is not None
    id_to_tactic = _invert_vocab(metadata.tactic_vocab)
    records: list[dict[str, object]] = []
    loss_sum = 0.0
    known_label_count = 0

    console_print(f"  Analyzing {canonical_split} split ({len(loader)} batches)...")
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch.y.numel())
            row_indices = _normalize_batch_ints(batch.row_index, batch_size)
            theorems = _normalize_batch_strings(batch.theorem, batch_size)
            true_tactic_names = _normalize_batch_strings(batch.tactic_name, batch_size)

            batch = batch.to(device, non_blocking=(device.type == "cuda" and config.training.pin_memory))
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(batch)
            metric_logits = logits.float()
            probabilities = metric_logits.softmax(dim=1)
            known_mask = batch.y.view(-1) != metadata.unknown_tactic_id
            if bool(known_mask.any()):
                known_logits = metric_logits[known_mask]
                known_targets = batch.y.view(-1)[known_mask]
                batch_loss = torch.nn.functional.cross_entropy(known_logits, known_targets)
                loss_sum += float(batch_loss.item()) * int(known_targets.numel())
                known_label_count += int(known_targets.numel())

            targets = batch.y.view(-1).detach().cpu()
            top_k_size = min(top_k, logits.size(1))
            topk = metric_logits.topk(top_k_size, dim=1)
            topk_ids = topk.indices.detach().cpu()
            topk_confidences = probabilities.gather(1, topk.indices).detach().float().cpu()
            target_logits = metric_logits.gather(1, batch.y.view(-1, 1))
            true_ranks = (metric_logits > target_logits).sum(dim=1).add(1).detach().cpu()
            true_probabilities = probabilities.gather(1, batch.y.view(-1, 1)).squeeze(1).detach().float().cpu()

            for index in range(batch_size):
                true_id = int(targets[index].item())
                predicted_ids = [int(item) for item in topk_ids[index].tolist()]
                predicted_confidences = [float(item) for item in topk_confidences[index].tolist()]
                predicted_tactics = [
                    id_to_tactic.get(predicted_id, UNKNOWN_TACTIC)
                    for predicted_id in predicted_ids
                ]
                is_unknown_target = true_id == metadata.unknown_tactic_id
                record = {
                    "row_index": row_indices[index],
                    "theorem": theorems[index],
                    "true_tactic": true_tactic_names[index],
                    "true_tactic_id": true_id,
                    "predicted_top1": predicted_tactics[0],
                    "predicted_top1_id": predicted_ids[0],
                    "predicted_top1_confidence": predicted_confidences[0],
                    "predicted_topk": predicted_tactics,
                    "predicted_topk_ids": predicted_ids,
                    "predicted_topk_confidences": predicted_confidences,
                    "true_probability": float(true_probabilities[index].item()),
                    "true_rank": int(true_ranks[index].item()),
                    "is_unknown_target": is_unknown_target,
                    "correct_top1": (not is_unknown_target) and (predicted_ids[0] == true_id),
                    "correct_top5": (not is_unknown_target) and (int(true_ranks[index].item()) <= 5),
                }
                records.append(record)

    known_records = [record for record in records if not bool(record["is_unknown_target"])]
    known_count = len(known_records)
    top1_correct = sum(1 for record in known_records if bool(record["correct_top1"]))
    top5_correct = sum(1 for record in known_records if bool(record["correct_top5"]))
    training_counts = {
        str(name): int(count)
        for name, count in dict(training_profile["tactic_counts"]).items()
    }
    per_tactic_summary, hardest_tactics = _build_per_tactic_summary(
        records,
        min_support=min_support,
        training_counts=training_counts,
        medium_frequency_min=medium_frequency_min,
        head_frequency_min=head_frequency_min,
    )
    frequency_buckets = _build_frequency_bucket_summary(per_tactic_summary)
    ranking = _build_rank_summary(records)
    calibration = _build_calibration_summary(records)
    frequency_baseline = _build_frequency_baselines(
        records,
        frequency_order=[str(item) for item in training_profile["frequency_order"]],
    )
    all_confusions = _build_confusion_summary(records)
    common_confusions = all_confusions[:15]
    sample_errors = _build_error_samples(records)
    macro_top1_recall = (
        sum(float(item["top1_accuracy"]) for item in per_tactic_summary) / len(per_tactic_summary)
        if per_tactic_summary else 0.0
    )

    analysis = {
        "run_dir": str(run_directory),
        "split": canonical_split,
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "overall": {
            "top1_accuracy": top1_correct / known_count if known_count else 0.0,
            "top5_accuracy": top5_correct / known_count if known_count else 0.0,
            "loss": loss_sum / known_label_count if known_label_count else 0.0,
            "macro_top1_recall": macro_top1_recall,
            "known_label_count": known_count,
            "unknown_label_excluded_count": len(records) - known_count,
            "evaluated_count": len(records),
        },
        "curve_context": load_metrics_history(run_directory),
        "ranking": ranking,
        "calibration": calibration,
        "frequency_baseline": frequency_baseline,
        "frequency_bucket_thresholds": {
            "medium_min": medium_frequency_min,
            "head_min": head_frequency_min,
        },
        "frequency_buckets": frequency_buckets,
        "training_ambiguity": training_profile["ambiguity"],
        "per_tactic_summary": per_tactic_summary,
        "hardest_tactics": hardest_tactics,
        "common_confusions": common_confusions,
        "sample_errors": sample_errors,
    }

    predictions_path = _write_jsonl(run_directory / f"predictions_{canonical_split}.jsonl", records)
    per_tactic_path = _write_csv(
        run_directory / f"per_tactic_{canonical_split}.csv",
        per_tactic_summary,
    )
    confusion_path = _write_csv(
        run_directory / f"confusion_pairs_{canonical_split}.csv",
        all_confusions,
    )
    analysis_json_path = _write_json(run_directory / f"analysis_{canonical_split}.json", analysis)
    analysis_markdown_path = _write_text(
        run_directory / f"analysis_{canonical_split}.md",
        _render_analysis_markdown(analysis),
    )
    analysis["artifacts"] = {
        "predictions_jsonl": str(predictions_path),
        "per_tactic_csv": str(per_tactic_path),
        "confusion_pairs_csv": str(confusion_path),
        "training_profile_json": str(training_profile_path),
        "analysis_json": str(analysis_json_path),
        "analysis_markdown": str(analysis_markdown_path),
    }
    _write_json(run_directory / f"analysis_{canonical_split}.json", analysis)
    return analysis


def compare_saved_runs(run_dirs: list[str | Path]) -> dict[str, object]:
    if not run_dirs:
        raise ValueError("At least one run directory must be provided for comparison.")

    runs: list[dict[str, object]] = []
    for run_dir in run_dirs:
        run_directory = Path(run_dir)
        summary = load_run_summary(run_directory)
        config = load_baseline_config(run_directory / "config.json")
        runs.append(
            {
                "run_dir": str(run_directory),
                "run_name": run_directory.name,
                "best_epoch": int(summary["best_epoch"]),
                "val_top1": float(summary["best_validation"]["top1_accuracy"]),
                "val_top5": float(summary["best_validation"]["top5_accuracy"]),
                "test_top1": float(summary["test_evaluation"]["top1_accuracy"]),
                "test_top5": float(summary["test_evaluation"]["top5_accuracy"]),
                "top1_gap": float(summary["best_validation"]["top1_accuracy"]) - float(summary["test_evaluation"]["top1_accuracy"]),
                "edge_mode": config.edge_mode,
                "hidden_dim": config.model.hidden_dim,
                "num_layers": config.model.num_layers,
                "use_node_type": config.use_node_type,
                "amp_enabled": bool(summary.get("amp_enabled", False)),
            }
        )

    runs.sort(key=lambda item: (-float(item["test_top1"]), -float(item["val_top1"]), str(item["run_name"])))
    return {"runs": runs}


def render_run_comparison_markdown(comparison: dict[str, object]) -> str:
    lines = [
        "# Run Comparison",
        "",
        "| Run | Best Epoch | Val Top-1 | Val Top-5 | Test Top-1 | Test Top-5 | Top-1 Gap | Edge Mode | Hidden | Layers | Node Type | AMP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for item in comparison["runs"]:
        lines.append(
            f"| {item['run_name']} | {item['best_epoch']} | "
            f"{item['val_top1']:.4f} | {item['val_top5']:.4f} | "
            f"{item['test_top1']:.4f} | {item['test_top5']:.4f} | "
            f"{item['top1_gap']:.4f} | {item['edge_mode']} | "
            f"{item['hidden_dim']} | {item['num_layers']} | "
            f"{item['use_node_type']} | {item['amp_enabled']} |"
        )
    return "\n".join(lines) + "\n"


def build_analyze_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a trained baseline run in detail")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to a completed run directory")
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Which split to analyze",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many predicted tactics to retain per example")
    parser.add_argument(
        "--min-support",
        type=int,
        default=20,
        help="Minimum support before a tactic is considered in the hardest-tactic table",
    )
    parser.add_argument(
        "--prepared-root",
        type=str,
        default=None,
        help="Optional prepared dataset override (useful when the run config points to stale /tmp data)",
    )
    parser.add_argument(
        "--medium-frequency-min",
        type=int,
        default=100,
        help="Minimum training support for the medium-frequency bucket",
    )
    parser.add_argument(
        "--head-frequency-min",
        type=int,
        default=1000,
        help="Minimum training support for the head-frequency bucket",
    )
    return parser


def build_compare_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare finished baseline runs")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more run directories to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the markdown comparison table",
    )
    return parser


def analyze_main(argv: list[str] | None = None) -> int:
    parser = build_analyze_arg_parser()
    args = parser.parse_args(argv)

    try:
        splits = ["val", "test"] if args.split == "both" else [args.split]
        for split in splits:
            analysis = analyze_saved_run(
                args.run_dir,
                split=split,
                top_k=args.top_k,
                min_support=args.min_support,
                prepared_root=args.prepared_root,
                medium_frequency_min=args.medium_frequency_min,
                head_frequency_min=args.head_frequency_min,
            )
            console_print(f"  Wrote analysis summary   : {analysis['artifacts']['analysis_json']}")
            console_print(f"  Wrote analysis markdown  : {analysis['artifacts']['analysis_markdown']}")
            console_print(f"  Wrote prediction records : {analysis['artifacts']['predictions_jsonl']}")
            console_print(f"  Wrote per-tactic table   : {analysis['artifacts']['per_tactic_csv']}")
            console_print(f"  Wrote confusion table    : {analysis['artifacts']['confusion_pairs_csv']}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0


def compare_main(argv: list[str] | None = None) -> int:
    parser = build_compare_arg_parser()
    args = parser.parse_args(argv)

    try:
        comparison = compare_saved_runs(args.run_dirs)
        markdown = render_run_comparison_markdown(comparison)
        if args.output:
            output_path = _write_text(Path(args.output), markdown)
            console_print(f"  Wrote comparison table   : {output_path}")
        else:
            console_print(markdown)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0
