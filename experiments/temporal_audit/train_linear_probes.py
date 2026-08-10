#!/usr/bin/env python3
"""Train seeded verb/noun probes and report temporal-retention metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_shards(directory: Path, split: str) -> dict[str, np.ndarray]:
    paths = sorted(directory.glob(f"{split}_shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards found in {directory}")
    arrays: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path) as data:
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])
    return {key: np.concatenate(values, axis=0) for key, values in arrays.items()}


class DualProbe(nn.Module):
    def __init__(self, dimension: int, verbs: int, nouns: int):
        super().__init__()
        self.verb = nn.Linear(dimension, verbs)
        self.noun = nn.Linear(dimension, nouns)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.verb(features), self.noun(features)


def evaluate(
    model: DualProbe,
    features: np.ndarray,
    verbs: np.ndarray,
    nouns: np.ndarray,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    verb_correct = noun_correct = action_correct = total = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = start + batch_size
            batch = torch.from_numpy(features[start:stop]).to(device)
            verb_logits, noun_logits = model(batch)
            verb_pred = verb_logits.argmax(dim=-1).cpu().numpy()
            noun_pred = noun_logits.argmax(dim=-1).cpu().numpy()
            true_verbs = verbs[start:stop]
            true_nouns = nouns[start:stop]
            verb_correct += int((verb_pred == true_verbs).sum())
            noun_correct += int((noun_pred == true_nouns).sum())
            action_correct += int(((verb_pred == true_verbs) & (noun_pred == true_nouns)).sum())
            total += len(true_verbs)
    return {
        "verb_top1": verb_correct / total,
        "noun_top1": noun_correct / total,
        "action_top1": action_correct / total,
    }


def run_seed(
    seed: int,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = train["feature_ordered"].astype(np.float32)
    verbs = train["verb"].astype(np.int64)
    nouns = train["noun"].astype(np.int64)
    dataset = TensorDataset(
        torch.from_numpy(features), torch.from_numpy(verbs), torch.from_numpy(nouns)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )
    model = DualProbe(
        features.shape[1], int(verbs.max()) + 1, int(nouns.max()) + 1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch_features, batch_verbs, batch_nouns in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_verbs = batch_verbs.to(device, non_blocking=True)
            batch_nouns = batch_nouns.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            verb_logits, noun_logits = model(batch_features)
            loss = criterion(verb_logits, batch_verbs) + criterion(noun_logits, batch_nouns)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        print(
            f"seed={seed} epoch={epoch + 1}/{args.epochs} loss={running / len(loader):.5f}",
            flush=True,
        )

    conditions = sorted(key.removeprefix("feature_") for key in validation if key.startswith("feature_"))
    metrics = {}
    ordered = validation["feature_ordered"].astype(np.float32)
    for condition in conditions:
        condition_features = validation[f"feature_{condition}"].astype(np.float32)
        scores = evaluate(
            model,
            condition_features,
            validation["verb"],
            validation["noun"],
            device,
            args.eval_batch_size,
        )
        similarity = np.sum(ordered * condition_features, axis=1)
        scores["ordered_feature_cosine"] = float(similarity.mean())
        metrics[condition] = scores
    for condition, scores in metrics.items():
        if condition == "ordered":
            continue
        for metric in ("verb_top1", "noun_top1", "action_top1"):
            baseline = metrics["ordered"][metric]
            scores[f"{metric}_retention"] = scores[metric] / baseline if baseline else 0.0
            scores[f"{metric}_drop_pp"] = 100.0 * (baseline - scores[metric])
    return {"seed": seed, "conditions": metrics}


def aggregate(seed_results: list[dict]) -> dict:
    conditions = seed_results[0]["conditions"]
    output = {}
    for condition, metrics in conditions.items():
        output[condition] = {}
        for metric in metrics:
            values = [result["conditions"][condition][metric] for result in seed_results]
            output[condition][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    return output


def run(args: argparse.Namespace) -> None:
    directory = Path(args.feature_dir)
    train = load_shards(directory, "train")
    validation = load_shards(directory, "validation")
    print(
        json.dumps(
            {
                "train_samples": len(train["verb"]),
                "validation_samples": len(validation["verb"]),
                "dimension": int(train["feature_ordered"].shape[1]),
            }
        ),
        flush=True,
    )
    seed_results = [run_seed(seed, train, validation, args) for seed in args.seeds]
    result = {
        "config": vars(args),
        "per_seed": seed_results,
        "aggregate": aggregate(seed_results),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
