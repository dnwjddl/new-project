#!/usr/bin/env python3
"""Probe VideoFlexTok prefixes, marginal blocks, temporal sensitivity, and redundancy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DualHead(nn.Module):
    def __init__(self, dimension: int, verbs: int, nouns: int):
        super().__init__()
        self.verb = nn.Linear(dimension, verbs)
        self.noun = nn.Linear(dimension, nouns)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.verb(features), self.noun(features)


class MultiBudgetProbe(nn.Module):
    def __init__(
        self,
        dimension_per_token: int,
        budgets: tuple[int, ...],
        verbs: int,
        nouns: int,
    ):
        super().__init__()
        self.dimension_per_token = dimension_per_token
        self.budgets = budgets
        self.prefix = nn.ModuleDict(
            {
                str(budget): DualHead(dimension_per_token * budget, verbs, nouns)
                for budget in budgets
            }
        )
        starts = (0,) + budgets[:-1]
        self.block = nn.ModuleDict(
            {
                str(budget): DualHead(dimension_per_token * (budget - start), verbs, nouns)
                for start, budget in zip(starts, budgets)
            }
        )

    def slices(self, features: torch.Tensor):
        start = 0
        for budget in self.budgets:
            stop_dim = budget * self.dimension_per_token
            start_dim = start * self.dimension_per_token
            yield (
                str(budget),
                self.prefix[str(budget)],
                features[:, :stop_dim],
                self.block[str(budget)],
                features[:, start_dim:stop_dim],
            )
            start = budget


class FixedPoolProbe(nn.Module):
    def __init__(self, dimension: int, budgets: tuple[int, ...], verbs: int, nouns: int):
        super().__init__()
        self.budgets = budgets
        self.heads = nn.ModuleDict(
            {
                family: nn.ModuleDict(
                    {str(budget): DualHead(dimension, verbs, nouns) for budget in budgets}
                )
                for family in ("prefix", "block")
            }
        )


def shard_paths(directory: Path, split: str) -> list[Path]:
    paths = sorted(directory.glob(f"{split}_shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards in {directory}")
    return paths


def load_labels(directory: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    verbs, nouns = [], []
    for path in shard_paths(directory, split):
        with np.load(path) as data:
            verbs.append(data["verb"])
            nouns.append(data["noun"])
    return np.concatenate(verbs).astype(np.int64), np.concatenate(nouns).astype(np.int64)


def load_array(directory: Path, split: str, key: str) -> np.ndarray:
    arrays = []
    for path in shard_paths(directory, split):
        with np.load(path) as data:
            arrays.append(data[key])
    return np.concatenate(arrays, axis=0)


def train_seed(
    seed: int,
    features: np.ndarray,
    verbs: np.ndarray,
    nouns: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> MultiBudgetProbe:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dimension_per_token = features.shape[1] // args.total_tokens
    model = MultiBudgetProbe(
        dimension_per_token,
        tuple(args.budgets),
        int(verbs.max()) + 1,
        int(nouns.max()) + 1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)

    for epoch in range(args.epochs):
        permutation = torch.randperm(len(features), generator=generator).numpy()
        running = 0.0
        steps = 0
        model.train()
        for start in range(0, len(features), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch = torch.from_numpy(features[indices].astype(np.float32)).to(device)
            batch = (batch - mean_tensor) / std_tensor
            batch_verbs = torch.from_numpy(verbs[indices]).to(device)
            batch_nouns = torch.from_numpy(nouns[indices]).to(device)
            losses = []
            for _, prefix_head, prefix, block_head, block in model.slices(batch):
                prefix_verb, prefix_noun = prefix_head(prefix)
                block_verb, block_noun = block_head(block)
                losses.extend(
                    (
                        criterion(prefix_verb, batch_verbs),
                        criterion(prefix_noun, batch_nouns),
                        criterion(block_verb, batch_verbs),
                        criterion(block_noun, batch_nouns),
                    )
                )
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        print(
            f"seed={seed} epoch={epoch + 1}/{args.epochs} loss={running / steps:.5f}",
            flush=True,
        )
    return model


def fixed_pool_features(
    features: np.ndarray, budgets: list[int], total_tokens: int
) -> dict[str, dict[str, np.ndarray]]:
    dimension = features.shape[1] // total_tokens
    values = features.reshape(len(features), total_tokens, dimension).astype(np.float32)
    output = {"prefix": {}, "block": {}}
    start = 0
    for budget in budgets:
        key = str(budget)
        output["prefix"][key] = values[:, :budget].mean(axis=1)
        output["block"][key] = values[:, start:budget].mean(axis=1)
        start = budget
    return output


def train_fixed_pool_seed(
    seed: int,
    features: dict[str, dict[str, np.ndarray]],
    verbs: np.ndarray,
    nouns: np.ndarray,
    statistics: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    args: argparse.Namespace,
) -> FixedPoolProbe:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    first = features["prefix"][str(args.budgets[0])]
    model = FixedPoolProbe(
        first.shape[1],
        tuple(args.budgets),
        int(verbs.max()) + 1,
        int(nouns.max()) + 1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        family: {
            key: (
                torch.from_numpy(values).to(device),
                torch.from_numpy(statistics[family][key][0]).to(device),
                torch.from_numpy(statistics[family][key][1]).to(device),
            )
            for key, values in rows.items()
        }
        for family, rows in features.items()
    }
    verb_tensor = torch.from_numpy(verbs).to(device)
    noun_tensor = torch.from_numpy(nouns).to(device)
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(verbs), generator=generator)
        running = 0.0
        steps = 0
        model.train()
        for start in range(0, len(verbs), args.batch_size):
            indices = permutation[start : start + args.batch_size].to(device)
            batch_verbs = verb_tensor[indices]
            batch_nouns = noun_tensor[indices]
            losses = []
            for family in ("prefix", "block"):
                for budget in args.budgets:
                    key = str(budget)
                    values, mean, std = tensors[family][key]
                    batch = (values[indices] - mean) / std
                    verb_logits, noun_logits = model.heads[family][key](batch)
                    losses.extend(
                        (criterion(verb_logits, batch_verbs), criterion(noun_logits, batch_nouns))
                    )
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        print(
            f"fixed-pool seed={seed} epoch={epoch + 1}/{args.epochs} "
            f"loss={running / steps:.5f}",
            flush=True,
        )
    return model


def evaluate_fixed_pool(
    model: FixedPoolProbe,
    features: dict[str, dict[str, np.ndarray]],
    verbs: np.ndarray,
    nouns: np.ndarray,
    statistics: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    batch_size: int,
) -> dict:
    device = next(model.parameters()).device
    counts = empty_counts(list(model.budgets))
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(verbs), batch_size):
            stop = start + batch_size
            true_verbs = verbs[start:stop]
            true_nouns = nouns[start:stop]
            for family in ("prefix", "block"):
                for budget in model.budgets:
                    key = str(budget)
                    mean, std = statistics[family][key]
                    batch = torch.from_numpy(
                        (features[family][key][start:stop] - mean) / std
                    ).to(device)
                    verb_logits, noun_logits = model.heads[family][key](batch)
                    verb_pred = verb_logits.argmax(dim=-1).cpu().numpy()
                    noun_pred = noun_logits.argmax(dim=-1).cpu().numpy()
                    counts[family][key]["verb"] += int((verb_pred == true_verbs).sum())
                    counts[family][key]["noun"] += int((noun_pred == true_nouns).sum())
                    counts[family][key]["action"] += int(
                        ((verb_pred == true_verbs) & (noun_pred == true_nouns)).sum()
                    )
    return {
        family: {
            budget: {metric: value / len(verbs) for metric, value in metrics.items()}
            for budget, metrics in rows.items()
        }
        for family, rows in counts.items()
    }


def empty_counts(budgets: list[int]) -> dict:
    return {
        family: {
            str(budget): {"verb": 0, "noun": 0, "action": 0} for budget in budgets
        }
        for family in ("prefix", "block")
    }


def evaluate(
    model: MultiBudgetProbe,
    features: np.ndarray,
    verbs: np.ndarray,
    nouns: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> dict:
    device = next(model.parameters()).device
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)
    counts = empty_counts(list(model.budgets))
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = start + batch_size
            batch = torch.from_numpy(features[start:stop].astype(np.float32)).to(device)
            batch = (batch - mean_tensor) / std_tensor
            true_verbs = verbs[start:stop]
            true_nouns = nouns[start:stop]
            for key, prefix_head, prefix, block_head, block in model.slices(batch):
                for family, head, values in (
                    ("prefix", prefix_head, prefix),
                    ("block", block_head, block),
                ):
                    verb_logits, noun_logits = head(values)
                    verb_pred = verb_logits.argmax(dim=-1).cpu().numpy()
                    noun_pred = noun_logits.argmax(dim=-1).cpu().numpy()
                    counts[family][key]["verb"] += int((verb_pred == true_verbs).sum())
                    counts[family][key]["noun"] += int((noun_pred == true_nouns).sum())
                    counts[family][key]["action"] += int(
                        ((verb_pred == true_verbs) & (noun_pred == true_nouns)).sum()
                    )
    return {
        family: {
            budget: {metric: value / len(features) for metric, value in metrics.items()}
            for budget, metrics in budget_rows.items()
        }
        for family, budget_rows in counts.items()
    }


def add_retention(condition_rows: dict, ordered_rows: dict) -> None:
    for family in ("prefix", "block"):
        for budget, metrics in condition_rows[family].items():
            for metric in ("verb", "noun", "action"):
                baseline = ordered_rows[family][budget][metric]
                value = metrics[metric]
                metrics[f"{metric}_retention"] = value / baseline if baseline else 0.0
                metrics[f"{metric}_drop_pp"] = 100.0 * (baseline - value)


def aggregate(seed_results: list[dict]) -> dict:
    def recurse(values: list):
        if isinstance(values[0], dict):
            return {key: recurse([value[key] for value in values]) for key in values[0]}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }

    return recurse(seed_results)


def linear_cka_matrix(
    features: np.ndarray,
    budgets: list[int],
    dimension_per_token: int,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(features), size=min(samples, len(features)), replace=False)
    blocks = []
    start = 0
    for budget in budgets:
        values = features[
            indices, start * dimension_per_token : budget * dimension_per_token
        ].astype(np.float32)
        values -= values.mean(axis=0, keepdims=True)
        gram = values @ values.T
        blocks.append(gram)
        start = budget
    matrix = np.zeros((len(blocks), len(blocks)), dtype=np.float64)
    for i, first in enumerate(blocks):
        for j, second in enumerate(blocks):
            numerator = float(np.sum(first * second))
            denominator = math.sqrt(float(np.sum(first * first) * np.sum(second * second)))
            matrix[i, j] = numerator / denominator if denominator else 0.0
    return {"budgets": budgets, "matrix": matrix.tolist(), "samples": len(indices)}


def token_statistics(tokens: np.ndarray, budgets: list[int]) -> dict:
    if tokens.ndim < 3 or tokens.shape[-1] != budgets[-1]:
        raise ValueError(f"Unexpected token shape: {tokens.shape}")
    temporal = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
    output = {}
    start = 0
    for budget in budgets:
        block = temporal[:, :, start:budget]
        values, counts = np.unique(block, return_counts=True)
        probabilities = counts.astype(np.float64) / counts.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        change = float((block[:, 1:] != block[:, :-1]).mean()) if block.shape[1] > 1 else 0.0
        output[str(budget)] = {
            "start": start,
            "stop": budget,
            "code_entropy_bits": entropy,
            "unique_codes": int(len(values)),
            "temporal_change_rate": change,
        }
        start = budget
    return output


def run_fixed_pool_representation(
    train: np.ndarray,
    validation: dict[str, np.ndarray],
    train_verbs: np.ndarray,
    train_nouns: np.ndarray,
    val_verbs: np.ndarray,
    val_nouns: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    train_features = fixed_pool_features(train, args.budgets, args.total_tokens)
    validation_features = {
        condition: fixed_pool_features(features, args.budgets, args.total_tokens)
        for condition, features in validation.items()
    }
    statistics = {
        family: {
            key: (
                values.mean(axis=0),
                np.maximum(values.std(axis=0), 1e-4),
            )
            for key, values in rows.items()
        }
        for family, rows in train_features.items()
    }
    seed_results = []
    for seed in args.seeds:
        model = train_fixed_pool_seed(
            seed,
            train_features,
            train_verbs,
            train_nouns,
            statistics,
            args,
        )
        rows = {
            condition: evaluate_fixed_pool(
                model,
                features,
                val_verbs,
                val_nouns,
                statistics,
                args.eval_batch_size,
            )
            for condition, features in validation_features.items()
        }
        for condition in args.conditions:
            if condition != "ordered":
                add_retention(rows[condition], rows["ordered"])
        seed_results.append({"seed": seed, "conditions": rows})
        model.to("cpu")
        torch.cuda.empty_cache()
    return {
        "dimension": train_features["prefix"][str(args.budgets[0])].shape[1],
        "pooling": "mean over token slots; temporal mean/std/delta channels retained",
        "per_seed": seed_results,
        "aggregate": aggregate([result["conditions"] for result in seed_results]),
    }


def run_representation(
    representation: str,
    directory: Path,
    train_verbs: np.ndarray,
    train_nouns: np.ndarray,
    val_verbs: np.ndarray,
    val_nouns: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    train = load_array(directory, "train", f"{representation}_ordered")
    mean = train.astype(np.float32).mean(axis=0)
    std = train.astype(np.float32).std(axis=0)
    std = np.maximum(std, 1e-4)
    validation = {
        condition: load_array(directory, "validation", f"{representation}_{condition}")
        for condition in args.conditions
    }
    seed_results = []
    for seed in args.seeds:
        model = train_seed(seed, train, train_verbs, train_nouns, mean, std, args)
        rows = {
            condition: evaluate(
                model,
                features,
                val_verbs,
                val_nouns,
                mean,
                std,
                args.eval_batch_size,
            )
            for condition, features in validation.items()
        }
        for condition in args.conditions:
            if condition != "ordered":
                add_retention(rows[condition], rows["ordered"])
        seed_results.append({"seed": seed, "conditions": rows})
    dimension_per_token = train.shape[1] // args.total_tokens
    return {
        "feature_shape": list(train.shape),
        "dimension_per_token": dimension_per_token,
        "per_seed": seed_results,
        "aggregate": aggregate([result["conditions"] for result in seed_results]),
        "block_cka": linear_cka_matrix(
            validation["ordered"],
            args.budgets,
            dimension_per_token,
            args.cka_samples,
            args.seeds[0],
        ),
        "fixed_dim_pool": run_fixed_pool_representation(
            train,
            validation,
            train_verbs,
            train_nouns,
            val_verbs,
            val_nouns,
            args,
        ),
    }


def run(args: argparse.Namespace) -> None:
    directory = Path(args.feature_dir)
    train_verbs, train_nouns = load_labels(directory, "train")
    val_verbs, val_nouns = load_labels(directory, "validation")
    result = {
        "config": vars(args),
        "samples": {"train": len(train_verbs), "validation": len(val_verbs)},
        "representations": {},
    }
    for representation in args.representations:
        print(f"Analyzing {representation}", flush=True)
        result["representations"][representation] = run_representation(
            representation,
            directory,
            train_verbs,
            train_nouns,
            val_verbs,
            val_nouns,
            args,
        )
    ordered_tokens = load_array(directory, "validation", "tokens_ordered")
    result["ordered_token_statistics"] = token_statistics(ordered_tokens, args.budgets)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--representations", nargs="+", default=("quant", "prequant"))
    parser.add_argument(
        "--conditions", nargs="+", default=("ordered", "reverse", "shuffle", "single")
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=(8, 16, 32, 64, 128, 256))
    parser.add_argument("--total-tokens", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cka-samples", type=int, default=256)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
