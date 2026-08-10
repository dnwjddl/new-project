#!/usr/bin/env python3
"""Evaluate query-agnostic trees and persistent branch reuse on Ego4D NLQ."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, CLIPModel


@dataclass
class ClipLeaves:
    features: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


@dataclass
class QueryRecord:
    query_id: str
    clip_uid: str
    text: str
    start: float
    end: float


@dataclass
class TreeNode:
    start: int
    stop: int
    feature: np.ndarray
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.stop - self.start == 1


class Router(nn.Module):
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.text = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_visual(self, values: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.visual(values), dim=-1)

    def encode_text(self, values: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.text(values), dim=-1)


class ResidualRouter(nn.Module):
    def __init__(self, input_dim: int, scale: float):
        super().__init__()
        self.visual_norm = nn.LayerNorm(input_dim)
        self.text_norm = nn.LayerNorm(input_dim)
        self.visual_delta = nn.Linear(input_dim, input_dim, bias=False)
        self.text_delta = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.zeros_(self.visual_delta.weight)
        nn.init.zeros_(self.text_delta.weight)
        self.scale = scale
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_visual(self, values: torch.Tensor) -> torch.Tensor:
        reference = nn.functional.normalize(values, dim=-1)
        return nn.functional.normalize(
            reference + self.scale * self.visual_delta(self.visual_norm(values)), dim=-1
        )

    def encode_text(self, values: torch.Tensor) -> torch.Tensor:
        reference = nn.functional.normalize(values, dim=-1)
        return nn.functional.normalize(
            reference + self.scale * self.text_delta(self.text_norm(values)), dim=-1
        )


def load_leaves(directory: Path, split: str) -> dict[str, ClipLeaves]:
    grouped: dict[str, list[tuple[int, float, float, np.ndarray]]] = defaultdict(list)
    paths = sorted(directory.glob(f"{split}_shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards in {directory}")
    for path in paths:
        with np.load(path) as data:
            clip_uids = data["clip_uid"]
            leaf_indices = data["leaf_index"]
            starts = data["start_sec"]
            ends = data["end_sec"]
            features = data["feature"]
            for index, clip_uid in enumerate(clip_uids):
                grouped[str(clip_uid)].append(
                    (
                        int(leaf_indices[index]),
                        float(starts[index]),
                        float(ends[index]),
                        features[index].copy(),
                    )
                )
    output = {}
    for clip_uid, rows in grouped.items():
        rows.sort(key=lambda row: row[0])
        output[clip_uid] = ClipLeaves(
            features=np.stack([row[3] for row in rows]),
            starts=np.asarray([row[1] for row in rows], dtype=np.float32),
            ends=np.asarray([row[2] for row in rows], dtype=np.float32),
        )
    return output


def load_queries(path: Path) -> list[QueryRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for video in payload["videos"]:
        for clip in video["clips"]:
            for annotation in clip["annotations"]:
                for index, query in enumerate(annotation["language_queries"]):
                    if not query.get("query"):
                        continue
                    records.append(
                        QueryRecord(
                            query_id=f"{annotation['annotation_uid']}:{index}",
                            clip_uid=clip["clip_uid"],
                            text=query["query"],
                            start=float(query["clip_start_sec"]),
                            end=float(query["clip_end_sec"]),
                        )
                    )
    return records


def count_query_entries(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        len(annotation["language_queries"])
        for video in payload["videos"]
        for clip in video["clips"]
        for annotation in clip["annotations"]
    )


def encode_texts(
    records: list[QueryRecord], cache: Path, model_id: str, batch_size: int
) -> dict[str, np.ndarray]:
    expected = {record.query_id for record in records}
    if cache.exists():
        with np.load(cache) as data:
            cached = {
                str(query_id): feature.astype(np.float32)
                for query_id, feature in zip(data["query_id"], data["feature"])
            }
        if set(cached) == expected:
            return cached

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = CLIPModel.from_pretrained(model_id, local_files_only=True).to("cuda").eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            tokens = tokenizer(
                [record.text for record in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = model.get_text_features(**tokens)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            outputs.append(nn.functional.normalize(features.float(), dim=-1).cpu().numpy())
    features = np.concatenate(outputs).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, query_id=np.asarray([record.query_id for record in records]), feature=features)
    model.to("cpu")
    torch.cuda.empty_cache()
    return {record.query_id: features[index] for index, record in enumerate(records)}


def positive_leaf(leaves: ClipLeaves, query: QueryRecord) -> int:
    center = (query.start + query.end) / 2.0
    return int(np.argmin(np.abs((leaves.starts + leaves.ends) / 2.0 - center)))


def train_router(
    seed: int,
    clips: dict[str, ClipLeaves],
    records: list[QueryRecord],
    text: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Router | ResidualRouter:
    usable = [record for record in records if record.clip_uid in clips]
    text_values = np.stack([text[record.query_id] for record in usable]).astype(np.float32)
    positive_indices = [positive_leaf(clips[record.clip_uid], record) for record in usable]
    visual_values = np.stack(
        [
            clips[record.clip_uid].features[index]
            for record, index in zip(usable, positive_indices)
        ]
    ).astype(np.float32)
    positive_keys = np.asarray(
        [f"{record.clip_uid}:{index}" for record, index in zip(usable, positive_indices)]
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.router_type == "linear":
        model = Router(text_values.shape[1], args.router_width).to("cuda")
    else:
        model = ResidualRouter(text_values.shape[1], args.residual_scale).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(usable), generator=generator).numpy()
        running = 0.0
        steps = 0
        model.train()
        for start in range(0, len(usable), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            batch_text = torch.from_numpy(text_values[indices]).to("cuda")
            batch_visual = torch.from_numpy(visual_values[indices]).to("cuda")
            encoded_text = model.encode_text(batch_text)
            encoded_visual = model.encode_visual(batch_visual)
            logits = model.logit_scale.exp().clamp(max=100) * encoded_text @ encoded_visual.T
            keys = positive_keys[indices]
            positives = torch.from_numpy(keys[:, None] == keys[None, :]).to("cuda")
            text_targets = positives.float() / positives.sum(dim=1, keepdim=True)
            visual_targets = positives.T.float() / positives.T.sum(dim=1, keepdim=True)
            text_loss = -(text_targets * logits.log_softmax(dim=1)).sum(dim=1).mean()
            visual_loss = -(visual_targets * logits.T.log_softmax(dim=1)).sum(dim=1).mean()
            loss = (text_loss + visual_loss) / 2.0
            if args.alignment_weight:
                reference_text = nn.functional.normalize(batch_text, dim=-1)
                reference_visual = nn.functional.normalize(batch_visual, dim=-1)
                alignment = (
                    (1.0 - (encoded_text * reference_text).sum(dim=-1)).mean()
                    + (1.0 - (encoded_visual * reference_visual).sum(dim=-1)).mean()
                ) / 2.0
                loss = loss + args.alignment_weight * alignment
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        print(f"seed={seed} epoch={epoch + 1}/{args.epochs} loss={running / steps:.5f}", flush=True)
    return model.eval()


def project_clips(
    model: Router | ResidualRouter,
    clips: dict[str, ClipLeaves],
    batch_size: int,
) -> dict[str, ClipLeaves]:
    output = {}
    with torch.inference_mode():
        for clip_uid, leaves in clips.items():
            chunks = []
            for start in range(0, len(leaves.features), batch_size):
                values = torch.from_numpy(
                    leaves.features[start : start + batch_size].astype(np.float32)
                ).to("cuda")
                chunks.append(model.encode_visual(values).cpu().numpy())
            output[clip_uid] = ClipLeaves(
                features=np.concatenate(chunks), starts=leaves.starts, ends=leaves.ends
            )
    return output


def project_text(
    model: Router | ResidualRouter,
    records: list[QueryRecord],
    text: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    values = np.stack([text[record.query_id] for record in records]).astype(np.float32)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(values), 2048):
            batch = torch.from_numpy(values[start : start + 2048]).to("cuda")
            chunks.append(model.encode_text(batch).cpu().numpy())
    projected = np.concatenate(chunks)
    return {record.query_id: projected[index] for index, record in enumerate(records)}


def normalized_mean(features: np.ndarray, start: int, stop: int) -> np.ndarray:
    value = features[start:stop].astype(np.float32).mean(axis=0)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def build_tree(features: np.ndarray, mode: str, start: int = 0, stop: int | None = None) -> TreeNode:
    stop = len(features) if stop is None else stop
    node = TreeNode(start=start, stop=stop, feature=normalized_mean(features, start, stop))
    if stop - start == 1:
        return node
    if mode == "uniform":
        split = (start + stop) // 2
    else:
        low = start + max(1, (stop - start) // 4)
        high = stop - max(1, (stop - start) // 4)
        candidates = np.arange(low, high + 1)
        if len(candidates) == 0:
            split = (start + stop) // 2
        else:
            distances = 1.0 - np.sum(features[candidates - 1] * features[candidates], axis=1)
            split = int(candidates[int(np.argmax(distances))])
    node.left = build_tree(features, mode, start, split)
    node.right = build_tree(features, mode, split, stop)
    return node


def route_flat_budgets(
    features: np.ndarray, query: np.ndarray, budgets: list[int]
) -> dict[int, tuple[list[int], int]]:
    scores = features @ query
    count = min(max(budgets), len(features))
    indices = np.argpartition(-scores, count - 1)[:count]
    indices = indices[np.argsort(-scores[indices])]
    return {
        budget: (indices[: min(budget, len(indices))].tolist(), len(features))
        for budget in budgets
    }


def route_tree_budgets(
    root: TreeNode, query: np.ndarray, budgets: list[int]
) -> dict[int, tuple[list[int], int]]:
    frontier = [(-float(root.feature @ query), 0, root)]
    selected = []
    comparison_history = []
    comparisons = 1
    serial = 1
    max_budget = max(budgets)
    while frontier and len(selected) < max_budget:
        _, _, node = heapq.heappop(frontier)
        if node.is_leaf:
            selected.append(node.start)
            comparison_history.append(comparisons)
            continue
        for child in (node.left, node.right):
            heapq.heappush(frontier, (-float(child.feature @ query), serial, child))
            serial += 1
            comparisons += 1
    return {
        budget: (
            selected[: min(budget, len(selected))],
            comparison_history[min(budget, len(selected)) - 1],
        )
        for budget in budgets
    }


def interval_iou(start: float, end: float, gt_start: float, gt_end: float) -> float:
    intersection = max(0.0, min(end, gt_end) - max(start, gt_start))
    union = max(end, gt_end) - min(start, gt_start)
    return intersection / union if union else 0.0


def metric_template() -> dict[str, float]:
    return defaultdict(float)


def evaluate_space(
    clips: dict[str, ClipLeaves],
    records: list[QueryRecord],
    text: dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> dict:
    trees = {
        clip_uid: {
            "uniform_tree": build_tree(leaves.features, "uniform"),
            "change_tree": build_tree(leaves.features, "change"),
        }
        for clip_uid, leaves in clips.items()
    }
    query_routes: dict[tuple[str, int, str], tuple[list[int], int]] = {}
    accuracy = {
        method: {str(budget): metric_template() for budget in args.budgets}
        for method in ("flat", "uniform_tree", "change_tree")
    }
    usable = [record for record in records if record.clip_uid in clips]
    for record in usable:
        leaves = clips[record.clip_uid]
        query = text[record.query_id]
        gt_center = (record.start + record.end) / 2.0
        duration = max(float(leaves.ends[-1]), 1e-6)
        for method in accuracy:
            if method == "flat":
                routes = route_flat_budgets(leaves.features, query, args.budgets)
            else:
                routes = route_tree_budgets(
                    trees[record.clip_uid][method], query, args.budgets
                )
            for budget, (selected, comparisons) in routes.items():
                query_routes[(record.query_id, budget, method)] = (selected, comparisons)
                first = selected[0]
                overlap = any(
                    leaves.ends[index] > record.start and leaves.starts[index] < record.end
                    for index in selected
                )
                center_hit = any(
                    leaves.starts[index] <= gt_center <= leaves.ends[index] for index in selected
                )
                row = accuracy[method][str(budget)]
                row["overlap_hit"] += float(overlap)
                row["center_hit"] += float(center_hit)
                row["top1_iou"] += interval_iou(
                    float(leaves.starts[first]), float(leaves.ends[first]), record.start, record.end
                )
                pred_center = float((leaves.starts[first] + leaves.ends[first]) / 2.0)
                row["normalized_center_error"] += abs(pred_center - gt_center) / duration
                row["router_comparisons"] += comparisons

    for method in accuracy:
        for budget in args.budgets:
            row = accuracy[method][str(budget)]
            for key in tuple(row):
                row[key] /= len(usable)
            row["queries"] = len(usable)

    grouped: dict[str, list[QueryRecord]] = defaultdict(list)
    for record in usable:
        grouped[record.clip_uid].append(record)
    amortization = {
        method: {
            str(budget): {str(count): metric_template() for count in args.query_counts}
            for budget in args.budgets
        }
        for method in accuracy
    }
    rng = random.Random(seed + 7001)
    for clip_uid, clip_queries in grouped.items():
        leaves = clips[clip_uid]
        for query_count in args.query_counts:
            if len(clip_queries) < query_count:
                continue
            episodes = min(args.episodes_per_clip, math.comb(len(clip_queries), query_count))
            for _ in range(episodes):
                episode = rng.sample(clip_queries, query_count)
                text_values = np.stack([text[record.query_id] for record in episode])
                similarity = text_values @ text_values.T
                upper = similarity[np.triu_indices(query_count, k=1)]
                semantic_similarity = float(upper.mean()) if len(upper) else 1.0
                for method in accuracy:
                    for budget in args.budgets:
                        selections = [
                            query_routes[(record.query_id, budget, method)][0] for record in episode
                        ]
                        comparisons = sum(
                            query_routes[(record.query_id, budget, method)][1] for record in episode
                        )
                        opened = set().union(*(set(values) for values in selections))
                        independent = sum(len(values) for values in selections)
                        no_reuse_cost = query_count * len(leaves.features) + independent
                        shared_cost = len(leaves.features) + len(opened)
                        row = amortization[method][str(budget)][str(query_count)]
                        row["unique_leaves_per_query"] += len(opened) / query_count
                        row["branch_cache_ratio"] += len(opened) / max(independent, 1)
                        row["total_visual_cost_ratio"] += shared_cost / no_reuse_cost
                        row["router_comparisons_per_query"] += comparisons / query_count
                        row["query_similarity"] += semantic_similarity
                        row["episodes"] += 1
    for method in amortization:
        for budget in args.budgets:
            for query_count in args.query_counts:
                row = amortization[method][str(budget)][str(query_count)]
                episodes = row["episodes"]
                if episodes:
                    for key in tuple(row):
                        if key != "episodes":
                            row[key] /= episodes
                    row["episodes"] = int(episodes)
    return {"accuracy": accuracy, "amortization": amortization}


def aggregate(rows: list[dict]) -> dict:
    def recurse(values: list):
        if isinstance(values[0], dict):
            return {key: recurse([value[key] for value in values]) for key in values[0]}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }

    return recurse(rows)


def normalize_clip_space(clips: dict[str, ClipLeaves]) -> dict[str, ClipLeaves]:
    output = {}
    for clip_uid, leaves in clips.items():
        features = leaves.features.astype(np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
        output[clip_uid] = ClipLeaves(features, leaves.starts, leaves.ends)
    return output


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.router_type == "linear" and args.alignment_weight:
        raise ValueError("Alignment regularization requires --router-type residual")
    directory = Path(args.feature_dir)
    train_clips = load_leaves(directory, "train")
    val_clips = load_leaves(directory, "val")
    train_annotation = Path(args.train_annotation)
    val_annotation = Path(args.val_annotation)
    train_queries = load_queries(train_annotation)
    val_queries = load_queries(val_annotation)
    usable_train_queries = [
        record for record in train_queries if record.clip_uid in train_clips
    ]
    usable_val_queries = [record for record in val_queries if record.clip_uid in val_clips]
    all_queries = train_queries + val_queries
    text = encode_texts(all_queries, Path(args.text_cache), args.model_id, args.text_batch_size)

    raw_clips = normalize_clip_space(val_clips)
    raw_text = {key: value / max(float(np.linalg.norm(value)), 1e-8) for key, value in text.items()}
    raw = evaluate_space(raw_clips, val_queries, raw_text, args, args.seeds[0])
    learned = []
    per_seed = []
    for seed in args.seeds:
        router = train_router(seed, train_clips, train_queries, text, args)
        projected_clips = project_clips(router, val_clips, args.project_batch_size)
        projected_text = project_text(router, val_queries, text)
        result = evaluate_space(projected_clips, val_queries, projected_text, args, seed)
        learned.append(result)
        per_seed.append({"seed": seed, "metrics": result})
        router.to("cpu")
        torch.cuda.empty_cache()

    output = {
        "config": vars(args),
        "samples": {
            "train_clips": len(train_clips),
            "val_clips": len(val_clips),
            "train_queries": len(train_queries),
            "val_queries": len(val_queries),
            "usable_train_queries": len(usable_train_queries),
            "usable_val_queries": len(usable_val_queries),
            "excluded_missing_query_text": (
                count_query_entries(train_annotation)
                + count_query_entries(val_annotation)
                - len(train_queries)
                - len(val_queries)
            ),
            "train_leaves": sum(len(leaves.features) for leaves in train_clips.values()),
            "val_leaves": sum(len(leaves.features) for leaves in val_clips.values()),
        },
        "raw_clip": raw,
        "learned_per_seed": per_seed,
        "learned_aggregate": aggregate(learned),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["samples"], indent=2), flush=True)
    print(f"Saved {path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--train-annotation", required=True)
    parser.add_argument("--val-annotation", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    parser.add_argument("--budgets", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--query-counts", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--router-width", type=int, default=256)
    parser.add_argument("--router-type", choices=("linear", "residual"), default="linear")
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--alignment-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--project-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--episodes-per-clip", type=int, default=8)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
