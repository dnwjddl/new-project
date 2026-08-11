#!/usr/bin/env python3
"""Pool full-split VideoFlexTok features into a compact EK100 stream cache."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-key", default="prequant_ordered")
    parser.add_argument("--source-slots", type=int, default=256)
    parser.add_argument("--pooled-slots", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_annotations(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["narration_id"]: row for row in csv.DictReader(handle)}


def load_split(
    feature_dir: Path,
    split: str,
    feature_key: str,
    source_slots: int,
    pooled_slots: int,
    annotations: dict[str, dict[str, str]],
) -> dict:
    paths = sorted(feature_dir.glob(f"{split}_shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards under {feature_dir}")
    if source_slots % pooled_slots:
        raise ValueError("source-slots must be divisible by pooled-slots")

    pooled_parts: list[np.ndarray] = []
    narration_ids: list[str] = []
    video_ids: list[str] = []
    verbs: list[int] = []
    nouns: list[int] = []
    start_frames: list[int] = []
    missing = 0

    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            flat = payload[feature_key]
            if flat.shape[1] % source_slots:
                raise ValueError(f"Cannot reshape {path}: {flat.shape}")
            token_dim = flat.shape[1] // source_slots
            tokens = flat.astype(np.float32).reshape(-1, source_slots, token_dim)
            group = source_slots // pooled_slots
            pooled = tokens.reshape(-1, pooled_slots, group, token_dim).mean(axis=2)

            keep: list[int] = []
            for index, narration_id in enumerate(payload["narration_id"].tolist()):
                row = annotations.get(str(narration_id))
                if row is None:
                    missing += 1
                    continue
                keep.append(index)
                narration_ids.append(str(narration_id))
                video_ids.append(row["video_id"])
                verbs.append(int(row["verb_class"]))
                nouns.append(int(row["noun_class"]))
                start_frames.append(int(row["start_frame"]))
            pooled_parts.append(pooled[np.asarray(keep, dtype=np.int64)])
        print(f"loaded {path.name}: {len(keep)} samples", flush=True)

    features = np.concatenate(pooled_parts, axis=0)
    order = np.lexsort((np.asarray(start_frames), np.asarray(video_ids)))
    return {
        "features": features[order],
        "narration_ids": [narration_ids[index] for index in order],
        "video_ids": [video_ids[index] for index in order],
        "verbs": np.asarray(verbs, dtype=np.int64)[order],
        "nouns": np.asarray(nouns, dtype=np.int64)[order],
        "start_frames": np.asarray(start_frames, dtype=np.int64)[order],
        "missing_annotations": missing,
    }


def make_sequences(video_ids: list[str]) -> list[torch.Tensor]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, video_id in enumerate(video_ids):
        grouped[video_id].append(index)
    return [torch.tensor(indices, dtype=torch.long) for _, indices in sorted(grouped.items())]


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"Already exists: {args.output}")
        return

    train_annotations = read_annotations(args.annotation_root / "EPIC_100_train.csv")
    validation_annotations = read_annotations(args.annotation_root / "EPIC_100_validation.csv")
    train = load_split(
        args.feature_dir,
        "train",
        args.feature_key,
        args.source_slots,
        args.pooled_slots,
        train_annotations,
    )
    validation = load_split(
        args.feature_dir,
        "validation",
        args.feature_key,
        args.source_slots,
        args.pooled_slots,
        validation_annotations,
    )

    mean = train["features"].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train["features"].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    action_pairs = sorted(set(zip(train["verbs"].tolist(), train["nouns"].tolist())))
    action_map = {pair: index for index, pair in enumerate(action_pairs)}

    output_splits = {}
    for split_name, split in (("train", train), ("validation", validation)):
        normalized = np.clip((split["features"] - mean) / std, -8.0, 8.0).astype(np.float16)
        actions = np.asarray(
            [action_map.get((int(verb), int(noun)), -1) for verb, noun in zip(split["verbs"], split["nouns"])],
            dtype=np.int64,
        )
        output_splits[split_name] = {
            "features": torch.from_numpy(normalized),
            "verbs": torch.from_numpy(split["verbs"]),
            "nouns": torch.from_numpy(split["nouns"]),
            "actions": torch.from_numpy(actions),
            "sequences": make_sequences(split["video_ids"]),
            "video_ids": split["video_ids"],
            "narration_ids": split["narration_ids"],
            "start_frames": torch.from_numpy(split["start_frames"]),
        }

    payload = {
        "format_version": 1,
        "feature_key": args.feature_key,
        "source_slots": args.source_slots,
        "pooled_slots": args.pooled_slots,
        "token_dim": int(train["features"].shape[-1]),
        "feature_mean": torch.from_numpy(mean),
        "feature_std": torch.from_numpy(std),
        "num_verbs": int(max(train["verbs"].max(), validation["verbs"].max()) + 1),
        "num_nouns": int(max(train["nouns"].max(), validation["nouns"].max()) + 1),
        "num_actions": len(action_pairs),
        "action_pairs": action_pairs,
        "splits": output_splits,
        "metadata": {
            "train_samples": len(train["features"]),
            "validation_samples": len(validation["features"]),
            "train_videos": len(output_splits["train"]["sequences"]),
            "validation_videos": len(output_splits["validation"]["sequences"]),
            "train_missing_annotations": train["missing_annotations"],
            "validation_missing_annotations": validation["missing_annotations"],
            "validation_unknown_actions": int((output_splits["validation"]["actions"] < 0).sum()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(json.dumps(payload["metadata"], indent=2), flush=True)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
