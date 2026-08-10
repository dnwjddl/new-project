#!/usr/bin/env python3
"""Extract released VideoFlexTok prefix representations on full EK100 splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from videoflextok.utils.misc import detect_bf16_support, get_bf16_context
from videoflextok.wrappers import VideoFlexTokFromHub


ALL_CONDITIONS = ("ordered", "reverse", "shuffle", "single")


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def balanced_video_shards(rows: list[dict[str, str]], num_shards: int) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["video_id"]] += 1
    loads = [0] * num_shards
    assignment = {}
    for video_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        assignment[video_id] = shard
        loads[shard] += count
    return assignment


def stratified_subsample(
    rows: list[dict[str, str]], total: int, seed: int
) -> list[dict[str, str]]:
    if total <= 0 or total >= len(rows):
        return rows
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["verb_class"], row["noun_class"])].append(row)
    if total < len(groups):
        raise ValueError(f"Need at least {len(groups)} samples to cover every action stratum")

    allocation = {key: 1 for key in groups}
    remaining = total - len(groups)
    capacity = {key: len(values) - 1 for key, values in groups.items()}
    total_capacity = sum(capacity.values())
    quotas = {
        key: (remaining * value / total_capacity if total_capacity else 0.0)
        for key, value in capacity.items()
    }
    for key, quota in quotas.items():
        allocation[key] += min(capacity[key], int(np.floor(quota)))
    leftover = total - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (
            -(quotas[key] - np.floor(quotas[key])),
            -capacity[key],
            key,
        ),
    )
    for key in order:
        if leftover == 0:
            break
        if allocation[key] < len(groups[key]):
            allocation[key] += 1
            leftover -= 1
    if leftover:
        raise RuntimeError(f"Could not allocate {leftover} stratified samples")

    rng = random.Random(seed)
    selected_ids = set()
    for key in sorted(groups):
        values = list(groups[key])
        rng.shuffle(values)
        selected_ids.update(row["narration_id"] for row in values[: allocation[key]])
    selected = [row for row in rows if row["narration_id"] in selected_ids]
    if len(selected) != total:
        raise RuntimeError(f"Expected {total} stratified rows, got {len(selected)}")
    return selected


def resolve_video(root: Path, split: str, video_id: str) -> Path:
    preferred_dir = "train" if split == "train" else "test"
    for directory in (preferred_dir, "ek100", "train", "test"):
        for extension in (".MP4", ".mp4"):
            path = root / directory / f"{video_id}{extension}"
            if path.exists():
                return path
    raise FileNotFoundError(video_id)


def open_video_reader(path: Path, crop_size: int, decode_threads: int) -> VideoReader:
    probe = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    height, width = probe[0].shape[:2]
    del probe
    if height <= width:
        target_height = crop_size
        target_width = max(crop_size, int(round(width * crop_size / height)))
    else:
        target_width = crop_size
        target_height = max(crop_size, int(round(height * crop_size / width)))
    return VideoReader(
        str(path),
        ctx=cpu(0),
        width=target_width,
        height=target_height,
        num_threads=decode_threads,
    )


def sample_segment(
    reader: VideoReader,
    start_seconds: float,
    stop_seconds: float,
    num_frames: int,
    crop_size: int,
) -> np.ndarray:
    fps = float(reader.get_avg_fps())
    start_frame = max(0, int(round(start_seconds * fps)))
    stop_frame = max(start_frame, int(round(stop_seconds * fps)))
    last_frame = min(max(start_frame, stop_frame), len(reader) - 1)
    indices = np.linspace(start_frame, last_frame, num_frames).round().astype(int)
    frames = reader.get_batch(indices).asnumpy()
    height, width = frames.shape[1:3]
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    frames = frames[:, top : top + crop_size, left : left + crop_size]
    return frames.astype(np.float32) / 255.0


def timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def perturb(frames: np.ndarray, condition: str, sample_seed: str) -> np.ndarray:
    if condition == "ordered":
        return frames
    if condition == "reverse":
        return frames[::-1].copy()
    if condition == "shuffle":
        indices = list(range(len(frames)))
        random.Random(sample_seed).shuffle(indices)
        if indices == list(range(len(frames))):
            indices = indices[1:] + indices[:1]
        return frames[indices]
    if condition == "single":
        center = frames[len(frames) // 2]
        return np.repeat(center[None], len(frames), axis=0)
    raise ValueError(condition)


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(conditions).difference(ALL_CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    return conditions


def cat_tensors(value) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        return torch.cat(value, dim=0)
    return value


def summarize_latents(latents: torch.Tensor, expected_tokens: int) -> torch.Tensor:
    if latents.ndim < 4:
        raise ValueError(f"Expected [B, time..., K, D] latents, got {tuple(latents.shape)}")
    if latents.shape[-2] != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} ordered token slots, got shape {tuple(latents.shape)}"
        )
    batch, tokens, dimension = latents.shape[0], latents.shape[-2], latents.shape[-1]
    temporal = latents.reshape(batch, -1, tokens, dimension).float()
    mean = temporal.mean(dim=1)
    std = temporal.std(dim=1, unbiased=False)
    delta = temporal[:, -1] - temporal[:, 0]
    return torch.cat((mean, std, delta), dim=-1).reshape(batch, -1)


class VideoFlexTokEmbedder:
    def __init__(self, model_id: str, expected_tokens: int):
        self.model = VideoFlexTokFromHub.from_pretrained(
            model_id,
            local_files_only=True,
        ).to("cuda").eval()
        self.base = self.model.video_tokenizer
        self.expected_tokens = expected_tokens
        self.enable_bf16 = detect_bf16_support()

    def encode(self, videos: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensor = torch.from_numpy(np.stack(videos)).permute(0, 4, 1, 2, 3)
        tensor = tensor.mul(2.0).sub(1.0).to("cuda", non_blocking=True)
        data = {self.base.vae.images_read_key: list(tensor.split(1, dim=0))}
        with torch.inference_mode(), get_bf16_context(self.enable_bf16):
            data = self.base.vae.encode(data)
            data = self.base.encoder(data)
            prequant = cat_tensors(data[self.base.regularizer.latents_read_key])
            prequant_features = summarize_latents(prequant, self.expected_tokens)
            data = self.base.regularizer(data)
            quant = cat_tensors(data[self.base.regularizer.quants_write_key])
            tokens = cat_tensors(data[self.base.regularizer.tokens_write_key])
            quant_features = summarize_latents(quant, self.expected_tokens)
        return (
            prequant_features.cpu().numpy().astype(np.float16),
            quant_features.cpu().numpy().astype(np.float16),
            tokens.cpu().numpy().astype(np.uint16),
        )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    output_path = Path(args.output_dir) / f"{args.split}_shard_{args.shard_id:03d}.npz"
    if output_path.exists() and not args.overwrite:
        print(f"Already exists: {output_path}", flush=True)
        return

    rows = read_annotations(Path(args.annotation))
    rows = stratified_subsample(rows, args.max_total_samples, args.subsample_seed)
    assignment = balanced_video_shards(rows, args.num_shards)
    selected = [row for row in rows if assignment[row["video_id"]] == args.shard_id]
    selected.sort(key=lambda row: (row["video_id"], int(row["start_frame"])))
    if args.max_samples:
        selected = selected[: args.max_samples]
    conditions = parse_conditions(args.conditions)
    if args.split == "train" and conditions != ("ordered",):
        raise ValueError("Train extraction must use ordered only")

    print(
        json.dumps(
            {
                "split": args.split,
                "shard": args.shard_id,
                "samples": len(selected),
                "global_samples": len(rows),
                "conditions": conditions,
                "gpu": torch.cuda.get_device_name(0),
            }
        ),
        flush=True,
    )
    embedder = VideoFlexTokEmbedder(args.model_id, args.expected_tokens)
    chunks = {
        condition: {"prequant": [], "quant": [], "tokens": []} for condition in conditions
    }
    pending = {condition: [] for condition in conditions}
    pending_rows: list[dict[str, str]] = []
    narration_ids: list[str] = []
    video_ids: list[str] = []
    verbs: list[int] = []
    nouns: list[int] = []
    failures = []
    current_video = None
    reader = None
    started = time.time()

    def flush() -> None:
        if not pending_rows:
            return
        for condition in conditions:
            prequant, quant, tokens = embedder.encode(pending[condition])
            chunks[condition]["prequant"].append(prequant)
            chunks[condition]["quant"].append(quant)
            chunks[condition]["tokens"].append(tokens)
            pending[condition].clear()
        for row in pending_rows:
            narration_ids.append(row["narration_id"])
            video_ids.append(row["video_id"])
            verbs.append(int(row["verb_class"]))
            nouns.append(int(row["noun_class"]))
        pending_rows.clear()

    for index, row in enumerate(selected, start=1):
        try:
            if row["video_id"] != current_video:
                path = resolve_video(Path(args.dataset_root), args.split, row["video_id"])
                reader = open_video_reader(path, args.resize, args.decode_threads)
                current_video = row["video_id"]
            frames = sample_segment(
                reader,
                timestamp_seconds(row["start_timestamp"]),
                timestamp_seconds(row["stop_timestamp"]),
                args.num_frames,
                args.resize,
            )
            for condition in conditions:
                pending[condition].append(perturb(frames, condition, row["narration_id"]))
            pending_rows.append(row)
            if len(pending_rows) >= args.batch_size:
                flush()
        except Exception as exc:
            failures.append({"narration_id": row["narration_id"], "error": str(exc)})
            if len(failures) <= 10:
                print(f"failure={failures[-1]}", flush=True)
        if index % args.log_every == 0:
            elapsed = time.time() - started
            print(
                f"shard={args.shard_id} {index}/{len(selected)} "
                f"rate={index / max(elapsed, 1e-6):.3f} samples/s failures={len(failures)}",
                flush=True,
            )
    flush()

    payload = {
        "narration_id": np.asarray(narration_ids),
        "video_id": np.asarray(video_ids),
        "verb": np.asarray(verbs, dtype=np.int16),
        "noun": np.asarray(nouns, dtype=np.int16),
    }
    for condition in conditions:
        for representation in ("prequant", "quant", "tokens"):
            payload[f"{representation}_{condition}"] = np.concatenate(
                chunks[condition][representation], axis=0
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, output_path)
    output_path.with_suffix(".failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "saved": len(narration_ids),
                "failed": len(failures),
                "seconds": round(time.time() - started, 2),
                "peak_gpu_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            }
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/dataset/EK100")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--conditions", default="ordered")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--model-id", default="EPFL-VILAB/videoflextok_d18_d18_k600")
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--resize", type=int, default=128)
    parser.add_argument("--expected-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-total-samples", type=int, default=0)
    parser.add_argument("--subsample-seed", type=int, default=941002)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
