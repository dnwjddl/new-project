#!/usr/bin/env python3
"""Extract ViCLIP features for full-split EPIC-Kitchens temporal audits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


ALL_CONDITIONS = ("ordered", "reverse", "shuffle", "single", "sparse", "freeze")


def hf_snapshot(model_id: str) -> Path:
    repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    ref = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    return repo_dir / "snapshots" / ref


class ViClipEmbedder:
    def __init__(self, model_id: str, bpe_path: str, batch_size: int, resize: int):
        self.device = "cuda"
        self.batch_size = batch_size
        self.resize = resize
        snapshot = hf_snapshot(model_id)
        config = AutoConfig.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=True,
        )
        config.tokenizer_path = bpe_path
        model_class = get_class_from_dynamic_module(
            "viclip.ViCLIP",
            model_id,
            local_files_only=True,
        )
        model = model_class(config)
        state = load_file(str(snapshot / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"ViCLIP load mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        self.model = model.to(self.device).eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)

    def _tensorize(self, video: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
        return (tensor - self.mean[:, None, None]) / self.std[:, None, None]

    def embed(self, videos: list[np.ndarray]) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(videos), self.batch_size):
                batch = torch.stack(
                    [self._tensorize(video) for video in videos[start : start + self.batch_size]]
                ).to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.get_vid_features(batch)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
                outputs.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"narration_id", "video_id", "start_frame", "stop_frame", "verb_class", "noun_class"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Annotation columns missing: {sorted(missing)}")
    return rows


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


def resolve_video(root: Path, split: str, video_id: str) -> Path:
    preferred_dir = "train" if split == "train" else "test"
    search_dirs = (preferred_dir, "ek100", "train", "test")
    candidates = tuple(
        root / directory / f"{video_id}{extension}"
        for directory in search_dirs
        for extension in (".MP4", ".mp4")
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Video not found for {video_id}: {candidates}")


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
    if condition == "sparse":
        unique = frames[::2]
        return np.repeat(unique, 2, axis=0)[: len(frames)]
    if condition == "freeze":
        frozen = frames.copy()
        frozen[len(frames) // 2 :] = frames[len(frames) // 2 - 1]
        return frozen
    raise ValueError(f"Unknown condition: {condition}")


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(conditions).difference(ALL_CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    return conditions


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for feature extraction")
    output_path = Path(args.output_dir) / f"{args.split}_shard_{args.shard_id:03d}.npz"
    if output_path.exists() and not args.overwrite:
        print(f"Already exists: {output_path}", flush=True)
        return

    rows = read_annotations(Path(args.annotation))
    assignments = balanced_video_shards(rows, args.num_shards)
    selected = [row for row in rows if assignments[row["video_id"]] == args.shard_id]
    selected.sort(key=lambda row: (row["video_id"], int(row["start_frame"])))
    if args.max_samples:
        selected = selected[: args.max_samples]
    conditions = parse_conditions(args.conditions)
    if args.split == "train" and conditions != ("ordered",):
        raise ValueError("Train extraction must use only the ordered condition")

    print(
        json.dumps(
            {
                "split": args.split,
                "shard": args.shard_id,
                "num_shards": args.num_shards,
                "samples": len(selected),
                "conditions": conditions,
                "gpu": torch.cuda.get_device_name(0),
            }
        ),
        flush=True,
    )
    embedder = ViClipEmbedder(args.model_id, args.bpe_path, args.model_batch_size, args.resize)
    feature_chunks: dict[str, list[np.ndarray]] = {condition: [] for condition in conditions}
    narration_ids: list[str] = []
    video_ids: list[str] = []
    verb_labels: list[int] = []
    noun_labels: list[int] = []
    failures = []
    pending: dict[str, list[np.ndarray]] = {condition: [] for condition in conditions}
    pending_rows: list[dict[str, str]] = []
    started = time.time()

    def flush() -> None:
        if not pending_rows:
            return
        for condition in conditions:
            feature_chunks[condition].append(embedder.embed(pending[condition]))
            pending[condition].clear()
        for row in pending_rows:
            narration_ids.append(row["narration_id"])
            video_ids.append(row["video_id"])
            verb_labels.append(int(row["verb_class"]))
            noun_labels.append(int(row["noun_class"]))
        pending_rows.clear()

    current_video = None
    reader = None
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
            if len(pending_rows) >= args.sample_batch_size:
                flush()
        except Exception as exc:
            failures.append({"narration_id": row["narration_id"], "error": str(exc)})
            if len(failures) <= 10:
                print(f"failure={failures[-1]}", flush=True)
        if index % args.log_every == 0:
            elapsed = time.time() - started
            rate = index / max(elapsed, 1e-6)
            print(
                f"shard={args.shard_id} {index}/{len(selected)} rate={rate:.2f} samples/s "
                f"failures={len(failures)}",
                flush=True,
            )
    flush()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "narration_id": np.asarray(narration_ids),
        "video_id": np.asarray(video_ids),
        "verb": np.asarray(verb_labels, dtype=np.int16),
        "noun": np.asarray(noun_labels, dtype=np.int16),
    }
    for condition, chunks in feature_chunks.items():
        payload[f"feature_{condition}"] = np.concatenate(chunks, axis=0)
    np.savez(output_path, **payload)
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
    parser.add_argument("--model-id", default="OpenGVLab/ViCLIP-L-14-hf")
    parser.add_argument(
        "--bpe-path", default="/home/dnwjddl/project1/vendor/bpe_simple_vocab_16e6.txt.gz"
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--sample-batch-size", type=int, default=12)
    parser.add_argument("--model-batch-size", type=int, default=12)
    parser.add_argument("--decode-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
