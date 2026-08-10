#!/usr/bin/env python3
"""Extract query-agnostic CLIP leaf features for official Ego4D NLQ clips."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import CLIPModel


CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073))
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711))


def read_clips(path: Path, stride_seconds: float) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = []
    for video in payload["videos"]:
        for clip in video["clips"]:
            duration = float(clip["video_end_sec"]) - float(clip["video_start_sec"])
            leaves = max(1, int(np.ceil(duration / stride_seconds)))
            clips.append(
                {
                    "video_uid": video["video_uid"],
                    "clip_uid": clip["clip_uid"],
                    "video_start_sec": float(clip["video_start_sec"]),
                    "duration": duration,
                    "leaves": leaves,
                }
            )
    return clips


def balanced_video_shards(clips: list[dict], num_shards: int) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for clip in clips:
        counts[clip["video_uid"]] += clip["leaves"]
    loads = [0] * num_shards
    assignment = {}
    for video_uid, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        assignment[video_uid] = shard
        loads[shard] += count
    return assignment


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


class ClipImageEmbedder:
    def __init__(self, model_id: str, batch_size: int):
        self.model = CLIPModel.from_pretrained(model_id, local_files_only=True).to("cuda").eval()
        self.batch_size = batch_size

    def encode(self, frames: np.ndarray) -> np.ndarray:
        outputs = []
        mean = CLIP_MEAN[:, None, None]
        std = CLIP_STD[:, None, None]
        with torch.inference_mode():
            for start in range(0, len(frames), self.batch_size):
                batch = torch.from_numpy(frames[start : start + self.batch_size]).permute(0, 3, 1, 2)
                batch = batch.float().div_(255.0)
                batch = ((batch - mean) / std).to("cuda", non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.get_image_features(pixel_values=batch)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output
                features = torch.nn.functional.normalize(features.float(), dim=-1)
                outputs.append(features.cpu().numpy().astype(np.float16))
        return np.concatenate(outputs, axis=0)


def center_crop(frames: np.ndarray, size: int) -> np.ndarray:
    height, width = frames.shape[1:3]
    top = (height - size) // 2
    left = (width - size) // 2
    return frames[:, top : top + size, left : left + size]


def decode_video(
    video_uid: str,
    clips: list[dict],
    video_root: str,
    resize: int,
    decode_threads: int,
    decode_batch_size: int,
    stride_seconds: float,
) -> tuple[str, list[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]]:
    path = Path(video_root) / f"{video_uid}.mp4"
    reader = open_video_reader(path, resize, decode_threads)
    fps = float(reader.get_avg_fps())
    decoded = []
    for clip in clips:
        rel_starts = np.arange(clip["leaves"], dtype=np.float64) * stride_seconds
        rel_stops = np.minimum(rel_starts + stride_seconds, clip["duration"])
        centers = clip["video_start_sec"] + (rel_starts + rel_stops) / 2.0
        frame_indices = np.rint(centers * fps).astype(np.int64)
        frame_indices = np.clip(frame_indices, 0, len(reader) - 1)
        chunks = []
        for start in range(0, len(frame_indices), decode_batch_size):
            batch_indices = frame_indices[start : start + decode_batch_size]
            chunks.append(center_crop(reader.get_batch(batch_indices).asnumpy(), resize))
        decoded.append((clip, np.concatenate(chunks), rel_starts, rel_stops))
    return video_uid, decoded


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    output = Path(args.output_dir) / f"{args.split}_shard_{args.shard_id:03d}.npz"
    if output.exists() and not args.overwrite:
        print(f"Already exists: {output}", flush=True)
        return
    clips = read_clips(Path(args.annotation), args.stride_seconds)
    assignment = balanced_video_shards(clips, args.num_shards)
    selected = [clip for clip in clips if assignment[clip["video_uid"]] == args.shard_id]
    selected.sort(key=lambda clip: (clip["video_uid"], clip["video_start_sec"], clip["clip_uid"]))
    if args.max_clips:
        selected = selected[: args.max_clips]
    by_video: dict[str, list[dict]] = defaultdict(list)
    for clip in selected:
        by_video[clip["video_uid"]].append(clip)

    print(
        json.dumps(
            {
                "split": args.split,
                "shard": args.shard_id,
                "clips": len(selected),
                "videos": len(by_video),
                "leaves": sum(clip["leaves"] for clip in selected),
                "gpu": torch.cuda.get_device_name(0),
            }
        ),
        flush=True,
    )
    embedder = ClipImageEmbedder(args.model_id, args.batch_size)
    clip_uids: list[str] = []
    video_uids: list[str] = []
    leaf_indices: list[int] = []
    starts: list[float] = []
    stops: list[float] = []
    feature_chunks: list[np.ndarray] = []
    failures = []
    processed = 0
    started = time.time()

    items = iter(by_video.items())
    with ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
        pending = {}

        def submit_next() -> bool:
            try:
                video_uid, video_clips = next(items)
            except StopIteration:
                return False
            future = executor.submit(
                decode_video,
                video_uid,
                video_clips,
                args.video_root,
                args.resize,
                args.decode_threads,
                args.decode_batch_size,
                args.stride_seconds,
            )
            pending[future] = video_uid
            return True

        for _ in range(args.decode_workers * 2):
            if not submit_next():
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                video_uid = pending.pop(future)
                try:
                    _, decoded_clips = future.result()
                    for clip, frames, rel_starts, rel_stops in decoded_clips:
                        feature_chunks.append(embedder.encode(frames))
                        clip_uids.extend([clip["clip_uid"]] * clip["leaves"])
                        video_uids.extend([video_uid] * clip["leaves"])
                        leaf_indices.extend(range(clip["leaves"]))
                        starts.extend(rel_starts.tolist())
                        stops.extend(rel_stops.tolist())
                        processed += clip["leaves"]
                        if processed // args.log_every != (
                            processed - clip["leaves"]
                        ) // args.log_every:
                            elapsed = time.time() - started
                            print(
                                f"shard={args.shard_id} leaves={processed} "
                                f"rate={processed / max(elapsed, 1e-6):.2f}/s "
                                f"failures={len(failures)}",
                                flush=True,
                            )
                except Exception as exc:
                    failures.append({"video_uid": video_uid, "error": str(exc)})
                    print(f"failure={failures[-1]}", flush=True)
                submit_next()

    if not feature_chunks:
        raise RuntimeError("No features were extracted")
    payload = {
        "clip_uid": np.asarray(clip_uids),
        "video_uid": np.asarray(video_uids),
        "leaf_index": np.asarray(leaf_indices, dtype=np.int16),
        "start_sec": np.asarray(starts, dtype=np.float32),
        "end_sec": np.asarray(stops, dtype=np.float32),
        "feature": np.concatenate(feature_chunks, axis=0),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, output)
    output.with_suffix(".failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "leaves": processed,
                "failed_videos": len(failures),
                "seconds": round(time.time() - started, 2),
                "peak_gpu_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            }
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--video-root", default="/dataset/Ego4D/v2/full_scale")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decode-threads", type=int, default=1)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
