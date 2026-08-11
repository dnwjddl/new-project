#!/usr/bin/env python3
"""Evaluate Gemma 4 E2B/E4B on Video-MME-v2 with native video input."""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROMPT = (
    "These are the frames of a video.\n"
    "Select the best answer to the following multiple-choice question based on "
    "the video. Respond with only the letter (A, B, C, D, E, F, G, or H) of "
    "the correct option.\n"
)


def patch_gemma4_video_features(model) -> None:
    """Work around Transformers 5.15 returning split video features as a tuple."""
    original = model.model.get_video_features

    def get_video_features(*args, **kwargs):
        output = original(*args, **kwargs)
        if isinstance(output.pooler_output, tuple):
            output.pooler_output = torch.cat(output.pooler_output, dim=0)
        return output

    model.model.get_video_features = get_video_features


def sample_video(path: Path, num_frames: int) -> tuple[np.ndarray, dict]:
    """Match the benchmark's fixed-frame sampling rule."""
    reader = decord.VideoReader(str(path), num_threads=2)
    if len(reader) == 0:
        raise ValueError(f"Empty video: {path}")
    count = min(num_frames, len(reader))
    step = len(reader) / (count + 1)
    indices = np.asarray([int(i * step) for i in range(1, count + 1)])
    metadata = {
        "total_num_frames": len(reader),
        "fps": float(reader.get_avg_fps()),
        "frames_indices": indices.tolist(),
    }
    return reader.get_batch(indices).asnumpy(), metadata


def build_video_index(video_root: Path) -> dict[str, Path]:
    index = {}
    for suffix in ("*.mp4", "*.mkv", "*.webm", "*.avi"):
        for path in video_root.rglob(suffix):
            index.setdefault(path.stem, path)
    return index


def final_text(processor, raw: str, prefix: torch.Tensor) -> str:
    try:
        parsed = processor.parse_response(raw, prefix=prefix)
    except Exception:
        return raw.strip()
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        for key in ("final", "response", "content", "text"):
            if key in parsed:
                return str(parsed[key]).strip()
    if isinstance(parsed, (list, tuple)) and parsed:
        return str(parsed[-1]).strip()
    return raw.strip()


def extract_prediction(text: str) -> str | None:
    """Match the benchmark's official standalone A-H extraction rule."""
    cleaned = re.sub(r"<[^>]+>", " ", text).strip()
    prefixes = (
        "Final Answer:",
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
    )
    for prefix in prefixes:
        cleaned = cleaned.replace(prefix, "")
    match = re.search(r"[A-H]", cleaned)
    return match.group(0) if match else None


def read_completed(path: Path, retry_noncompliant: bool) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            if (
                retry_noncompliant
                and not item.get("retry_pass")
                and not re.fullmatch(r"\s*[A-H]\s*", item.get("response", ""))
            ):
                continue
            completed.add(int(item["global_idx"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retry-noncompliant", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    frame = pd.read_parquet(args.parquet).reset_index(drop=True)
    records = []
    video_ranks: dict[str, int] = {}
    for global_idx, row in frame.iterrows():
        video_id = str(row["video_id"])
        if video_id not in video_ranks:
            video_ranks[video_id] = len(video_ranks) % args.world_size
        if video_ranks[video_id] != args.rank:
            continue
        record = row.to_dict()
        record["global_idx"] = int(global_idx)
        records.append(record)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.output, args.retry_noncompliant)
    records = [record for record in records if record["global_idx"] not in completed]
    video_index = build_video_index(args.video_root)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model)
    processor.video_processor.do_sample_frames = False
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    patch_gemma4_video_features(model)
    device = next(model.parameters()).device
    cached_video_id = None
    cached_frames = None
    cached_video_metadata = None

    with args.output.open("a", buffering=1) as stream:
        for record in tqdm(records, desc=f"Video-MME-v2 rank {args.rank}"):
            started = time.perf_counter()
            peak_memory = 0.0
            error = None
            raw = ""
            answer_text = ""
            prediction = None
            input_tokens = 0
            generated_tokens = 0
            try:
                video_id = str(record["video_id"])
                if video_id not in video_index:
                    raise FileNotFoundError(f"No video found for id {video_id}")
                if video_id != cached_video_id:
                    cached_frames, cached_video_metadata = sample_video(
                        video_index[video_id], args.num_frames
                    )
                    cached_video_id = video_id
                frames = cached_frames
                video_metadata = cached_video_metadata
                question = f"{record['question']}\n{record['options']}"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames},
                            {"type": "text", "text": PROMPT + "Question: " + question},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=args.thinking,
                    processor_kwargs={
                        "do_sample_frames": False,
                        "video_metadata": [video_metadata],
                    },
                )
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
                input_tokens = int(inputs["input_ids"].shape[-1])
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                generated = outputs[0, input_tokens:]
                generated_tokens = int(generated.numel())
                raw = processor.decode(generated, skip_special_tokens=False)
                answer_text = final_text(processor, raw, inputs["input_ids"])
                prediction = extract_prediction(answer_text)
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception:
                error = traceback.format_exc(limit=12)

            payload = {
                "global_idx": record["global_idx"],
                "video_id": str(record["video_id"]),
                "question_id": str(record["question_id"]),
                "group_type": str(record["group_type"]),
                "group_structure": str(record["group_structure"]),
                "level": str(record["level"]),
                "second_head": str(record["second_head"]),
                "third_head": str(record["third_head"]),
                "target": str(record["answer"]).strip().upper(),
                "prediction": prediction,
                "correct": prediction == str(record["answer"]).strip().upper(),
                "response": answer_text,
                "raw_response": raw,
                "error": error,
                "num_frames": args.num_frames,
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "max_new_tokens": args.max_new_tokens,
                "retry_pass": args.retry_noncompliant,
                "latency_seconds": time.perf_counter() - started,
                "peak_memory_gib": peak_memory,
            }
            stream.write(json.dumps(payload, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
