#!/usr/bin/env python3
"""Evaluate Gemma 4 E2B/E4B on deterministic TempCompass tasks."""

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


TASKS = ("multi-choice", "yes_no", "caption_matching")


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
    reader = decord.VideoReader(str(path), num_threads=2)
    if len(reader) == 0:
        raise ValueError(f"Empty video: {path}")
    count = min(num_frames, len(reader))
    indices = np.linspace(0, len(reader) - 1, count).round().astype(np.int64)
    metadata = {
        "total_num_frames": len(reader),
        "fps": float(reader.get_avg_fps()),
        "frames_indices": indices.tolist(),
    }
    return reader.get_batch(indices).asnumpy(), metadata


def build_prompt(task: str, question: str) -> str:
    suffix = {
        "multi-choice": "\nAnswer only with the option letter, such as A.",
        "yes_no": "\nAnswer only yes or no.",
        "caption_matching": "\nAnswer only Option 1 or Option 2.",
    }[task]
    return question + suffix


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


def extract_prediction(task: str, text: str) -> str | None:
    cleaned = re.sub(r"<[^>]+>", " ", text).strip()
    if task == "yes_no":
        match = re.search(r"\b(yes|no)\b", cleaned, flags=re.IGNORECASE)
        return match.group(1).lower() if match else None
    if task == "caption_matching":
        match = re.search(r"\b(?:option\s*)?([12])\b", cleaned, flags=re.IGNORECASE)
        return f"Option {match.group(1)}" if match else None
    patterns = (
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*([A-D])\b",
        r"^\s*([A-D])(?:\b|[.)])",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def normalize_target(task: str, answer: str) -> str:
    if task == "yes_no":
        return answer.strip().lower()
    if task == "caption_matching":
        match = re.search(r"Option\s*([12])", answer, flags=re.IGNORECASE)
        if match:
            return f"Option {match.group(1)}"
        match = re.match(r"\s*(?:Sentence|Caption)\s*([AB])\s*:", answer, re.I)
        if match:
            return "Option 1" if match.group(1).upper() == "A" else "Option 2"
        return answer.strip()
    match = re.match(r"\s*([A-D])", answer, flags=re.IGNORECASE)
    return match.group(1).upper() if match else answer.strip()


def load_records(data_root: Path) -> list[dict]:
    records: list[dict] = []
    for task in TASKS:
        frame = pd.read_parquet(data_root / task / "test-00000-of-00001.parquet")
        for row_idx, row in frame.iterrows():
            item = row.to_dict()
            item.update(task=task, row_idx=int(row_idx), global_idx=len(records))
            records.append(item)
    return records


def read_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text().splitlines():
        if line.strip():
            completed.add(int(json.loads(line)["global_idx"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    records = [
        record
        for record in load_records(args.data_root)
        if record["global_idx"] % args.world_size == args.rank
    ]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.output)
    records = [record for record in records if record["global_idx"] not in completed]

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

    with args.output.open("a", buffering=1) as stream:
        for record in tqdm(records, desc=f"TempCompass rank {args.rank}"):
            started = time.perf_counter()
            peak_memory = 0.0
            error = None
            raw = ""
            answer_text = ""
            prediction = None
            input_tokens = 0
            try:
                frames, video_metadata = sample_video(
                    args.video_root / f"{record['video_id']}.mp4", args.num_frames
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames},
                            {
                                "type": "text",
                                "text": build_prompt(record["task"], str(record["question"])),
                            },
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
                raw = processor.decode(generated, skip_special_tokens=False)
                answer_text = final_text(processor, raw, inputs["input_ids"])
                prediction = extract_prediction(record["task"], answer_text)
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception:
                error = traceback.format_exc(limit=12)

            target = normalize_target(record["task"], str(record["answer"]))
            payload = {
                "global_idx": record["global_idx"],
                "task": record["task"],
                "row_idx": record["row_idx"],
                "video_id": str(record["video_id"]),
                "dim": str(record["dim"]),
                "question": str(record["question"]),
                "target": target,
                "prediction": prediction,
                "correct": prediction == target,
                "response": answer_text,
                "raw_response": raw,
                "error": error,
                "num_frames": args.num_frames,
                "input_tokens": input_tokens,
                "latency_seconds": time.perf_counter() - started,
                "peak_memory_gib": peak_memory,
            }
            stream.write(json.dumps(payload, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
