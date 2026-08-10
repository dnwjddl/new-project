#!/usr/bin/env python3
"""Smoke test for a query-free recurrent visual latent reasoning pipeline.

The script freezes CLIP-L/14, extracts one frame per EPIC-KITCHENS action,
and predicts the next verb from a short causal action context. It compares a
plain temporal state, a weight-tied recurrent reasoning head, and a matched-
compute unshared-depth head. This only validates plumbing and metrics; it is not
paper evidence and must not be used to accept or reject the hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPModel


@dataclass
class ActionRecord:
    video_id: str
    start_frame: int
    stop_frame: int
    verb_class: int
    narration: str


class SequenceDataset(Dataset):
    def __init__(self, examples: list[tuple[torch.Tensor, int]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.examples[index]


class ReasoningBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, state: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(state).unsqueeze(1)
        values = self.memory_norm(memory)
        update, _ = self.attention(query, values, values, need_weights=False)
        state = state + update.squeeze(1)
        return state + self.ffn(self.ffn_norm(state))


class TemporalReasoner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        classes: int,
        max_steps: int,
        shared: bool,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.shared = shared
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.temporal_state = nn.GRU(width, width, batch_first=True)
        if max_steps == 0:
            self.blocks = nn.ModuleList()
        elif shared:
            self.blocks = nn.ModuleList([ReasoningBlock(width, heads)])
        else:
            self.blocks = nn.ModuleList([ReasoningBlock(width, heads) for _ in range(max_steps)])
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, classes))

    def forward(self, features: torch.Tensor, steps: int | None = None) -> list[torch.Tensor]:
        memory = self.input_projection(features)
        _, hidden = self.temporal_state(memory)
        state = hidden[-1]
        logits = [self.classifier(state)]
        requested_steps = self.max_steps if steps is None else min(steps, self.max_steps)

        for step in range(requested_steps):
            block = self.blocks[0] if self.shared else self.blocks[step]
            state = block(state, memory)
            logits.append(self.classifier(state))
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/dataset/EK100/train"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/dataset/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv"),
    )
    parser.add_argument("--cache", type=Path, default=Path("ek100_clip_pilot.pt"))
    parser.add_argument("--results", type=Path, default=Path("latent_reasoning_pilot_results.json"))
    parser.add_argument("--max-videos", type=int, default=16)
    parser.add_argument("--max-actions-per-video", type=int, default=140)
    parser.add_argument("--top-verbs", type=int, default=12)
    parser.add_argument("--context-length", type=int, default=6)
    parser.add_argument("--feature-batch-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=941002)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_actions(path: Path) -> list[ActionRecord]:
    actions: list[ActionRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                actions.append(
                    ActionRecord(
                        video_id=row["video_id"],
                        start_frame=int(row["start_frame"]),
                        stop_frame=int(row["stop_frame"]),
                        verb_class=int(row["verb_class"]),
                        narration=row["narration"],
                    )
                )
            except (KeyError, ValueError):
                continue
    return actions


def choose_videos(
    actions: list[ActionRecord], data_root: Path, max_videos: int, seed: int
) -> dict[str, list[ActionRecord]]:
    grouped: dict[str, list[ActionRecord]] = defaultdict(list)
    for action in actions:
        if (data_root / f"{action.video_id}.MP4").exists():
            grouped[action.video_id].append(action)

    candidates = [video_id for video_id, rows in grouped.items() if len(rows) >= 35]
    random.Random(seed).shuffle(candidates)
    selected = candidates[:max_videos]
    return {video_id: sorted(grouped[video_id], key=lambda row: row.start_frame) for video_id in selected}


def iter_batches(items: list[np.ndarray], batch_size: int) -> Iterable[list[np.ndarray]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_features(args: argparse.Namespace, selected: dict[str, list[ActionRecord]]) -> dict:
    print(f"Loading frozen visual backbone on {args.device}...", flush=True)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    model = model.eval().to(args.device)
    if args.device.startswith("cuda"):
        model = model.half()

    payload: dict[str, list[dict]] = {}
    total = len(selected)
    started = time.time()

    for video_index, (video_id, rows) in enumerate(selected.items(), start=1):
        path = args.data_root / f"{video_id}.MP4"
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            print(f"Skipping unreadable video: {path}", flush=True)
            continue

        if len(rows) > args.max_actions_per_video:
            indices = np.linspace(0, len(rows) - 1, args.max_actions_per_video).round().astype(int)
            rows = [rows[index] for index in indices]

        valid_rows: list[ActionRecord] = []
        frames: list[np.ndarray] = []
        for row in rows:
            center_frame = max(0, (row.start_frame + row.stop_frame) // 2)
            capture.set(cv2.CAP_PROP_POS_FRAMES, center_frame)
            ok, frame = capture.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            valid_rows.append(row)
        capture.release()

        video_features: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in iter_batches(frames, args.feature_batch_size):
                pixel_values = processor(images=batch, return_tensors="pt")["pixel_values"].to(args.device)
                if args.device.startswith("cuda"):
                    pixel_values = pixel_values.half()
                encoded = model.get_image_features(pixel_values=pixel_values)
                if not isinstance(encoded, torch.Tensor):
                    encoded = encoded.pooler_output
                encoded = torch.nn.functional.normalize(encoded.float(), dim=-1)
                video_features.extend(encoded.cpu())

        payload[video_id] = [
            {"record": asdict(row), "feature": feature}
            for row, feature in zip(valid_rows, video_features, strict=True)
        ]
        elapsed = time.time() - started
        print(
            f"[{video_index:02d}/{total:02d}] {video_id}: {len(video_features)} frames "
            f"({elapsed / 60:.1f} min)",
            flush=True,
        )

    model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def load_or_extract(args: argparse.Namespace) -> dict:
    if args.cache.exists():
        print(f"Loading feature cache: {args.cache}", flush=True)
        return torch.load(args.cache, map_location="cpu", weights_only=False)

    actions = read_actions(args.annotations)
    selected = choose_videos(actions, args.data_root, args.max_videos, args.seed)
    print(f"Selected {len(selected)} videos from {len(actions)} annotated actions", flush=True)
    payload = extract_features(args, selected)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.cache)
    print(f"Saved feature cache: {args.cache}", flush=True)
    return payload


def build_examples(args: argparse.Namespace, payload: dict) -> tuple[list, list, dict]:
    verb_counts = Counter(
        item["record"]["verb_class"] for items in payload.values() for item in items
    )
    top_verbs = [verb for verb, _ in verb_counts.most_common(args.top_verbs)]
    class_map = {verb: index for index, verb in enumerate(top_verbs)}

    video_ids = sorted(payload)
    random.Random(args.seed + 1).shuffle(video_ids)
    validation_count = max(2, math.ceil(len(video_ids) * 0.25))
    validation_videos = set(video_ids[:validation_count])
    train_examples: list[tuple[torch.Tensor, int]] = []
    validation_examples: list[tuple[torch.Tensor, int]] = []

    for video_id, items in payload.items():
        destination = validation_examples if video_id in validation_videos else train_examples
        for index in range(args.context_length, len(items)):
            target_verb = items[index]["record"]["verb_class"]
            if target_verb not in class_map:
                continue
            context = torch.stack(
                [item["feature"].float() for item in items[index - args.context_length : index]]
            )
            destination.append((context, class_map[target_verb]))

    metadata = {
        "selected_videos": len(video_ids),
        "train_videos": len(video_ids) - validation_count,
        "validation_videos": validation_count,
        "top_verbs": top_verbs,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
    }
    return train_examples, validation_examples, metadata


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


@torch.inference_mode()
def evaluate(
    model: TemporalReasoner,
    loader: DataLoader,
    device: str,
    steps: int,
    context_crop: int = 0,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    correct = 0
    total = 0
    predictions: list[int] = []
    targets_all: list[int] = []
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        if context_crop:
            features = features[:, :-context_crop]
        logits = model(features, steps=steps)[-1]
        predicted = logits.argmax(dim=-1)
        correct += (predicted == targets).sum().item()
        total += targets.numel()
        predictions.extend(predicted.cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
    return correct / max(total, 1), predictions, targets_all


def train_model(
    name: str,
    model: TemporalReasoner,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
) -> dict:
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_state = None
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_items = 0
        for features, targets in train_loader:
            features = features.to(args.device)
            targets = targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits_by_step = model(features)
            if len(logits_by_step) == 1:
                loss = nn.functional.cross_entropy(logits_by_step[0], targets)
            else:
                weights = torch.linspace(0.35, 1.0, len(logits_by_step), device=args.device)
                losses = torch.stack(
                    [nn.functional.cross_entropy(logits, targets) for logits in logits_by_step]
                )
                loss = (weights * losses).sum() / weights.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * targets.numel()
            epoch_items += targets.numel()
        scheduler.step()

        validation_accuracy, _, _ = evaluate(
            model, validation_loader, args.device, steps=model.max_steps
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"{name:10s} epoch {epoch:02d}/{args.epochs}: "
                f"loss={epoch_loss / max(epoch_items, 1):.4f} val={validation_accuracy:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    step_curve = {}
    for steps in range(model.max_steps + 1):
        score, _, _ = evaluate(model, validation_loader, args.device, steps=steps)
        step_curve[str(steps)] = score

    full_accuracy, full_predictions, targets = evaluate(
        model, validation_loader, args.device, steps=model.max_steps
    )
    cropped_accuracy, cropped_predictions, _ = evaluate(
        model,
        validation_loader,
        args.device,
        steps=model.max_steps,
        context_crop=1,
    )
    corrected = sum(
        before != target and after == target
        for before, after, target in zip(cropped_predictions, full_predictions, targets, strict=True)
    )
    regressions = sum(
        before == target and after != target
        for before, after, target in zip(cropped_predictions, full_predictions, targets, strict=True)
    )
    initially_wrong = sum(before != target for before, target in zip(cropped_predictions, targets, strict=True))

    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_full_accuracy": full_accuracy,
        "step_curve": step_curve,
        "late_evidence": {
            "cropped_accuracy": cropped_accuracy,
            "full_accuracy": full_accuracy,
            "correction_rate_given_wrong_prior": corrected / max(initially_wrong, 1),
            "corrected_examples": corrected,
            "regressed_examples": regressions,
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    payload = load_or_extract(args)
    train_examples, validation_examples, metadata = build_examples(args, payload)
    if not train_examples or not validation_examples:
        raise RuntimeError(f"Insufficient examples after filtering: {metadata}")
    print(json.dumps(metadata, indent=2), flush=True)

    train_loader = DataLoader(
        SequenceDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        SequenceDataset(validation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    input_dim = train_examples[0][0].shape[-1]
    classes = args.top_verbs

    configurations = {
        "no_reason": dict(max_steps=0, shared=False),
        "shared_recurrent": dict(max_steps=args.max_steps, shared=True),
        "unshared_depth": dict(max_steps=args.max_steps, shared=False),
    }
    results = {
        "status": "completed",
        "seed": args.seed,
        "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
        "metadata": metadata,
        "models": {},
    }

    for model_index, (name, configuration) in enumerate(configurations.items()):
        set_seed(args.seed + model_index)
        model = TemporalReasoner(
            input_dim=input_dim,
            width=args.width,
            classes=classes,
            **configuration,
        )
        print(f"\nTraining {name} ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)
        results["models"][name] = train_model(
            name, model, train_loader, validation_loader, args
        )

    shared_curve = results["models"]["shared_recurrent"]["step_curve"]
    shared_scores = [shared_curve[str(step)] for step in range(args.max_steps + 1)]
    results["diagnostics"] = {
        "shared_k_gain": shared_scores[-1] - shared_scores[0],
        "shared_monotonic_steps": sum(
            later >= earlier for earlier, later in zip(shared_scores, shared_scores[1:])
        ),
        "shared_vs_unshared_kmax": (
            results["models"]["shared_recurrent"]["best_full_accuracy"]
            - results["models"]["unshared_depth"]["best_full_accuracy"]
        ),
        "shared_vs_no_reason": (
            results["models"]["shared_recurrent"]["best_full_accuracy"]
            - results["models"]["no_reason"]["best_full_accuracy"]
        ),
    }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results: {args.results}")
    print(json.dumps(results["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
