#!/usr/bin/env python3
"""Train matched-compute ablations for predictive latent belief correction."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


VARIANTS = (
    "depth_only",
    "predict",
    "raw_correct",
    "calibrated_correct",
    "observation_gate",
    "innovation_gate",
)


@dataclass
class ModelConfig:
    slots: int
    token_dim: int
    width: int
    memory_tokens: int
    heads: int
    num_verbs: int
    num_nouns: int
    num_actions: int
    correction_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=941002)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--memory-tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--correction-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--prediction-weight", type=float, default=0.5)
    parser.add_argument("--gate-weight", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthetic_cache(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    slots, token_dim = 4, 6
    num_verbs, num_nouns = 5, 7
    action_pairs = [(verb, noun) for verb in range(num_verbs) for noun in range(num_nouns)]
    action_map = {pair: index for index, pair in enumerate(action_pairs)}

    def make_split(videos: int, length: int) -> dict:
        features, verbs, nouns, actions, sequences, video_ids = [], [], [], [], [], []
        for video_index in range(videos):
            start = len(features)
            state = rng.normal(size=(slots, token_dim)).astype(np.float32)
            verb, noun = rng.integers(num_verbs), rng.integers(num_nouns)
            for step in range(length):
                if rng.random() < 0.15:
                    verb = int(rng.integers(num_verbs))
                    noun = int(rng.integers(num_nouns))
                    state += rng.normal(scale=1.2, size=state.shape)
                else:
                    verb = int((verb + (state.mean() > 0)) % num_verbs)
                    noun = int((noun + (step % 3 == 0)) % num_nouns)
                    state = 0.82 * state + 0.08 * verb + 0.04 * noun
                observation = state + rng.normal(scale=0.08, size=state.shape)
                features.append(observation.astype(np.float16))
                verbs.append(verb)
                nouns.append(noun)
                actions.append(action_map[(verb, noun)])
                video_ids.append(f"synthetic_{video_index:03d}")
            sequences.append(torch.arange(start, len(features), dtype=torch.long))
        return {
            "features": torch.tensor(np.asarray(features), dtype=torch.float16),
            "verbs": torch.tensor(verbs, dtype=torch.long),
            "nouns": torch.tensor(nouns, dtype=torch.long),
            "actions": torch.tensor(actions, dtype=torch.long),
            "sequences": sequences,
            "video_ids": video_ids,
        }

    return {
        "pooled_slots": slots,
        "token_dim": token_dim,
        "num_verbs": num_verbs,
        "num_nouns": num_nouns,
        "num_actions": len(action_pairs),
        "splits": {"train": make_split(14, 38), "validation": make_split(5, 34)},
        "metadata": {"synthetic": True},
    }


class WindowDataset(Dataset):
    def __init__(self, split: dict, window: int, stride: int) -> None:
        self.split = split
        self.window = window
        self.examples: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(split["sequences"]):
            maximum = len(sequence) - (window + 1)
            if maximum < 0:
                continue
            starts = list(range(0, maximum + 1, stride))
            if starts[-1] != maximum:
                starts.append(maximum)
            self.examples.extend((sequence_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        sequence_index, start = self.examples[item]
        indices = self.split["sequences"][sequence_index][start : start + self.window + 1]
        return (
            self.split["features"][indices],
            self.split["verbs"][indices[1:]],
            self.split["nouns"][indices[1:]],
            self.split["actions"][indices[1:]],
        )


class SharedRecurrentCore(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.mode = nn.Embedding(2, width)
        self.state_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor | None, mode: int) -> torch.Tensor:
        normalized = self.state_norm(state)
        update, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        state = state + update
        mode_token = self.mode.weight[mode].view(1, 1, -1).expand(state.shape[0], -1, -1)
        full_context = mode_token if context is None else torch.cat((mode_token, context), dim=1)
        update, _ = self.cross_attention(
            self.state_norm(state),
            self.context_norm(full_context),
            self.context_norm(full_context),
            need_weights=False,
        )
        state = state + update
        return state + self.ffn(self.ffn_norm(state))


class PredictiveLatentReasoner(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.initial_memory = nn.Parameter(torch.randn(config.memory_tokens, config.width) * 0.02)
        self.observation_projection = nn.Linear(config.token_dim, config.width)
        self.innovation_projection = nn.Linear(config.token_dim, config.width)
        self.core = SharedRecurrentCore(config.width, config.heads)
        self.future_queries = nn.Parameter(torch.randn(config.slots, config.width) * 0.02)
        self.future_attention = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.future_norm = nn.LayerNorm(config.width)
        self.future_mean = nn.Linear(config.width, config.token_dim)
        self.future_log_scale = nn.Linear(config.width, config.token_dim)
        gate_width = max(32, config.width // 2)
        self.slot_gate = nn.Sequential(
            nn.Linear(config.token_dim * 4, gate_width),
            nn.GELU(),
            nn.Linear(gate_width, 1),
        )
        self.memory_gate = nn.Sequential(
            nn.Linear(config.width * 2 + 1, gate_width),
            nn.GELU(),
            nn.Linear(gate_width, 1),
        )
        self.readout_norm = nn.LayerNorm(config.width)
        self.verb_head = nn.Linear(config.width, config.num_verbs)
        self.noun_head = nn.Linear(config.width, config.num_nouns)
        self.action_head = nn.Linear(config.width, config.num_actions)

    def initial_state(self, batch: int) -> torch.Tensor:
        return self.initial_memory.unsqueeze(0).expand(batch, -1, -1)

    def reason(self, state: torch.Tensor) -> torch.Tensor:
        return self.core(state, None, mode=0)

    def future(self, predicted_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.future_queries.unsqueeze(0).expand(predicted_state.shape[0], -1, -1)
        decoded, _ = self.future_attention(queries, predicted_state, predicted_state, need_weights=False)
        decoded = self.future_norm(decoded)
        mean = self.future_mean(decoded)
        log_scale = self.future_log_scale(decoded).clamp(-4.0, 3.0)
        return mean, log_scale

    def task_logits(self, predicted_state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        pooled = self.readout_norm(predicted_state.mean(dim=1))
        return self.verb_head(pooled), self.noun_head(pooled), self.action_head(pooled)

    def correct(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        innovation: torch.Tensor,
        steps: int,
        slot_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gated_innovation = innovation if slot_gate is None else innovation * slot_gate
        context = torch.cat(
            (
                self.observation_projection(observation),
                self.innovation_projection(gated_innovation),
            ),
            dim=1,
        )
        for _ in range(steps):
            candidate = self.core(state, context, mode=1)
            if slot_gate is None:
                state = candidate
            else:
                gate_summary = slot_gate.mean(dim=1, keepdim=True).expand(-1, state.shape[1], -1)
                memory_gate = torch.sigmoid(
                    self.memory_gate(torch.cat((state, candidate, gate_summary), dim=-1))
                )
                state = state + gate_summary * memory_gate * (candidate - state)
        return state

    def update_gate(
        self,
        observation: torch.Tensor,
        mean: torch.Tensor,
        log_scale: torch.Tensor,
        use_innovation: bool,
    ) -> torch.Tensor:
        calibrated = (observation - mean) * torch.exp(-log_scale)
        if use_innovation:
            gate_input = torch.cat((observation, mean, calibrated.abs(), log_scale), dim=-1)
        else:
            zeros = torch.zeros_like(observation)
            gate_input = torch.cat((observation, zeros, zeros, zeros), dim=-1)
        return torch.sigmoid(self.slot_gate(gate_input))

    def bootstrap(self, observation: torch.Tensor) -> torch.Tensor:
        state = self.initial_state(observation.shape[0])
        zeros = torch.zeros_like(observation)
        return self.correct(state, observation, zeros, steps=1)


def future_nll(target: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    residual = (target - mean) * torch.exp(-log_scale)
    return (0.5 * residual.square() + log_scale).mean()


def task_loss(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    verb_logits, noun_logits, action_logits = logits
    verbs, nouns, actions = labels
    loss = 0.5 * F.cross_entropy(verb_logits, verbs) + 0.5 * F.cross_entropy(noun_logits, nouns)
    valid = actions >= 0
    if valid.any():
        loss = loss + F.cross_entropy(action_logits[valid], actions[valid])
    return loss


def choose_innovation(
    variant: str, target: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    if variant in ("depth_only", "predict", "observation_gate"):
        return torch.zeros_like(target)
    residual = target - mean
    if variant == "raw_correct":
        return residual
    if variant in ("calibrated_correct", "innovation_gate"):
        return residual * torch.exp(-log_scale)
    raise ValueError(variant)


def train_batch(
    model: PredictiveLatentReasoner,
    batch: tuple[torch.Tensor, ...],
    variant: str,
    prediction_weight: float,
    gate_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    features, verbs, nouns, actions = (tensor.to(device, non_blocking=True) for tensor in batch)
    features = features.float()
    state = model.bootstrap(features[:, 0])
    task_total = torch.zeros((), device=device)
    prediction_total = torch.zeros((), device=device)
    innovation_total = torch.zeros((), device=device)
    gate_total = torch.zeros((), device=device)
    steps = features.shape[1] - 1

    for index in range(steps):
        predicted_state = model.reason(state)
        mean, log_scale = model.future(predicted_state)
        target = features[:, index + 1]
        logits = model.task_logits(predicted_state)
        task_total = task_total + task_loss(
            logits, (verbs[:, index], nouns[:, index], actions[:, index])
        )
        if variant != "depth_only":
            prediction_total = prediction_total + future_nll(target, mean, log_scale)
        innovation = choose_innovation(variant, target, mean, log_scale)
        slot_gate = None
        if variant == "observation_gate":
            slot_gate = model.update_gate(target, mean, log_scale, use_innovation=False)
        elif variant == "innovation_gate":
            slot_gate = model.update_gate(target, mean, log_scale, use_innovation=True)
        innovation_total = innovation_total + innovation.square().mean().sqrt()
        if slot_gate is not None:
            gate_total = gate_total + slot_gate.mean()
        state = model.correct(
            state,
            target,
            innovation,
            steps=model.config.correction_steps,
            slot_gate=slot_gate,
        )

    task_total = task_total / steps
    prediction_total = prediction_total / steps
    innovation_total = innovation_total / steps
    gate_total = gate_total / steps
    total = task_total + prediction_weight * prediction_total
    if variant in ("observation_gate", "innovation_gate"):
        total = total + gate_weight * gate_total
    return total, {
        "loss": float(total.detach()),
        "task_loss": float(task_total.detach()),
        "prediction_loss": float(prediction_total.detach()),
        "innovation_rms": float(innovation_total.detach()),
        "mean_update_gate": float(gate_total.detach()),
    }


def topk_stats(logits: torch.Tensor, labels: torch.Tensor, max_k: int = 5) -> dict:
    valid = labels >= 0
    logits, labels = logits[valid], labels[valid]
    if not len(labels):
        return {"count": 0, "top1": 0.0, "top5": 0.0, "macro_recall5": 0.0}
    k = min(max_k, logits.shape[1])
    top = logits.topk(k, dim=1).indices
    correct = top.eq(labels[:, None])
    per_class_hits: dict[int, list[float]] = defaultdict(list)
    for label, hit in zip(labels.tolist(), correct.any(dim=1).float().tolist()):
        per_class_hits[int(label)].append(hit)
    return {
        "count": len(labels),
        "top1": float(top[:, 0].eq(labels).float().mean()),
        "top5": float(correct.any(dim=1).float().mean()),
        "macro_recall5": float(np.mean([np.mean(values) for values in per_class_hits.values()])),
    }


@torch.inference_mode()
def evaluate(
    model: PredictiveLatentReasoner,
    split: dict,
    variant: str,
    device: torch.device,
) -> dict:
    model.eval()
    collected = {"verb_logits": [], "noun_logits": [], "action_logits": [], "verb": [], "noun": [], "action": []}
    cosine_values, nll_values, innovations, update_norms, gate_values = [], [], [], [], []

    for indices in split["sequences"]:
        if len(indices) < 2:
            continue
        features = split["features"][indices].to(device).float()
        state = model.bootstrap(features[0:1])
        for index in range(len(features) - 1):
            predicted_state = model.reason(state)
            mean, log_scale = model.future(predicted_state)
            target = features[index + 1 : index + 2]
            verb_logits, noun_logits, action_logits = model.task_logits(predicted_state)
            target_index = indices[index + 1]
            collected["verb_logits"].append(verb_logits.cpu())
            collected["noun_logits"].append(noun_logits.cpu())
            collected["action_logits"].append(action_logits.cpu())
            collected["verb"].append(split["verbs"][target_index].view(1))
            collected["noun"].append(split["nouns"][target_index].view(1))
            collected["action"].append(split["actions"][target_index].view(1))
            cosine_values.append(float(F.cosine_similarity(mean.flatten(1), target.flatten(1)).mean()))
            nll_values.append(float(future_nll(target, mean, log_scale)))
            innovation = choose_innovation(variant, target, mean, log_scale)
            slot_gate = None
            if variant == "observation_gate":
                slot_gate = model.update_gate(target, mean, log_scale, use_innovation=False)
            elif variant == "innovation_gate":
                slot_gate = model.update_gate(target, mean, log_scale, use_innovation=True)
            innovations.append(float(innovation.square().mean().sqrt()))
            if slot_gate is not None:
                gate_values.append(float(slot_gate.mean()))
            previous = state
            state = model.correct(
                state,
                target,
                innovation,
                steps=model.config.correction_steps,
                slot_gate=slot_gate,
            )
            update_norms.append(float((state - previous).square().mean().sqrt()))

    tensors = {key: torch.cat(values) for key, values in collected.items()}
    return {
        "verb": topk_stats(tensors["verb_logits"], tensors["verb"]),
        "noun": topk_stats(tensors["noun_logits"], tensors["noun"]),
        "action": topk_stats(tensors["action_logits"], tensors["action"]),
        "future_cosine": float(np.mean(cosine_values)),
        "future_nll": float(np.mean(nll_values)),
        "innovation_rms": float(np.mean(innovations)),
        "state_update_rms": float(np.mean(update_norms)),
        "mean_update_gate": float(np.mean(gate_values)) if gate_values else None,
        "evaluated_transitions": len(cosine_values),
    }


def average_logs(logs: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in logs])) for key in logs[0]}


def main() -> None:
    args = parse_args()
    if not args.synthetic and args.cache is None:
        raise ValueError("--cache is required unless --synthetic is used")
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing a silent CPU fallback")
    device = torch.device(args.device)
    cache = synthetic_cache(args.seed) if args.synthetic else torch.load(args.cache, map_location="cpu", weights_only=False)
    train_split = cache["splits"]["train"]
    validation_split = cache["splits"]["validation"]
    dataset = WindowDataset(train_split, args.window, args.stride)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )
    config = ModelConfig(
        slots=int(cache["pooled_slots"]),
        token_dim=int(cache["token_dim"]),
        width=args.width,
        memory_tokens=args.memory_tokens,
        heads=args.heads,
        num_verbs=int(cache["num_verbs"]),
        num_nouns=int(cache["num_nouns"]),
        num_actions=int(cache["num_actions"]),
        correction_steps=args.correction_steps,
    )
    model = PredictiveLatentReasoner(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best_score = -math.inf
    best_state = None
    best_epoch = 0
    epochs_without_gain = 0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_logs = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, logs = train_batch(
                    model,
                    batch,
                    args.variant,
                    args.prediction_weight,
                    args.gate_weight,
                    device,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_logs.append(logs)

        metrics = evaluate(model, validation_split, args.variant, device)
        score = metrics["action"]["macro_recall5"]
        row = {"epoch": epoch, "train": average_logs(epoch_logs), "validation": metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
        if epochs_without_gain >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, validation_split, args.variant, device)
    result = {
        "status": "completed",
        "variant": args.variant,
        "seed": args.seed,
        "config": asdict(config),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cache_metadata": cache.get("metadata", {}),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_windows": len(dataset),
        "best_epoch": best_epoch,
        "metrics": final_metrics,
        "history": history,
        "runtime_seconds": round(time.time() - started, 3),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("variant", "seed", "best_epoch", "metrics", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
