#!/usr/bin/env python3
"""Aggregate three-seed predictive latent reasoning core ablations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(len(array))) if len(array) > 1 else 0.0,
        "n": len(values),
    }


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(args.input_dir.glob("*_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed" and payload.get("variant"):
            grouped[payload["variant"]].append(payload)
    if not grouped:
        raise RuntimeError(f"No completed results in {args.input_dir}")

    summary = {"variants": {}, "paired_deltas": {}}
    metric_paths = {
        "action_recall5": ("action", "macro_recall5"),
        "action_top5": ("action", "top5"),
        "verb_recall5": ("verb", "macro_recall5"),
        "noun_recall5": ("noun", "macro_recall5"),
        "future_cosine": ("future_cosine",),
        "future_nll": ("future_nll",),
    }
    by_seed = {}
    for variant, rows in sorted(grouped.items()):
        by_seed[variant] = {int(row["seed"]): row for row in rows}
        variant_summary = {}
        for name, path in metric_paths.items():
            values = []
            for row in rows:
                value = row["metrics"]
                for key in path:
                    value = value[key]
                values.append(float(value))
            variant_summary[name] = mean_ci(values)
        variant_summary["runtime_seconds"] = mean_ci([float(row["runtime_seconds"]) for row in rows])
        summary["variants"][variant] = variant_summary

    reference = "predict"
    if reference in by_seed:
        for variant in (
            "raw_correct",
            "calibrated_correct",
            "observation_gate",
            "innovation_gate",
        ):
            common = sorted(set(by_seed.get(variant, {})).intersection(by_seed[reference]))
            if not common:
                continue
            deltas = []
            for seed in common:
                candidate = by_seed[variant][seed]["metrics"]["action"]["macro_recall5"]
                baseline = by_seed[reference][seed]["metrics"]["action"]["macro_recall5"]
                deltas.append(float(candidate - baseline))
            summary["paired_deltas"][f"{variant}_minus_{reference}"] = mean_ci(deltas)

    if "observation_gate" in by_seed and "innovation_gate" in by_seed:
        common = sorted(set(by_seed["observation_gate"]).intersection(by_seed["innovation_gate"]))
        deltas = []
        for seed in common:
            innovation = by_seed["innovation_gate"][seed]["metrics"]["action"]["macro_recall5"]
            observation = by_seed["observation_gate"][seed]["metrics"]["action"]["macro_recall5"]
            deltas.append(float(innovation - observation))
        summary["paired_deltas"]["innovation_gate_minus_observation_gate"] = mean_ci(deltas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
