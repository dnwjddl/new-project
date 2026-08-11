#!/usr/bin/env python3
"""Aggregate per-budget temporal-pair metrics and compute K90."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = [json.loads(path.read_text()) for path in args.inputs]
    runs = sorted((run for run in runs if run.get("budget") is not None), key=lambda run: run["budget"])
    if not runs:
        raise SystemExit("No budgeted result files were supplied")

    dense = max(runs, key=lambda run: run["budget"])
    target = 0.9 * dense["paired_accuracy"]
    k90 = next((run["budget"] for run in runs if run["paired_accuracy"] >= target), None)
    result = {
        "largest_budget_reference": dense["budget"],
        "largest_budget_paired_accuracy": dense["paired_accuracy"],
        "k90": k90,
        "curve": [
            {
                "budget": run["budget"],
                "sample_accuracy": run["sample_accuracy"],
                "paired_accuracy": run["paired_accuracy"],
                "answer_flip_consistency": run["answer_flip_consistency"],
                "counterfactual_pairs": run["counterfactual_pairs"],
            }
            for run in runs
        ],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
