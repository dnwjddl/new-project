#!/usr/bin/env python3
"""Merge TempCompass shards and report exact-match metrics."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def normalize_target(task: str, target: str) -> str:
    if task != "caption_matching":
        return target
    match = re.search(r"Option\s*([12])", target, flags=re.I)
    if match:
        return f"Option {match.group(1)}"
    match = re.match(r"\s*(?:Sentence|Caption)\s*([AB])\s*:", target, flags=re.I)
    if match:
        return "Option 1" if match.group(1).upper() == "A" else "Option 2"
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    by_index = {}
    for path in args.inputs:
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                by_index[int(item["global_idx"])] = item
    records = [by_index[index] for index in sorted(by_index)]

    groups = defaultdict(list)
    for record in records:
        groups[(record["task"], record["dim"])].append(record)

    def metrics(items: list[dict]) -> dict:
        valid = [item for item in items if not item.get("error")]
        return {
            "samples": len(items),
            "valid": len(valid),
            "accuracy": sum(
                item.get("prediction")
                == normalize_target(item["task"], str(item["target"]))
                for item in valid
            )
            / max(1, len(valid)),
            "mean_latency_seconds": statistics.fmean(
                item["latency_seconds"] for item in valid
            )
            if valid
            else None,
            "mean_input_tokens": statistics.fmean(item["input_tokens"] for item in valid)
            if valid
            else None,
            "max_peak_memory_gib": max((item["peak_memory_gib"] for item in valid), default=None),
        }

    summary = {
        "overall": metrics(records),
        "by_task": {
            task: metrics([item for item in records if item["task"] == task])
            for task in sorted({item["task"] for item in records})
        },
        "by_dimension": {
            dimension: metrics(
                [item for item in records if item["dim"] == dimension]
            )
            for dimension in sorted({item["dim"] for item in records})
        },
        "by_task_and_dimension": {
            f"{task}/{dimension}": metrics(items)
            for (task, dimension), items in sorted(groups.items())
        },
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
