#!/usr/bin/env python3
"""Aggregate exact-budget temporal metrics across compression methods."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def infer_method(path: Path) -> str:
    return re.sub(r"-k\d+$", "", path.stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        run = json.loads(path.read_text())
        if run.get("budget") is None:
            continue
        rows.append(
            {
                "method": run.get("method") or infer_method(path),
                "budget": run["budget"],
                "samples": run["samples"],
                "counterfactual_families": run["counterfactual_pairs"],
                "sample_accuracy": run["sample_accuracy"],
                "strict_family_accuracy": run["paired_accuracy"],
                "answer_flip_consistency": run["answer_flip_consistency"],
                "confidence_intervals_95": run.get("confidence_intervals_95"),
            }
        )

    rows.sort(key=lambda row: (row["method"], row["budget"]))
    result = {
        "methods": sorted({row["method"] for row in rows}),
        "budgets": sorted({row["budget"] for row in rows}),
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
