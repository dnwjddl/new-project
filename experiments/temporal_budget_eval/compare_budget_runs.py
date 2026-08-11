#!/usr/bin/env python3
"""Run paired exact tests between fixed-budget TempCompass sample logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from score_tempcompass_pairs import find_log, load_records, normalize_record


def exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def load_evidence(path: Path) -> dict[str, dict[tuple[str, str, str], bool]]:
    normalized = [item for record in load_records(find_log(path)) if (item := normalize_record(record))]
    samples = {
        (item["video_id"], item["aspect"], item["question_family"]): item["correct"]
        for item in normalized
    }

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in normalized:
        groups[(item["base_video_id"], item["aspect"], item["question_family"])].append(item)
    changed = {
        key: items
        for key, items in groups.items()
        if len({item["video_id"] for item in items}) >= 2
        and len({item["target"] for item in items}) >= 2
    }
    return {
        "samples": samples,
        "strict_families": {key: all(item["correct"] for item in items) for key, items in changed.items()},
        "flip_families": {
            key: len({item["prediction"] for item in items}) >= 2
            for key, items in changed.items()
        },
    }


def compare(candidate: dict[Any, bool], reference: dict[Any, bool]) -> dict[str, int | float]:
    keys = sorted(candidate.keys() & reference.keys())
    gains = sum(not candidate[key] and reference[key] for key in keys)
    losses = sum(candidate[key] and not reference[key] for key in keys)
    candidate_accuracy = sum(candidate[key] for key in keys) / max(1, len(keys))
    reference_accuracy = sum(reference[key] for key in keys) / max(1, len(keys))
    return {
        "common_samples": len(keys),
        "candidate_accuracy": candidate_accuracy,
        "reference_accuracy": reference_accuracy,
        "reference_minus_candidate": reference_accuracy - candidate_accuracy,
        "candidate_wrong_reference_right": gains,
        "candidate_right_reference_wrong": losses,
        "mcnemar_exact_p": exact_mcnemar_p(gains, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_evidence(args.reference)
    result = {}
    for path in args.candidate:
        candidate = load_evidence(path)
        result[str(path)] = {
            metric: compare(candidate[metric], reference[metric])
            for metric in reference
        }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
