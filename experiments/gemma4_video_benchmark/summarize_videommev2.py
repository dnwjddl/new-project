#!/usr/bin/env python3
"""Merge Video-MME-v2 shards and compute its official grouped metrics."""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


RELEVANCE_SCORE = {0: 0.0, 1: 100.0 / 16, 2: 25.0, 3: 56.25, 4: 100.0}


def logic_rating(scores: list[bool], structure: str) -> float:
    layout = ast.literal_eval(structure)
    last_correct = -1
    for idx, correct in enumerate(scores):
        if not correct:
            break
        last_correct = idx

    if layout == [1, 2, 3, 4]:
        score_map = RELEVANCE_SCORE
    elif layout == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 / 3, 3: 700.0 / 12, 4: 100.0}
        if last_correct == 0 and scores[2]:
            last_correct += 1
    elif layout == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 10.0, 2: 20.0, 3: 50.0, 4: 100.0}
        if last_correct == -1 and scores[1]:
            last_correct += 1
    else:
        raise ValueError(f"Unknown logic structure: {layout}")
    return score_map[last_correct + 1]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


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
        groups[record["video_id"]].append(record)

    complete_groups = []
    for items in groups.values():
        items.sort(key=lambda item: int(item["question_id"].rsplit("-", 1)[-1]))
        if len(items) == 4:
            complete_groups.append(items)

    grouped_rows = []
    for items in complete_groups:
        scores = [bool(item["correct"]) for item in items]
        if items[-1]["group_type"] == "relevance":
            rating = RELEVANCE_SCORE[sum(scores)]
        else:
            rating = logic_rating(scores, items[-1]["group_structure"])
        grouped_rows.append(
            {
                "rating": rating,
                "group_type": items[-1]["group_type"],
                "level": f"level_{int(items[-1]['level'])}",
                "second_head": items[-1]["second_head"],
                "third_head": items[-1]["third_head"],
            }
        )

    def grouped_breakdown(key: str) -> dict[str, float | None]:
        values = defaultdict(list)
        for row in grouped_rows:
            values[row[key]].append(row["rating"])
        return {name: mean(scores) for name, scores in sorted(values.items())}

    def accuracy_breakdown(key: str) -> dict[str, float | None]:
        values = defaultdict(list)
        for record in records:
            name = record[key]
            if key == "level":
                name = f"level_{int(name)}"
            values[name].append(100.0 if record["correct"] else 0.0)
        return {name: mean(scores) for name, scores in sorted(values.items())}

    summary = {
        "questions": len(records),
        "expected_questions": 3200,
        "errors": sum(bool(item.get("error")) for item in records),
        "accuracy": 100.0 * sum(bool(item["correct"]) for item in records) / max(1, len(records)),
        "strict_letter_rate": 100.0
        * sum(
            bool(re.fullmatch(r"\s*[A-H]\s*", item.get("response", "")))
            for item in records
        )
        / max(1, len(records)),
        "generation_limit_hits": sum(
            item.get("generated_tokens", 0) >= item.get("max_new_tokens", 10**9)
            for item in records
        ),
        "complete_groups": len(complete_groups),
        "expected_groups": 800,
        "nonlinear_group_score": mean([row["rating"] for row in grouped_rows]),
        "group_type_score": grouped_breakdown("group_type"),
        "level_score": grouped_breakdown("level"),
        "second_head_score": grouped_breakdown("second_head"),
        "third_head_score": grouped_breakdown("third_head"),
        "group_type_accuracy": accuracy_breakdown("group_type"),
        "level_accuracy": accuracy_breakdown("level"),
        "second_head_accuracy": accuracy_breakdown("second_head"),
        "third_head_accuracy": accuracy_breakdown("third_head"),
        "mean_latency_seconds": mean(
            [item["latency_seconds"] for item in records if not item.get("error")]
        ),
        "mean_input_tokens": mean(
            [item["input_tokens"] for item in records if not item.get("error")]
        ),
        "max_peak_memory_gib": max(
            (item["peak_memory_gib"] for item in records if not item.get("error")),
            default=None,
        ),
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
