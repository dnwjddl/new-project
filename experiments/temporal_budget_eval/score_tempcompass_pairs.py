#!/usr/bin/env python3
"""Score temporal counterfactual pairs from lmms-eval sample logs."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


VARIANT_SUFFIX = re.compile(
    r"(?:_reverse|_reversed|_slow(?:ed)?|_fast(?:er)?|_speedup|_slowdown|_concat_[0-9]+|_[0-9]+|_\d+(?:\.\d+)?x)$",
    re.IGNORECASE,
)
OPTION = re.compile(r"^\s*([A-F])[.)]\s*(.+?)\s*$", re.IGNORECASE)


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        return records

    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("logs", "samples", "results", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if all(isinstance(value, dict) for value in payload.values()):
        return list(payload.values())
    return []


def nested(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    for container_name in ("doc", "avg_accuracy"):
        container = record.get(container_name)
        if isinstance(container, dict):
            for key in keys:
                if key in container and container[key] not in (None, ""):
                    return container[key]
    return None


def first_text(value: Any) -> str:
    while isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        for key in ("text", "content", "response", "prediction"):
            if key in value:
                return first_text(value[key])
    return "" if value is None else str(value).strip()


def parse_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in question.splitlines():
        match = OPTION.match(line)
        if match:
            options[match.group(1).upper()] = match.group(2).strip().lower()
    return options


def answer_letter(value: Any) -> str | None:
    text = first_text(value)
    patterns = (
        r"^\s*([A-F])(?:\s*$|[.)\s:])",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-F])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    candidates = set(re.findall(r"(?<![A-Za-z])([A-F])(?:[.)]|\b)", text, re.IGNORECASE))
    return candidates.pop().upper() if len(candidates) == 1 else None


def bootstrap_mean_ci(values: list[bool], *, seed: int, draws: int = 10_000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return [means[int(0.025 * draws)], means[int(0.975 * draws)]]


def semantic_answer(value: Any, options: dict[str, str]) -> str:
    text = first_text(value)
    letter = answer_letter(text)
    if letter and letter in options:
        return options[letter]
    return re.sub(r"\s+", " ", text.lower()).strip()


def video_id(record: dict[str, Any]) -> str:
    value = nested(record, "video_id", "video", "video_name")
    if value in (None, ""):
        value = nested(record, "doc_id", "id")
    return Path(first_text(value)).stem


def base_video_id(value: str) -> str:
    previous = None
    while value != previous:
        previous = value
        value = VARIANT_SUFFIX.sub("", value)
    return value


def question_family(question: str) -> str:
    stem = question.splitlines()[0].lower()
    stem = re.sub(r"[^a-z0-9 ]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()


def normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    question = first_text(nested(record, "question", "prompt", "query"))
    target = nested(record, "answer", "target", "gold", "label")
    prediction = nested(record, "filtered_resps", "resps", "prediction", "response", "output")
    vid = video_id(record)
    if not question or target is None or prediction is None or not vid:
        return None

    options = parse_options(question)
    target_letter = answer_letter(target)
    prediction_letter = answer_letter(prediction)
    return {
        "video_id": vid,
        "base_video_id": base_video_id(vid),
        "aspect": first_text(nested(record, "aspect", "dimension", "dim", "category", "task_type")) or "unknown",
        "question_family": question_family(question),
        "target": semantic_answer(target, options),
        "prediction": semantic_answer(prediction, options),
        "correct": bool(target_letter and prediction_letter and target_letter == prediction_letter),
    }


def score(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized = [item for record in records if (item := normalize_record(record))]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in normalized:
        groups[(item["base_video_id"], item["aspect"], item["question_family"])].append(item)

    pairs = [items for items in groups.values() if len({item["video_id"] for item in items}) >= 2]
    changed_pairs = [items for items in pairs if len({item["target"] for item in items}) >= 2]
    paired_correct = [all(item["correct"] for item in items) for items in changed_pairs]
    flip_consistent = [len({item["prediction"] for item in items}) >= 2 for items in changed_pairs]
    sample_correct = [item["correct"] for item in normalized]

    by_aspect: dict[str, dict[str, float | int]] = {}
    aspects = sorted({item["aspect"] for item in normalized})
    for aspect in aspects:
        samples = [item for item in normalized if item["aspect"] == aspect]
        aspect_pairs = [items for items in changed_pairs if items[0]["aspect"] == aspect]
        by_aspect[aspect] = {
            "samples": len(samples),
            "sample_accuracy": sum(item["correct"] for item in samples) / max(1, len(samples)),
            "counterfactual_pairs": len(aspect_pairs),
            "paired_accuracy": sum(all(item["correct"] for item in items) for items in aspect_pairs)
            / max(1, len(aspect_pairs)),
        }

    return {
        "samples": len(normalized),
        "sample_accuracy": sum(sample_correct) / max(1, len(sample_correct)),
        "counterfactual_pairs": len(changed_pairs),
        "paired_accuracy": sum(paired_correct) / max(1, len(paired_correct)),
        "answer_flip_consistency": sum(flip_consistent) / max(1, len(flip_consistent)),
        "confidence_intervals_95": {
            "sample_accuracy": bootstrap_mean_ci(sample_correct, seed=0),
            "paired_accuracy": bootstrap_mean_ci(paired_correct, seed=1),
            "answer_flip_consistency": bootstrap_mean_ci(flip_consistent, seed=2),
        },
        "by_aspect": by_aspect,
    }


def find_log(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(
        [*path.rglob("*.jsonl"), *path.rglob("*.json")],
        key=lambda item: ("sample" in item.name.lower(), item.stat().st_mtime),
        reverse=True,
    )
    for candidate in candidates:
        if "sample" in candidate.name.lower() or "tempcompass" in candidate.name.lower():
            return candidate
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No JSON log found under {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="lmms-eval JSON file or output directory")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    log = find_log(args.input)
    result = score(load_records(log))
    result["source"] = str(log)
    result["budget"] = args.budget
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
