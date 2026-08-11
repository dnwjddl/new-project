#!/usr/bin/env python3
"""Incrementally extract Video-MME-v2 archives while they download."""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path


def already_extracted(archive: Path, output: Path) -> bool:
    try:
        with zipfile.ZipFile(archive) as stream:
            members = [item for item in stream.infolist() if not item.is_dir()]
            return bool(members) and all(
                (output / item.filename).is_file()
                and (output / item.filename).stat().st_size == item.file_size
                for item in members
            )
    except zipfile.BadZipFile:
        return False


def extract_ready(archive_root: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    for archive in sorted(archive_root.glob("*.zip")):
        if already_extracted(archive, output):
            completed += 1
            continue
        try:
            with zipfile.ZipFile(archive) as stream:
                if stream.testzip() is not None:
                    continue
                print(f"Extracting {archive.name}", flush=True)
                stream.extractall(output)
            completed += 1
        except zipfile.BadZipFile:
            continue
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-archives", type=int, default=40)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        completed = extract_ready(args.archive_root, args.output)
        videos = sum(1 for _ in args.output.rglob("*.mp4"))
        print(
            f"Ready archives: {completed}/{args.expected_archives}; videos: {videos}/800",
            flush=True,
        )
        if args.once or completed >= args.expected_archives:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
