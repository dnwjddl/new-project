#!/usr/bin/env python3
"""Inspect the released VideoFlexTok encoder's intermediate tensor contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from videoflextok.utils.demo import read_mp4
from videoflextok.utils.misc import detect_bf16_support, get_bf16_context
from videoflextok.wrappers import VideoFlexTokFromHub


def describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    return type(value).__name__


def snapshot(data: dict[str, Any]) -> dict[str, Any]:
    return {key: describe(value) for key, value in sorted(data.items())}


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = VideoFlexTokFromHub.from_pretrained(
        args.model_id,
        local_files_only=True,
    ).to("cuda").eval()
    preprocess_args = dict(model.video_preprocess_args)
    if "overlap_size_frames" in preprocess_args:
        preprocess_args["overlap_size"] = preprocess_args.pop("overlap_size_frames")
    read_args = dict(preprocess_args)
    # The released helper's validity check rejects a non-overlapping 17-frame
    # chunk even though the model wrapper requires exactly that input.
    if read_args.get("overlap_size") == 0:
        read_args["overlap_size"] = 1
    video = read_mp4(
        str(Path(args.video)),
        num_frames=args.num_frames,
        **read_args,
    )
    base = model.video_tokenizer
    report: dict[str, Any] = {
        "model_id": args.model_id,
        "gpu": torch.cuda.get_device_name(0),
        "video": describe(video),
        "video_preprocess_args": preprocess_args,
        "read_mp4_args": read_args,
        "images_read_key": base.vae.images_read_key,
        "prequant_key": base.regularizer.latents_read_key,
        "quant_key": base.regularizer.quants_write_key,
        "token_key": base.regularizer.tokens_write_key,
    }
    data = {base.vae.images_read_key: [video[None].to("cuda")]}
    with get_bf16_context(detect_bf16_support()):
        data = base.vae.encode(data)
        report["after_vae"] = snapshot(data)
        data = base.encoder(data)
        report["after_encoder"] = snapshot(data)
        data = base.regularizer(data)
        report["after_regularizer"] = snapshot(data)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="EPFL-VILAB/videoflextok_d18_d18_k600")
    parser.add_argument(
        "--video",
        default=(
            "/home/dnwjddl/project1_temporal_audit/vendor/ml-videoflextok/"
            "data/video_examples/porsche.mp4"
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=17,
        help="One released-model chunk; do not bypass chunking with a longer tensor.",
    )
    parser.add_argument(
        "--output",
        default="/home/dnwjddl/project1_temporal_audit/results/videoflextok_contract.json",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
