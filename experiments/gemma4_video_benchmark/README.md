# Gemma 4 Video Benchmark Reproduction

This directory evaluates `google/gemma-4-E2B-it` and `google/gemma-4-E4B-it`
with the same deterministic video protocol.

## Initial protocol

- TempCompass objective tasks: multi-choice, yes/no, and caption matching
- 32 frames sampled uniformly over the full clip
- Native Gemma 4 video processor, BF16, SDPA
- Original frame timestamps supplied to the Gemma 4 video processor
- Thinking disabled and greedy answer-only decoding
- Eight independent inference shards per model

The official captioning task is excluded from exact-match aggregation because it
requires an external semantic judge. It can be added as a separate judged run.

The second protocol uses Video-MME-v2:

- all 3,200 questions over 800 videos
- 64 fixed frames with the official frame-index rule
- visual-only, non-reasoning prompt from the official benchmark
- greedy decoding with a 128-token generation cap
- average accuracy and the official grouped nonlinear score
- eight video-grouped inference shards per model
- one 64-frame decode reused across each video's four related questions

## Server layout

The Slurm jobs use `/scratch/dnwjddl/gemma4_video_eval`. The environment is
linked as `env`, code is copied to `code`, and each shard writes resumable JSONL
under `results/<model>/`.

Video-MME-v2 ships as 40 archives. `extract_videommev2.py` watches the download
directory and extracts each complete archive without redoing finished files.
`download_videommev2.sh` downloads the archives concurrently, resumes partial
files, validates each zip, and exposes an archive to the extractor only after
the transfer finishes.

```bash
./download_videommev2.sh 1 40 8
```

`transformers==5.15.0` returns Gemma 4's split video features as a tuple while
its forward path expects one tensor. The evaluators conditionally concatenate
that tuple; already-fixed tensor outputs are left unchanged.

## Summarize

```bash
python summarize_tempcompass.py results/model/tempcompass_32f_shard*.jsonl \
  --output results/model/tempcompass_32f_summary.json

python summarize_videommev2.py results/model/videommev2_64f_shard*.jsonl \
  --output results/model/videommev2_64f_summary.json
```
