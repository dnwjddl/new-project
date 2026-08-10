# Visual Latent Reasoning Pilot

`latent_reasoning_pilot.py` is a cheap directional test for Proposal 03.

It freezes CLIP-L/14, extracts one center frame per EPIC-KITCHENS-100 action,
and predicts the next verb from the preceding causal context. Three heads are
compared:

- `no_reason`: temporal state with no inner latent loop;
- `shared_recurrent`: one weight-tied reasoning block unrolled for K steps;
- `unshared_depth`: K independent blocks with the same per-step compute.

The useful signal is not absolute accuracy. The proposal only earns a larger
experiment if recurrent accuracy improves as K grows, remains competitive with
unshared depth, and new evidence corrects a meaningful fraction of wrong prior
predictions.

Example Slurm command:

```bash
srun --qos base_qos --partition=suma_rtx4090 --gres=gpu:1 \
  --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
```

Then run:

```bash
conda activate ar_video
python experiments/latent_reasoning_pilot.py \
  --cache /scratch/dnwjddl/latent-pilot/ek100_clip_pilot.pt \
  --results /scratch/dnwjddl/latent-pilot/results.json
```

This pilot is not suitable as paper evidence. A positive signal must be
replicated with a video-native teacher, multiple seeds, matched FLOPs, and a
controlled belief-revision benchmark.

## First run

The first run used 12 videos, 495 train examples, 230 video-disjoint validation
examples, and one seed on an RTX 4090. The shared recurrent curve was:

```text
K=0  30.4%
K=1  32.2%
K=2  32.2%
K=3  31.7%
K=4  31.3%
```

The no-reasoning head reached 32.2%, while shared and unshared K=4 heads both
reached 31.3%. This is a weak/no-go signal for the simple architecture: extra
iterations did not produce monotonic gains or meaningful belief correction.
The next test must add an explicit future-latent residual and a controlled
prior/posterior contradiction target before any large-model scaling.
