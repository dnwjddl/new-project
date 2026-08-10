# Visual Latent Reasoning Pilot

`latent_reasoning_pilot.py` is a smoke test for the Proposal 03 training and
evaluation pipeline.

It freezes CLIP-L/14, extracts one center frame per EPIC-KITCHENS-100 action,
and predicts the next verb from the preceding causal context. Three heads are
compared:

- `no_reason`: temporal state with no inner latent loop;
- `shared_recurrent`: one weight-tied reasoning block unrolled for K steps;
- `unshared_depth`: K independent blocks with the same per-step compute.

Its outputs may expose implementation problems, collapsed representations, or
broken metrics. They must not be used to accept or reject the research
hypothesis.

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

This smoke test is not suitable as paper evidence. A research conclusion
requires official dataset splits, a video-native pretrained backbone, at least
three seeds with confidence intervals, matched-compute baselines, and a
controlled belief-revision benchmark. The central result should also replicate
on a second dataset.

## First run

The first smoke run used 12 videos, 495 train examples, 230 video-disjoint
validation examples, and one seed on an RTX 4090. The shared recurrent curve
was:

```text
K=0  30.4%
K=1  32.2%
K=2  32.2%
K=3  31.7%
K=4  31.3%
```

The no-reasoning head reached 32.2%, while shared and unshared K=4 heads both
reached 31.3%. These numbers only confirm that the metric can detect a
non-monotonic step curve in the current toy pipeline. They do not establish
that recurrent latent reasoning works or fails. The paper-grade study must test
an explicit future-latent residual and controlled prior/posterior contradiction
target under the evidence policy below.

## Evidence policy

A run may inform a paper claim only when all of the following hold:

- official train/validation splits and the complete evaluation set;
- a strong video-native pretrained backbone and task-appropriate baselines;
- at least three independent seeds with 95% confidence intervals;
- matched FLOPs, memory-token count, and wall-clock latency comparisons;
- preregistered primary metrics and ablations that isolate the claimed cause;
- replication of the central effect on a second temporal-video benchmark.

For Proposal 03, the primary tests are anytime step scaling, matched-compute
recurrence versus unshared depth, and calibrated correction after deliberately
late or contradictory evidence. For Proposal 01 and Proposal 02, the same
policy applies; synthetic query workloads and small video subsets remain smoke
tests only.
