# Fixed-budget temporal evidence diagnostic

Run date: 2026-08-11 KST

## Setup

- Data: full TempCompass multi-choice split, 1,580 questions from 410 videos.
- Temporal evaluation: 401 counterfactual families whose correct semantic
  answer changes under reversal, concatenation/order, direction, speed, or
  attribute change.
- Common stack: public `lmms-lab/llava-onevision-qwen2-0.5b-ov`, 32 sampled
  frames, identical prompts, decoding, and model weights.
- Budgets: exactly 4, 8, 16, or 32 output visual tokens.
- Compute: one RTX 3090 per configuration. All 20 full runs completed model
  inference in 27:50-37:42 and the complete job in less than 55 minutes.
- Statistics: 10,000-replicate bootstrap 95% confidence intervals and paired
  exact McNemar tests on the same questions or counterfactual families.

The official VQToken-trained checkpoint is gated and was unavailable to the
server account. `VQToken-style` therefore means the official test-time K-means
compressor on the public base, not the paper checkpoint. EarlyTom natively emits
at least 13 tokens in this stack; `EarlyTom + cap` applies an order-preserving
exact-K cap after the official selector. The three remaining entries are common
backbone controls rather than separately trained checkpoints.

## Full results

| Compression | K | QA accuracy | Strict family | Answer flip | Inference |
|---|---:|---:|---:|---:|---:|
| VQToken-style K-means | 4 | 37.34 | 1.50 | 23.19 | 34:45 |
| VQToken-style K-means | 8 | 38.29 | 1.75 | 23.19 | 36:18 |
| VQToken-style K-means | 16 | 37.78 | 1.00 | 25.69 | 34:32 |
| VQToken-style K-means | 32 | 38.73 | 2.00 | 27.43 | 37:42 |
| EarlyTom + cap | 4 | 37.47 | 2.00 | 21.20 | 28:37 |
| EarlyTom + cap | 8 | 37.66 | 1.75 | 21.95 | 28:20 |
| EarlyTom + cap | 16 | 39.05 | 1.50 | 21.20 | 27:50 |
| EarlyTom + cap | 32 | 43.16 | 1.75 | 24.94 | 27:51 |
| Ordered uniform sample | 4 | 38.61 | 2.00 | 24.94 | 33:58 |
| Ordered uniform sample | 8 | 41.96 | 2.24 | 25.94 | 32:01 |
| Ordered uniform sample | 16 | 43.54 | 3.24 | 25.19 | 31:26 |
| Ordered uniform sample | 32 | 45.13 | 1.75 | 25.19 | 31:14 |
| Contiguous uniform pool | 4 | 37.53 | 2.00 | 23.19 | 32:54 |
| Contiguous uniform pool | 8 | 38.16 | 2.24 | 23.69 | 33:16 |
| Contiguous uniform pool | 16 | 39.37 | 2.24 | 24.19 | 32:33 |
| Contiguous uniform pool | 32 | 40.00 | 2.24 | 21.45 | 32:55 |
| Feature-norm top-K | 4 | 38.16 | 1.25 | 20.95 | 31:25 |
| Feature-norm top-K | 8 | 37.97 | 1.25 | 20.20 | 31:17 |
| Feature-norm top-K | 16 | 37.97 | 1.00 | 19.95 | 32:50 |
| Feature-norm top-K | 32 | 37.91 | 0.75 | 20.20 | 32:49 |

All values except time are percentages. Strict family accuracy requires every
member of a changed-answer family to be correct. It has a severe floor effect in
this public 0.5B base, so answer-flip consistency is the more sensitive temporal
diagnostic at this stage.

## Native-scale EarlyTom reference

EarlyTom's official native retain ratio of 0.10 produced 631-633 tokens per
video. At that native scale it reached 52.34 QA accuracy, 5.99 strict-family
accuracy, and 24.69 answer-flip consistency; model inference took 27:46 on one
RTX 3090. This row is intentionally excluded from the exact K=4-32 table because
its token budget is roughly 20 times larger than K=32.

Against `EarlyTom + cap` at K=32, native EarlyTom gains 9.18 QA points
(`p=1.54e-14`) and 4.24 strict-family points (`p=0.0015`), but answer-flip is
effectively unchanged (-0.25 points, `p=1.0`). The native run establishes that
the released method and model stack work at their intended scale; the exact-K
adapter specifically probes the extreme low-budget regime.

## Main findings

### 1. Ordinary QA hides low-budget temporal loss

For VQToken-style compression, K=4 and K=8 are both 4.24 points below K=32
on answer-flip consistency (`p=0.0115`), while their ordinary QA differences
from K=32 are not significant (`p=0.153` and `p=0.658`). A representation can
therefore preserve aggregate QA while losing evidence needed to react to a
temporal counterfactual.

### 2. More tokens can improve QA while hurting temporal sensitivity

For contiguous pooling, K=32 is 1.84 points better than K=8 on ordinary QA
(`p=0.0060`) but 2.24 points worse on answer-flip consistency (`p=0.0490`).
Relative to K=16, K=32 is also 2.74 points worse on answer flip
(`p=0.0127`). Token count alone is not a monotonic proxy for temporal evidence;
the compression operator determines which information the extra tokens retain.

### 3. Compression methods change the QA and temporal rankings differently

EarlyTom + cap at K=32 is 4.43 QA points above VQToken-style K=32
(`p=0.00003`), yet its answer-flip score is 2.49 points lower (not significant,
`p=0.282`). At K=16, the QA difference is not significant, but EarlyTom + cap
is 4.49 answer-flip points lower (`p=0.0198`). Feature-norm top-K remains near
38% QA at every budget while producing the lowest answer-flip curve. Saliency,
clustering quality, and aggregate QA are therefore insufficient definitions of
a temporal gist.

### 4. Aggregate gains are dominated by the non-counterfactual action subset

Ordered uniform sampling gains 103 additional correct answers from K=4 to K=32.
Eighty-two of them (79.6%) come from the action subset; order contributes zero
and speed contributes three. The aggregate improvement mainly reflects action
or appearance recognition rather than stronger order-sensitive evidence.

## What this establishes

This is positive evidence for the Proposal 1 problem statement, not evidence
that the proposed encoder is already solved:

1. Existing fixed-budget evaluation can rank compressors incorrectly for a
   temporal-gist objective.
2. A useful early representation needs an explicit temporal sufficiency target,
   not only reconstruction, saliency, full-feature imitation, or ordinary QA.
3. The next method experiment should train the minimum prefix directly on
   order/state-change counterfactual risk and test whether its answer-flip curve
   becomes monotonic without sacrificing the QA-compute Pareto frontier.

The result does not establish that the official VQToken checkpoint fails, and
`EarlyTom + cap` is an extreme-budget adapter rather than an official native
setting. Native STC, HaltingVT, STA, Dynamic-VLM, and PyraTok results require
different task and budget definitions; see `METHOD_COVERAGE.md` instead of
mixing those numbers into this exact-global-K table.

## Artifacts

- `results/method-matrix.json`: all 20 exact-budget metric rows and confidence
  intervals.
- `results/method-tests-k{4,8,16,32}.json`: paired comparisons against
  VQToken-style compression at each K.
- `results/{method}-budget-tests.json`: within-method lower-K comparisons
  against K=32.
- `results/{method}-k{4,8,16,32}.json`: per-method metrics and aspect breakdowns.
- `results/earlytom-native-r010.json`: native-ratio EarlyTom reference at
  631-633 realized tokens.
- `results/earlytom-native-tests.json`: paired native-scale comparisons against
  the K=32 EarlyTom and VQToken-style rows.
- `score_tempcompass_pairs.py`: semantic-answer and counterfactual-family scorer.
- `compare_budget_runs.py`: exact paired tests.
- `METHOD_COVERAGE.md`: native-protocol and reproducibility boundary.
