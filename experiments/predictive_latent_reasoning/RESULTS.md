# Predictive Latent Reasoning: Core Validation Results

## Scope

This run tests the smallest falsifiable version of Proposal 3 before adding
multi-horizon rollout or adaptive reasoning depth. It uses all available
EPIC-KITCHENS-100 train/validation action segments in the existing frozen
VideoFlexTok cache:

- 67,216 training segments from 495 videos;
- 9,657 validation segments from 138 videos;
- 9,519 causal validation transitions;
- 8 memory tokens, width 192, and 16 pooled observation slots;
- three paired seeds per variant.

Training uses truncated causal windows. Validation preserves complete videos
and processes each one as a continuous stream. Every variant uses the same
bounded memory, task heads, and recurrent compute budget.

## Main Result

| Variant | Action mR@5 (%) | Action top-5 (%) | Future cosine | Runtime / seed |
|---|---:|---:|---:|---:|
| `depth_only` | 1.333 +/- 0.084 | 10.311 | -0.000 | 8.2 min |
| `predict` | 1.309 +/- 0.050 | 10.202 | 0.438 | 8.9 min |
| `raw_correct` | 1.327 +/- 0.103 | 9.940 | 0.426 | 7.0 min |
| `calibrated_correct` | 1.142 +/- 0.232 | 9.991 | 0.436 | 7.0 min |
| `observation_gate` | 1.324 +/- 0.073 | 10.235 | 0.445 | 10.2 min |
| `innovation_gate` | 1.339 +/- 0.291 | 9.772 | 0.445 | 10.3 min |

Values are means across three seeds. The interval beside action mR@5 is the
95% confidence interval used by the aggregation script.

Paired action mR@5 differences:

| Comparison | Mean delta (%p) | 95% CI (%p) |
|---|---:|---:|
| `raw_correct - predict` | +0.017 | +/- 0.130 |
| `calibrated_correct - predict` | -0.168 | +/- 0.281 |
| `innovation_gate - observation_gate` | +0.015 | +/- 0.238 |

The future prediction objective clearly learns a predictable latent signal:
future cosine rises from approximately zero to 0.43-0.45. That signal does not
improve anticipation under the current correction rule. Raw correction is
indistinguishable from prediction alone, and uncertainty-normalized correction
is worse on average.

The learned gates also converge close to always updating. The selected
innovation-gate runs have mean update gates between 0.978 and 0.997. Therefore,
the task objective plus a weak gate-cost penalty does not teach selective
memory revision, and calibrated innovation does not outperform an
observation-only gate.

## Decision

**Validated:** a bounded recurrent state can predict the next frozen video
latent on the full split.

**Rejected in its current form:** directly injecting raw or
uncertainty-normalized latent residuals as the correction signal.

**Not started:** multi-horizon prediction and adaptive latent depth. Both would
add complexity before the one-step correction mechanism has shown task value.

The next test should replace the weak gate penalty with a value-aligned target:
supervise each write by the measured reduction in future semantic/task risk,
then compare an observation-only controller with a value-aligned innovation
controller at matched compute.

## Interpretation Boundary

This is a full-split structural validation, not an official leaderboard result.
It uses cached action-segment VideoFlexTok features and a custom causal-stream
probe rather than the complete official anticipation protocol. The low absolute
action mR@5 means these numbers cannot support a state-of-the-art claim. They
are sufficient for the narrower decision tested here: whether the proposed
prediction-error correction improves a matched model under this controlled
setup. It does not.

Raw per-seed files and aggregate statistics are stored in `results/`.
