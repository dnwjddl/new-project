# Predictive Latent Reasoning: Core Validation

This experiment directly validates the minimal Proposal 3 method on the full
official EPIC-KITCHENS-100 train/validation splits. It reuses the frozen,
video-native VideoFlexTok action-segment cache and does not use the earlier
12-video CLIP smoke test.

The first matched-compute table contains four variants:

- `depth_only`: shared recurrent depth with task supervision only;
- `predict`: adds one-step future-latent likelihood, but does not use its error;
- `raw_correct`: uses the raw future residual during posterior correction;
- `calibrated_correct`: normalizes the residual by predicted uncertainty.

All variants use the same bounded memory, shared recurrent core, task heads,
and one prospective plus one corrective step. The comparison therefore tests
the future objective and correction signal rather than extra depth.

Primary endpoint: validation action class-mean recall@5. Secondary endpoints:
verb/noun recall@5, top-5 accuracy, future-latent cosine/NLL, runtime, and state
update magnitude. Three fixed seeds are paired across variants.

The cache uses all available official samples and keeps videos intact during
validation. Training uses truncated causal windows; validation runs each video
as one continuous stream.

After cache preparation and the Slurm array complete:

```bash
python summarize_core.py \
  --input-dir /scratch/dnwjddl/predictive_latent_reasoning/results \
  --output /scratch/dnwjddl/predictive_latent_reasoning/core_summary.json
```

Proceed to adaptive `K` only if calibrated correction improves the paired
primary endpoint over `predict`, and proceed to multi-horizon rollout only if
one-step prediction itself is useful.

## Current Verdict

The full experiment is complete. Future-latent prediction is learnable, but
raw correction, calibrated correction, and innovation-based gating do not
improve the paired anticipation endpoint. Adaptive `K` and multi-horizon
rollout are therefore paused. See [RESULTS.md](RESULTS.md) and
`results/core_gate_summary.json` for the complete decision record.
