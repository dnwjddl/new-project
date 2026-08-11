# Temporal evidence under a fixed video-token budget

This experiment tests whether a compressed video representation preserves temporal evidence, rather than only reporting ordinary VideoQA accuracy.

The completed full-split results and statistical tests are summarized in [`RESULTS.md`](RESULTS.md).
[`METHOD_COVERAGE.md`](METHOD_COVERAGE.md) separates methods that support the
same exact-global-token comparison from methods that require their native
streaming, recognition, or tokenizer protocol.

## Protocol

- Model: public LLaVA-OneVision 0.5B base with the official VQToken test-time clustering code. The official VQToken-trained checkpoint is gated; use it through `PRETRAIN` after access is approved.
- Data: full TempCompass multi-choice split.
- Budgets: 4, 8, 16, and 32 video tokens. The released adaptive VQToken path uses 32 as its maximum, so larger fixed values are outside the method's calibrated range and make k-means needlessly expensive.
- Reference metric: ordinary sample accuracy.
- Main metric: strict counterfactual-pair accuracy. A pair passes only when the model answers both temporal variants correctly.
- Diagnostic: answer-flip consistency, which checks whether the prediction changes when the temporally modified video changes the correct answer.

The fixed-budget patch changes only VQToken's test-time clustering budget. Model weights, frame sampling, prompts, and decoding remain identical across budgets. Results from the public base isolate compression sensitivity and must not be reported as a reproduction of the VQToken-trained checkpoint.

`fixed_budget_baselines.patch` adds three exact-budget controls to the same public backbone: temporally ordered uniform sampling, contiguous uniform pooling, and feature-norm top-K. Set `VIDEO_TOKEN_BASELINE` to `uniform_sample`, `uniform_pool`, or `norm_topk`; the default remains VQToken K-means. These are controlled compression operators, not reproductions of separately trained checkpoints.

The released repository also contains a constructor-name mismatch between `vq_token.py` and its bundled `KMeansTorch`. `vqtoken_kmeans_api.patch` only aligns `n_clusters/max_iter` with the bundled `num_clusters/max_iteration` API. The official TempCompass evaluator sends responses outside a narrow prefix format to an external GPT judge. `tempcompass_deterministic_mc.patch` accepts unambiguous local A-D answers and disables that nondeterministic fallback when `TEMPCOMPASS_DISABLE_GPT_JUDGE=1`.

`run_earlytom_tempcompass.sbatch` evaluates the official training-free EarlyTom outer compressor before any common exact-budget normalization. The released repository omits `earlytom` from its own wrapper allowlist; `earlytom_release_fixes.patch` repairs that connection and applies the same deterministic TempCompass parser. Native-ratio runs are kept separate from exact-token comparisons until their realized token counts are measured.

The native calibration produced 13, 19, 37, and 67 tokens at retain ratios 0.001, 0.0025, 0.005, and 0.01. Because EarlyTom cannot natively emit 4 or 8 tokens, `earlytom_exact_budget.patch` applies an order-preserving cap after EarlyTom selection. `run_earlytom_exact_budget.sbatch` reports these runs as `EarlyTom + budget cap`, using native ratios 0.001, 0.001, 0.0025, and 0.005 before exact K=4, 8, 16, and 32 caps. A completed native 0.10-ratio reference realized 631-633 tokens and is reported separately in `RESULTS.md`; it is not mixed into the exact-K table.

## Remote layout

The experiment is staged on `/scratch/dnwjddl/project1_budget_eval` and exposed through `/home/dnwjddl/project1_budget_eval` on the Slurm cluster. The environment and caches also live on scratch because the shared home filesystem is full. A single-budget smoke test should be run before submitting the full four-budget array.

```bash
LMMS_LIMIT=16 VQTOKEN_BUDGET=16 sbatch --array=0 slurm/run_vqtoken_tempcompass.sbatch
sbatch slurm/run_vqtoken_tempcompass.sbatch
python aggregate_budget_curve.py \
  /home/dnwjddl/project1_budget_eval/scores/vqtoken-k4.json \
  /home/dnwjddl/project1_budget_eval/scores/vqtoken-k8.json \
  /home/dnwjddl/project1_budget_eval/scores/vqtoken-k16.json \
  /home/dnwjddl/project1_budget_eval/scores/vqtoken-k32.json \
  --output /home/dnwjddl/project1_budget_eval/scores/budget-curve.json
```

`aggregate_budget_curve.py` reports the smallest tested budget that reaches 90% of the largest-budget strict pair accuracy. This is a diagnostic reference, not a claim of optimal stopping.
