# Proposal 1-2 full-split feasibility audits

This experiment measures how much standard EPIC-Kitchens-100 verb, noun, and
action accuracy changes when temporal evidence is perturbed while appearance is
held as constant as possible.

## Protocol

- Backbone: frozen ViCLIP-L/14.
- Data: complete official EPIC-Kitchens-100 train and validation splits.
- Probe: independent linear verb and noun heads, trained with three seeds.
- Conditions: ordered, reversed, deterministically shuffled, single middle
  frame, half-rate sparse frames, and second-half freeze.
- Primary comparison: standard ordered accuracy versus temporal accuracy
  retention under each intervention.

Feature extraction is sharded by whole videos, so annotations from a video are
never split across workers. Training features are extracted only for ordered
clips; all conditions are extracted from each validation clip's same decoded
frame set.

## Proposal 1: ordered-token role audit

The released `EPFL-VILAB/videoflextok_d18_d18_k600` checkpoint is evaluated
without retraining. The paper-grade pilot uses a deterministic 16,000-example
training subset that covers every one of the 3,568 observed EK100 verb-noun
action strata, plus the complete 9,668-example validation split. Ordered and
reversed clips are encoded into 256 ordered registers at six prefix budgets:
8, 16, 32, 64, 128, and 256.

The analysis reports cumulative-prefix and marginal-block verb, noun, and
action accuracy; temporal retention under reversal; token entropy and temporal
change; and block-to-block linear CKA. Three probe seeds are used. This is an
audit of the released ordering, not yet evidence that the proposed
state-innovation objective improves it.

## Proposal 2: real multi-query amortization

The multi-query study uses official Ego4D NLQ train and validation annotations,
not synthetic query types. Frozen CLIP-L/14 features form a query-agnostic
4-second leaf sequence for each clip. A small router is trained on the official
training split and evaluated on all validation queries with three seeds.

Three retrieval structures are compared at 1, 2, 4, and 8 opened leaves:

- flat dense retrieval;
- a query-agnostic uniform temporal tree;
- a query-agnostic visual-change tree.

Primary accuracy metrics are evidence-overlap hit rate, target-center hit rate,
top-1 temporal IoU, and normalized center error. Efficiency metrics include
router comparisons, unique opened leaves per query, cumulative branch-cache
ratio, and total visual cost as 1, 2, 4, or 8 real queries accumulate on the
same clip. Four official entries with missing query text are excluded and
reported explicitly.

## Measured findings

### Proposal 1: the standard probe is weakly temporal

The ViCLIP audit used 67,216 decodable training segments and 9,659 decodable
validation segments with three probe seeds. One train segment and nine
validation segments were excluded because the source MP4 packets were
corrupted; all conditions use the same retained validation set.

- Ordered action top-1: 15.02%.
- Reversed action top-1: 14.60%, a 0.42 percentage-point drop.
- Shuffled action top-1: 14.62%, a 0.40 percentage-point drop.
- Single-frame repetition action top-1: 10.24%, a 4.78 percentage-point drop.

The common frozen-feature action probe responds strongly when visual evidence
is removed, but barely responds when temporal order is destroyed. This
supports the benchmark-audit premise of Proposal 1; it does not yet establish
that the proposed state-innovation objective improves the encoder.

The released VideoFlexTok ordering was then audited on a deterministic 16,000
sample train set covering all 3,568 action strata and 9,657 decodable
validation segments. Concatenated quantized prefixes peak at 4.10% action top-1
with 8 tokens, versus 3.35% with all 256 tokens. A fixed 18-dimensional pooling
control preserves the pattern: 2.77+-0.14% for the 8-token prefix versus
1.83+-0.09% for 256 tokens. The first 8-token block reaches 2.69%, while later
blocks range from 1.15% to 1.63%. Thus action and order evidence is front-loaded
rather than successively refined by later token groups. Explicit stable-state
and order-sensitive-innovation roles remain a meaningful method target.

### Proposal 2: reuse works, but the hierarchy has an accuracy gap

The full Ego4D NLQ study contains 1,271 train clips, 415 validation clips,
13,847 usable train queries, 4,552 usable validation queries, and 209,486
cached leaves. There were no decode failures.

- At leaf budget 4 and Q=8, shared-cache visual cost is 14.7% of independent
  per-query execution, or about 6.8x lower.
- Opened leaves per query fall from 4.00 at Q=1 to 3.27 at Q=8 because branches
  discovered by earlier queries are reused.
- At budget 8, uniform-tree routing uses 47.3 comparisons versus 137.5 for flat
  search, a 65.6% reduction.
- Raw CLIP overlap hit is 41.65% for flat search and 31.88% for the uniform
  tree. The current hierarchy therefore does not yet preserve flat accuracy.
- An unconstrained 256-dimensional learned router degrades flat overlap hit to
  33.44%. A residual router with CLIP-geometry alignment weight 10 recovers it
  to 40.79%, showing that pretrained alignment should be preserved rather than
  replaced.

The feasible paper direction is cache-aware hierarchical routing with flat-rank
distillation and geometry-preserving residual adaptation. Multi-query reuse is
validated, while closing the roughly 10-point tree gap remains the method's
central go/no-go test.

## Measured runtime envelope

- ViCLIP EK100 extraction: 8 mixed RTX 4090/A6000 GPUs. The slowest complete
  train shard took 2h53m and the slowest validation shard took 27m38s; the
  three-seed probe took 49 seconds on one A6000.
- VideoFlexTok on RTX 3090: batch 12 is the measured optimum at about 0.477
  clips/s and 17.95 GiB peak reserved memory. Batch 16 is slower and reaches
  22.50 GiB. With 8 mixed RTX 3090/A6000/4090 GPUs, the slowest 2,000-sample
  train shard took 52.4 minutes and the slowest validation shard took 43.6
  minutes. The concatenated and fixed-pool three-seed analysis took 8m23s on
  one RTX 4090.
- Ego4D NLQ leaves: 8 mixed RTX 3090/A6000 GPUs, 4-second cadence, 157,459 train
  and 52,027 validation leaves. The slowest train shard took 55.2 minutes and
  the slowest validation shard took 20.8 minutes. Full three-seed router/tree
  analysis took 76 seconds on one A6000 after eliminating repeated NPZ reads
  and repeated per-budget traversal.

Smoke runs validate code, memory, and throughput only. Research conclusions are
drawn only from the official-split, three-seed outputs described above.
