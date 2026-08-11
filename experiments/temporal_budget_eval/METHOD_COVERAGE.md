# Dynamic video-token method coverage

This file records which published methods can enter the exact-global-budget
TempCompass comparison without changing the task, backbone, or meaning of the
budget. It prevents results from incompatible protocols from being presented in
one table.

Repository snapshots inspected on 2026-08-10: VQToken `ed949e019f6a`, EarlyTom
`83d80e0c52ab`, STC `cf53f781d874`, Dynamic-VLM/ByteVideoLLM `f0d5e5abf9e1`,
HaltingVT `4830e2aa5988`, and PyraTok `7e6ecca6343c`.

## Exact global K comparison

| Entry | Status | What is held constant | Reporting label |
|---|---|---|---|
| [VQToken](https://github.com/Hai-chao-Zhang/VQToken) | Full evaluation complete | LLaVA-OneVision 0.5B, 32 frames, TempCompass prompts and decoding | VQToken-style K-means on public base |
| [EarlyTom](https://github.com/viridisGreen/EarlyTom) | Full evaluation complete | Same model, frames, prompts, decoding, and exact K | EarlyTom + order-preserving budget cap |
| Uniform sample | Full evaluation complete | Same encoded features and exact K | Ordered uniform sampling control |
| Uniform pool | Full evaluation complete | Same encoded features and exact K | Contiguous sequence pooling control |
| Feature top-K | Full evaluation complete | Same encoded features and exact K | Feature-norm top-K control |

The released VQToken-trained checkpoint is gated. The public-base entry isolates
the compressor's sensitivity to K and is not an official checkpoint
reproduction. EarlyTom natively emits at least 13 visual tokens in this setup;
K=4/8/16/32 therefore uses the official selector followed by a documented,
order-preserving cap. A native EarlyTom ratio run is reported separately.

## Separate protocol required

| Method | Why it is not in the exact-K VideoQA table | Proper comparison |
|---|---|---|
| [STC](https://github.com/lern-to-write/STC) | Its pruner allocates tokens per frame and its cacher changes ViT recomputation. With 32 frames, K=4 or K=8 is not a native setting. | ReKV or another supported streaming framework on OVO-Bench/StreamingBench; report per-frame tokens, ViT latency, prefill latency, and streaming accuracy. |
| [HaltingVT](https://github.com/dun-research/HaltingVT) | It performs layer-wise token halting for action recognition and does not expose a VideoQA representation with a global output K. | Mini-Kinetics or ActivityNet with top-1 accuracy versus encoder GFLOPs. |
| [STA](https://github.com/Mark12Ding/STA) | Its semantic-aware temporal accumulation progressively prunes ViT/VideoSwin tokens for action recognition; the released protocol is Kinetics-400 and Something-Something V2, not VideoQA. | Reproduce its native top-1 accuracy versus encoder GFLOPs and use its pruning score as an ablation only after adapting a common VideoQA backbone. |
| [Dynamic-VLM](https://github.com/Hon-Wong/ByteVideoLLM) | The public 14B checkpoint and evaluation stack are model-specific; changing it to the 0.5B common decoder would no longer reproduce the paper. | Run the released checkpoint on its supported VideoQA suite and compare matched retention/latency within that stack. |
| [PyraTok](https://github.com/PLAN-Lab/PyraTok) | It is a learned discrete video VAE/tokenizer with pyramidal codebooks, not a drop-in visual-token pruner for LLaVA-OneVision. | Evaluate its released hierarchy on reconstruction and supported zero-shot understanding tasks at native pyramid levels. |
| [SCORE](https://arxiv.org/abs/2603.26365) | No public implementation or checkpoint was located as of 2026-08-10. | Reproduce after code release; until then, cite reported results only and do not create an unofficial result under its name. |

## Interpretation rule

Only methods in the first table support a paired answer-level test at the same
global K. Results from the second table can establish external validity and
system-level efficiency, but they must remain in separate tables with their
native task and compute definitions.
