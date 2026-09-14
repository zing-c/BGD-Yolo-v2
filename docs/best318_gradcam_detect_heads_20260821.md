# Best318-era Grad-CAM Detect-head experiments (2026-08-21)

> **Archived legacy results:** these runs predate the corrected in-place Detail scatter,
> `[x,y] -> BCHW [y,x]` mapping, and scale-aware P3/P4/P5 patch support. Use
> [`CORRECTED_SINGLE_HEAD_SUM_RESULTS.md`](CORRECTED_SINGLE_HEAD_SUM_RESULTS.md) for the
> corrected no-alpha single-head ablation used in current comparisons.

This snapshot records four 50-epoch whole-network fine-tuning runs derived from
the independent YOLO and Detail checkpoints that preceded `best318.pt`.

## Shared training configuration

- image size: 640
- epochs: 50
- optimizer: AdamW
- initial learning rate: 0.0001
- final learning-rate factor: 0.01
- momentum: 0.937
- weight decay: 0.0005
- warmup: 3 epochs, momentum 0.8, bias LR 0.0
- batch size: 32
- AMP: enabled
- candidate confidence: 0.2
- patience: 50
- seed: 0 (Ultralytics default)

The low/P5 run was resumed after a host-RAM OOM. It resumed from epoch 7 with
two DataLoader workers and preserved the model, optimizer, EMA, scheduler, and
epoch state. The high/P3 and mid/P4 runs completed normally. Running these
Grad-CAM models in parallel with eight workers each used roughly 14-15 GB host
RAM per main process and caused the OOM; sequential execution with two workers
was stable.

## Variants

| Run | Detect target | Fusion rule | W&B |
| --- | --- | --- | --- |
| low/P5 | `Detect.cv3[2][1]` | `base + 0.5 * (fused - base)` | [1ghxfq1l](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/1ghxfq1l) |
| mid/P4 | `Detect.cv3[1][1]` | `base + 0.5 * (fused - base)` | [a5ucwbtv](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/a5ucwbtv) |
| mid/P4 learnable | `Detect.cv3[1][1]` | `base + sigmoid(beta) * (fused - base)`, alpha initialized at 0.5 | [cy2abxu9](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/cy2abxu9) |
| high/P3 | `Detect.cv3[0][1]` | original direct fused-logit replacement, no alpha | [2mx54q9b](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/2mx54q9b) |

Best rows in the 50-epoch validation CSV files:

| Run | Epoch | P | R | mAP50 | mAP75 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low/P5 alpha 0.5 | 28 | 0.96998 | 0.94313 | 0.97394 | 0.94600 | 0.91971 |
| mid/P4 alpha 0.5 | 41 | 0.96598 | 0.94211 | 0.97255 | 0.94352 | 0.91871 |
| mid/P4 learnable alpha, max mAP50-95 row | 50 | 0.98497 | 0.92417 | 0.97282 | 0.94928 | 0.91925 |
| high/P3 direct | 49 | 0.97059 | 0.93852 | 0.97250 | 0.94742 | 0.92072 |

This fork selects `best.pt` using validation mAP50. For the learnable-alpha
run, `best.pt` is therefore epoch 14 (`mAP50=0.97334`, `mAP50-95=0.91233`,
`alpha=0.49828720`), while `last.pt` is epoch 50 (`mAP50=0.97282`,
`mAP50-95=0.91925`, `alpha=0.49697876`). The official unified test below uses
the predeclared `best.pt`, not a post-hoc choice of `last.pt`.

## Exact test comparison

All five tests used the same test split and evaluation settings:
`imgsz=640`, `conf=0.5`, `iou=0.5`, `batch=1`, and GPU 0.

| Model | P | R | mAP50 | mAP75 | mAP50-95 | inference ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `best318.pt` baseline | 0.954128 | 0.888889 | 0.926514 | 0.901398 | 0.872197 | 54.9 |
| low/P5 alpha 0.5 | 0.950123 | 0.895489 | 0.935050 | 0.916033 | **0.887504** | 53.33 |
| mid/P4 alpha 0.5 | 0.945701 | 0.893162 | 0.932220 | 0.912353 | 0.882639 | 53.03 |
| mid/P4 learnable alpha (`best.pt`) | 0.945510 | 0.889851 | 0.935196 | 0.916356 | 0.885936 | **52.15** |
| high/P3 direct | 0.942222 | **0.905983** | **0.937386** | 0.907642 | 0.882735 | 53.11 |

The low/P5 alpha-0.5 model improves test mAP50-95 by 0.015307 over
`best318.pt`; consequently the conditional follow-up with another alpha was
not triggered. The mid/P4 alpha-0.5 model improves test mAP50-95 by 0.010442,
mAP75 by 0.010955, and mAP50 by 0.005706 over `best318.pt`, so its conditional
alpha retry was not triggered either. The separate direct multi-head experiment
is documented in [`best318_multihead_direct.md`](best318_multihead_direct.md).
Making the P4 alpha learnable improves over fixed P4 by 0.002976 mAP50,
0.004002 mAP75, and 0.003297 mAP50-95, while reducing inference by about
0.87 ms/image. Relative to `best318.pt`, it improves mAP50-95 by 0.013739.

## Artifacts

All completed checkpoints listed below are published in the GitHub Release
[best318-gradcam-heads-20260821](https://github.com/zing-c/BGD-Yolo-v2/releases/tag/best318-gradcam-heads-20260821).

| Model / test mAP50-95 | GitHub weight | Local weight | Size | SHA-256 |
| --- | --- | --- | ---: | --- |
| low/P5 alpha 0.5 / **0.887504** | [best318_low_p5_alpha05_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_low_p5_alpha05_best.pt) | `experiments/runs/best318_low_detectconv_gradcam_alpha05_e50_lr1e4/weights/best.pt` | 685,018,627 bytes | `f1a976ed1e34a7c9da759cd772600e6900f90ac18ab394db1dcb42a8fa234a4b` |
| mid/P4 alpha 0.5 / **0.882639** | [best318_mid_p4_alpha05_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_mid_p4_alpha05_best.pt) | `experiments/runs/best318_mid_p4_detectconv_alpha05_e50_lr1e4/weights/best.pt` | 685,017,091 bytes | `1042eef87aa36bb2af7d0a97f84ff39934c0b51c1696e9cfde0309d09fcb6a30` |
| mid/P4 learnable best / **0.885936** | [best318_mid_p4_learnable_alpha_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_mid_p4_learnable_alpha_best.pt) | `experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/weights/best.pt` | 685,017,534 bytes | `e0a53a49f76bc9d4675b35191e7b5b4689497489e77febcc8bc57df1bcb4687b` |
| mid/P4 learnable last / not test-selected | [best318_mid_p4_learnable_alpha_last.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_mid_p4_learnable_alpha_last.pt) | `experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/weights/last.pt` | 685,017,534 bytes | `91b94bf4250e2cedaa9f72ca901411334b9925b5b3859b2a698b4301861cf809` |
| high/P3 direct / **0.882735** | [best318_high_p3_direct_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_high_p3_direct_best.pt) | `experiments/runs/best318_high_detectconv_direct_e50_lr1e4/weights/best.pt` | 685,017,027 bytes | `75e20890c69e3da39d3c477d2877a384c025ad6eb36f2b4555ed2dfb995bf904` |

The local paths and checksums are repeated below for command-line use:

- `experiments/runs/best318_low_detectconv_gradcam_alpha05_e50_lr1e4/weights/best.pt`
  - SHA-256: `f1a976ed1e34a7c9da759cd772600e6900f90ac18ab394db1dcb42a8fa234a4b`
- `experiments/runs/best318_mid_p4_detectconv_alpha05_e50_lr1e4/weights/best.pt`
  - SHA-256: `1042eef87aa36bb2af7d0a97f84ff39934c0b51c1696e9cfde0309d09fcb6a30`
- `experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/weights/best.pt`
  - SHA-256: `e0a53a49f76bc9d4675b35191e7b5b4689497489e77febcc8bc57df1bcb4687b`
- `experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/weights/last.pt`
  - SHA-256: `91b94bf4250e2cedaa9f72ca901411334b9925b5b3859b2a698b4301861cf809`
- `experiments/runs/best318_high_detectconv_direct_e50_lr1e4/weights/best.pt`
  - SHA-256: `75e20890c69e3da39d3c477d2877a384c025ad6eb36f2b4555ed2dfb995bf904`

The lightweight CSV histories and exact test metrics are committed beside this
report. Model artifacts are also associated with the W&B runs linked above.
