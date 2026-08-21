# Best318-era Grad-CAM Detect-head experiments (2026-08-21)

This snapshot records two 50-epoch whole-network fine-tuning runs derived from
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
epoch state. The high/P3 run completed normally. Running these Grad-CAM models
in parallel with eight workers each used roughly 14-15 GB host RAM per main
process and caused the OOM; sequential execution with two workers was stable.

## Variants

| Run | Detect target | Fusion rule | W&B |
| --- | --- | --- | --- |
| low/P5 | `Detect.cv3[2][1]` | `base + 0.5 * (fused - base)` | [1ghxfq1l](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/1ghxfq1l) |
| high/P3 | `Detect.cv3[0][1]` | original direct fused-logit replacement, no alpha | [2mx54q9b](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/2mx54q9b) |

Best rows in the 50-epoch validation CSV files:

| Run | Epoch | P | R | mAP50 | mAP75 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low/P5 alpha 0.5 | 28 | 0.96998 | 0.94313 | 0.97394 | 0.94600 | 0.91971 |
| high/P3 direct | 49 | 0.97059 | 0.93852 | 0.97250 | 0.94742 | 0.92072 |

## Exact test comparison

All three tests used the same test split and evaluation settings:
`imgsz=640`, `conf=0.5`, `iou=0.5`, `batch=1`, and GPU 0.

| Model | P | R | mAP50 | mAP75 | mAP50-95 | inference ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `best318.pt` baseline | 0.954128 | 0.888889 | 0.926514 | 0.901398 | 0.872197 | 54.9 |
| low/P5 alpha 0.5 | 0.950123 | 0.895489 | 0.935050 | 0.916033 | **0.887504** | 53.33 |
| high/P3 direct | 0.942222 | **0.905983** | **0.937386** | 0.907642 | 0.882735 | 53.11 |

The low/P5 alpha-0.5 model improves test mAP50-95 by 0.015307 over
`best318.pt`; consequently the conditional follow-up with another alpha was
not triggered. The next experiment should use direct fusion at multiple Detect
levels and introduce no alpha mixing hyperparameter.

## Artifacts

Weights remain local because each file is about 685 MB and this repository has
no Git LFS installation:

- `experiments/runs/best318_low_detectconv_gradcam_alpha05_e50_lr1e4/weights/best.pt`
  - SHA-256: `f1a976ed1e34a7c9da759cd772600e6900f90ac18ab394db1dcb42a8fa234a4b`
- `experiments/runs/best318_high_detectconv_direct_e50_lr1e4/weights/best.pt`
  - SHA-256: `75e20890c69e3da39d3c477d2877a384c025ad6eb36f2b4555ed2dfb995bf904`

The lightweight CSV histories and exact test metrics are committed beside this
report. Model artifacts are also associated with the W&B runs linked above.
