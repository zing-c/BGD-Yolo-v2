# YOLO-FineDet: Code Structure and Reproduction

This branch contains the no-alpha YOLO-FineDet implementation for
independent P3/P4/P5 experiments and one shared P3+P4+P5 model.

## Method structure

```text
640x640 image
  -> YOLO backbone and neck
  -> Detect classification feature(s): P3, P4 and/or P5
  -> candidates with confidence > 0.2
  -> Grad-CAM candidate localization
  -> native 64x64 detail crop
  -> shared Detail encoder -> 3x4x4 feature
  -> input64 scatter: P3=8x8, P4=4x4, P5=2x2
  -> concat(global 64ch, detail 3ch)
  -> head-specific Conv3x3(67->64) + BN + ReLU
  -> original classification predictor
```

The regression branch is unchanged. Detail features are accumulated with
in-place sum, and no fixed or learnable alpha is used. In the shared model,
YOLO and the Detail encoder are shared while P3/P4/P5 have independent fusion
projections.

## Main files

```text
experiments/run_318_fusion_alpha.py     train/validation entry point
experiments/direct_geometry_fusion.py   CAM, crop, coordinate mapping and fusion
experiments/direct_geometry_monitor.py  zero-CAM monitoring
experiments/legacy_bgd/{val,gradcam}.py  embedded legacy wrapper compatibility
experiments/launch_geometry_v5_heads.py independent P3/P4/P5 launcher
experiments/launch_input64_multi.py     shared P3+P4+P5 launcher
ultralytics/yolo/data/direct_geometry.py augmentation-aware source geometry
ultralytics/yolo/data/{base,augment,dataset}.py
ultralytics/nn/tasks.py                 batch metadata and custom forward
ultralytics/yolo/engine/{trainer,validator}.py
Resnet6.py                              historical wide Detail encoder
```

## Key hyperparameters

| Item | Value |
|---|---:|
| Input size | 640 |
| Detail crop | 64x64 |
| Candidate confidence | 0.2 |
| Scatter | sum, no alpha |
| P3 / P4 / P5 support | 8x8 / 4x4 / 2x2 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Batch size | 32 |
| Epochs | 50 |
| Best checkpoint | validation mAP50 |
| Test confidence / NMS IoU | 0.5 / 0.5 |

## Required local inputs

```bash
export FINEDET_YOLO_WEIGHT=/absolute/path/to/yolo_best.pt
export FINEDET_DETAIL_WEIGHT=/absolute/path/to/detail_net_attn.pt
export FINEDET_DATA_YAML=/absolute/path/to/BrokenGlass.yaml
```

Weights and datasets are not stored in this branch.

## Independent P3/P4/P5 training

Run each head as an independent experiment. Each command trains one model and
evaluates its validation-selected `best.pt` on Test.

P3:

```bash
python -m experiments.launch_geometry_v5_heads \
  --heads high \
  --epochs 50 \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" \
  --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --data "$FINEDET_DATA_YAML"
```

P4:

```bash
python -m experiments.launch_geometry_v5_heads \
  --heads mid \
  --epochs 50 \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" \
  --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --data "$FINEDET_DATA_YAML"
```

P5:

```bash
python -m experiments.launch_geometry_v5_heads \
  --heads low \
  --epochs 50 \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" \
  --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --data "$FINEDET_DATA_YAML"
```

## Shared P3+P4+P5 training

This command trains one model with one shared YOLO, one shared Detail encoder
and three independent fusion projections:

```bash
python -m experiments.launch_input64_multi \
  --epochs 50 \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" \
  --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --data "$FINEDET_DATA_YAML"
```
