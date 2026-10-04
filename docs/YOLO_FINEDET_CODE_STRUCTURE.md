# YOLO-FineDet Global--Local Fusion: Code Structure and Commands

This branch contains the corrected Geometry-v5 implementation used by the
current YOLO-FineDet experiments. It supports both independent single-head
models and one shared P3+P4+P5 model. The recommended configuration uses a
native 64x64 detail crop, sum scatter, and no alpha coefficient.

## 1. Supported modes

| Mode | Enabled heads | Shared parameters | Head-specific parameters | Scatter support |
|---|---|---|---|---|
| P3-only | P3 (`high`) | YOLO and Detail encoder inside the run | one P3 fusion projection | 8x8 |
| P4-only | P4 (`mid`) | YOLO and Detail encoder inside the run | one P4 fusion projection | 4x4 |
| P5-only | P5 (`low`) | YOLO and Detail encoder inside the run | one P5 fusion projection | 2x2 |
| Shared multi-head | P3+P4+P5 | one YOLO and one Detail encoder | independent P3/P4/P5 fusion projections | 8x8 / 4x4 / 2x2 |

The shared multi-head mode produces one joint `best.pt`. It does not train
three separate checkpoints. The independent mode launches one complete model
per selected head and therefore produces separate checkpoints.

## 2. Runtime structure

```text
experiments/
├── run_318_fusion_alpha.py       # model construction, checkpoint I/O, train/val entry point
├── direct_geometry_fusion.py     # corrected CAM, native crop, scatter and fusion runtime
├── direct_geometry_monitor.py    # per-epoch CAM-value counters
├── launch_geometry_v5_heads.py   # independent P3/P4/P5 supervisor
├── launch_input64_multi.py       # one shared P3+P4+P5 supervisor
├── test_direct_geometry.py       # geometry, XY, border, CAM and gradient tests
├── test_multi_geometry.py        # multi-head CAM aggregation tests
├── test_geometry_launcher.py     # independent launcher tests
└── test_input64_multi_launcher.py# shared launcher tests

ultralytics/yolo/data/
├── direct_geometry.py            # source-image geometry records and transforms
├── base.py                       # initializes source records before augmentation
├── augment.py                    # propagates records through Mosaic/affine/flip/letterbox
└── dataset.py                    # collates geometry metadata

ultralytics/yolo/engine/
├── trainer.py                    # saves Detail auxiliary weights from the selected EMA
└── validator.py                  # validation path used by the joint model

ultralytics/nn/tasks.py           # model forward/loss integration
```

## 3. Forward path

```text
input image
  -> YOLO backbone and neck
  -> Detect classification activations (P3, P4 and/or P5)
  -> original YOLO candidate boxes, score > 0.2
  -> Grad-CAM on the selected classification activations
  -> candidate-local maximum and exact source-image coordinate mapping
  -> native 64x64 RGB crop from the corresponding source image
  -> shared Detail encoder -> 3x4x4 detail feature
  -> resize only for scatter support
       P3: 3x8x8
       P4: 3x4x4
       P5: 3x2x2
  -> sparse placement at the candidate location using BCHW[y, x]
  -> concat([detail 3ch, global classification feature 64ch])
  -> head-specific Conv3x3(67->64) + BN + ReLU
  -> original classification predictor
  -> refined classification logits
```

The box-regression/DFL branches are unchanged. Overlapping Detail patches are
combined by in-place `add_()` sum. No fixed or learnable alpha is attached.

For the shared model, one YOLO forward captures all three activations, the
Detail encoder runs once for the selected crops, and three independent fusion
projections refine the three classification branches.

## 4. Corrected geometry and Grad-CAM behavior

- Candidate centers remain in `[x, y]`; tensor writes use BCHW `[y, x]`.
- Non-square CAMs are resized with the real `(input_height, input_width)`.
- Multi-head CAMs are normalized on their native grids, resized to the input,
  averaged, and normalized once more.
- `autograd.grad` requests gradients only for the hooked activations and does
  not erase or overwrite optimizer parameter gradients.
- Letterbox integer padding, four-tile Mosaic, affine transforms, HSV changes,
  and horizontal/vertical flips are propagated to source-image records.
- Border crops use their actual clipped crop center. Scatter clips at the
  feature-map boundary without shifting or stretching the patch.
- No-candidate, no-crop, small-box, padded-region and zero-CAM cases have
  explicit fallback paths.
- Training records CAM coverage separately from images that contain no
  candidate, avoiding a false zero-CAM diagnosis.

## 5. Required inputs

Fresh training starts from two independent pretrained checkpoints:

1. a YOLO checkpoint;
2. the pretrained Detail encoder checkpoint.

The historical Detail constructor is loaded from the tracked compatibility
archive `what is code.zip`. The dataset argument must point to a YOLO data YAML
containing the `train`, `val` and `test` splits.

Set local paths once before using the commands below:

```bash
export FINEDET_YOLO_WEIGHT=/absolute/path/to/yolo_best.pt
export FINEDET_DETAIL_WEIGHT=/absolute/path/to/detail_net_attn.pt
export FINEDET_DATA_YAML=/absolute/path/to/BrokenGlass.yaml
export FINEDET_ARCHIVE="$PWD/what is code.zip"
```

The weight files and dataset are intentionally not stored on this code branch.

## 6. Recommended independent single-head training

This command trains P3, P4 and P5 as three independent models. `--max-parallel 1`
is the safest single-GPU setting. Increase it only after checking GPU memory
and dataloader/CPU contention. After all training jobs finish, the supervisor
automatically evaluates each Val-selected `best.pt` on Test.

```bash
python -m experiments.launch_geometry_v5_heads \
  --heads high mid low \
  --max-parallel 1 \
  --epochs 50 \
  --seed 3187 \
  --tag seed3187 \
  --workers 8 \
  --threads 8 \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" \
  --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --archive "$FINEDET_ARCHIVE" \
  --data "$FINEDET_DATA_YAML"
```

To train only one independent head, change `--heads`:

```bash
# P3-only
python -m experiments.launch_geometry_v5_heads --heads high \
  --max-parallel 1 --epochs 50 --seed 3187 --tag p3_seed3187 \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --archive "$FINEDET_ARCHIVE" --data "$FINEDET_DATA_YAML"

# P4-only
python -m experiments.launch_geometry_v5_heads --heads mid \
  --max-parallel 1 --epochs 50 --seed 3187 --tag p4_seed3187 \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --archive "$FINEDET_ARCHIVE" --data "$FINEDET_DATA_YAML"

# P5-only
python -m experiments.launch_geometry_v5_heads --heads low \
  --max-parallel 1 --epochs 50 --seed 3187 --tag p5_seed3187 \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --archive "$FINEDET_ARCHIVE" --data "$FINEDET_DATA_YAML"
```

Mapping: `high=P3`, `mid=P4`, and `low=P5`.

## 7. Recommended shared P3+P4+P5 training

This command starts one model with one shared YOLO, one shared Detail encoder,
and three independent fusion projections. It automatically evaluates the
Val-selected joint `best.pt` on Test after 50 epochs.

```bash
python -m experiments.launch_input64_multi \
  --epochs 50 \
  --seed 3187 \
  --tag seed3187 \
  --workers 8 \
  --threads 8 \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" \
  --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --archive "$FINEDET_ARCHIVE" \
  --data "$FINEDET_DATA_YAML"
```

The equivalent low-level method flags are:

```text
--multi-head-direct
--multi-head-scale-aware
--multi-heads high mid low
--corrected-geometry
--geometry-input-support
--scatter-reduce sum
--candidate-conf 0.2
```

## 8. Direct low-level single-head command

The launchers above are recommended because they prevent accidental run-name
overwrites and perform the final Test only after training. For a manually
managed P4-only run, the underlying command is:

```bash
python -m experiments.run_318_fusion_alpha train \
  --corrected-direct-fusion \
  --corrected-geometry \
  --geometry-input-support \
  --scatter-reduce sum \
  --head-select mid \
  --candidate-conf 0.2 \
  --epochs 50 \
  --optimizer AdamW \
  --lr0 0.0001 \
  --lrf 0.01 \
  --momentum 0.937 \
  --weight-decay 0.0005 \
  --warmup-epochs 3 \
  --warmup-momentum 0.8 \
  --warmup-bias-lr 0 \
  --batch 32 \
  --imgsz 640 \
  --workers 8 \
  --device 0 \
  --seed 3187 \
  --save-period -1 \
  --wandb-mode disabled \
  --yolo-weight "$FINEDET_YOLO_WEIGHT" \
  --detail-weight "$FINEDET_DETAIL_WEIGHT" \
  --archive "$FINEDET_ARCHIVE" \
  --data "$FINEDET_DATA_YAML" \
  --name yolo_finedet_p4_seed3187
```

Use `--head-select high` or `--head-select low` for P3 or P5.

## 9. Evaluate a joint checkpoint

The following example evaluates a shared P3+P4+P5 checkpoint on Test with the
protocol used by the recorded experiment: FP32, batch 1, final confidence 0.5,
and NMS IoU 0.5.

```bash
python -m experiments.run_318_fusion_alpha val \
  --checkpoint /absolute/path/to/best.pt \
  --multi-head-direct \
  --multi-head-scale-aware \
  --multi-heads high mid low \
  --corrected-geometry \
  --geometry-input-support \
  --scatter-reduce sum \
  --candidate-conf 0.2 \
  --split test \
  --eval-conf 0.5 \
  --eval-iou 0.5 \
  --eval-batch 1 \
  --workers 4 \
  --device 0 \
  --seed 3187 \
  --wandb-mode disabled \
  --archive "$FINEDET_ARCHIVE" \
  --data "$FINEDET_DATA_YAML" \
  --name yolo_finedet_multi_test
```

For a single-head checkpoint, replace the multi-head flags with:

```text
--corrected-direct-fusion
--corrected-geometry
--geometry-input-support
--scatter-reduce sum
--head-select high|mid|low
```

## 10. Checkpoint selection and contents

- `best.pt` is selected only by validation mAP50.
- Test metrics are never used to choose a checkpoint.
- A shared run stores YOLO, the shared Detail encoder, three fusion
  projections and their BN state in one checkpoint.
- An independent run stores YOLO, Detail and one active head fusion projection
  in its own checkpoint.
- Geometry-v5 auxiliary Detail weights are saved from the same selected EMA as
  the model checkpoint, rather than from raw last-step training weights.

## 11. Verification commands

Run the CPU regression suite in the project environment:

```bash
MPLCONFIGDIR=/tmp/yolo_finedet_mpl \
python -m unittest \
  experiments.test_direct_geometry \
  experiments.test_multi_geometry \
  experiments.test_geometry_launcher \
  experiments.test_input64_multi_launcher -v
```

The current release passed 36 tests covering XY ordering, rectangular CAMs,
input64 support sizes, border clipping, Mosaic/affine/flip tracking, zero-CAM
accounting, gradient preservation, EMA serialization, independent launchers
and the shared multi-head launcher.

## 12. Files intentionally excluded

This code branch does not add trained `.pt` files, datasets, generated crops,
CAM arrays, visualizations, training logs, `experiments/runs`,
`experiments/analysis`, local split manifests, archives of old experiments or
temporary benchmark outputs. These are not required to review the method code.
