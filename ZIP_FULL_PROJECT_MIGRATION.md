# Complete ZIP-method migration

The active project method is now the repaired ZIP revision. The pre-migration
code snapshot is `archives/project_method_before_full_zip_migration_20260816.tar.gz`;
the former current method remains available through
`experiments/run_gradcam_global_local.py`.

## Active execution chain

`train.py` now launches `experiments/run_zip_bgd.py`, which builds
`ZipBGDYOLO` from `zip_bgd_fusion.py`.

1. Load the original train10 YOLO `best.pt` and the compressed Detail
   classifier checkpoint.
2. Keep every decoded candidate whose internal confidence is at least `0.1`.
   No candidate NMS and no top-k limit are applied.
3. Capture the second classification convolution at the selected Detect head
   (`low`/P5 by default).
4. Compute classic Grad-CAM using the sum of all class scores. LayerCAM is not
   used. A flat CAM falls back to the candidate box center.
5. Remove duplicate crop centers per image and crop fixed 64x64 RGB patches
   from the original image with ImageNet normalization.
6. The repaired compressed Detail model produces one 3x4x4 feature per crop.
7. Add overlapping Detail maps at fixed 4x4 positions, concatenate them with
   the 64-channel classification feature, and apply Conv(67,64,3)+BN+ReLU.
8. Recompute only the selected head's class logits. Box regression and the
   other detection heads remain unchanged.
9. Fine-tune the entire effective network; only DFL's fixed expectation vector
   remains non-trainable.

The migration fixes ZIP-source implementation hazards without changing the
method definition: x/y coordinates are no longer swapped, boundary placements
are always exactly 4x4, flat CAM does not select an arbitrary top-left window,
and custom modules/configuration survive EMA checkpoint save and reload.

## Commands

Train for the requested 60 epochs:

```bash
python train.py --epochs 60 --candidate-conf 0.1 --head-select low
```

Test with the agreed reporting thresholds (these do not affect training-time
candidate generation or validation checkpoint selection):

```bash
python train.py val --yolo-weight PATH/TO/best.pt --split test --eval-conf 0.5 --eval-iou 0.7
```
