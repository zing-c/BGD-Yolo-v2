# Broken-GCW test results with legacy Ultralytics AP

The table covers 15 detector checkpoints on the Broken-GCW test split (1,598 images and 234 annotated instances).

- [Results](results.csv) includes precision, recall, AP50, AP75, mAP50:95, parameter counts, forward GFLOPs, and per-image inference time.
- [Checkpoint manifest](checkpoints.csv) maps every row to its original weight asset and SHA-256 digest.
- [Release and weights](https://github.com/zing-c/BGD-Yolo-v2/releases/tag/brokengcw-legacy-ap-20261124) contains the checkpoints and a downloadable copy of the results.

## Evaluation protocol

Predictions with scores greater than 0.50 were retained. NMS-based detectors used an NMS IoU of 0.70. YOLOv10n, YOLOv26n, D-FINE-N, and RT-DETRv3-R18 used native end-to-end prediction selection.

Precision and recall were computed directly from TP, FP, and FN at a prediction-to-ground-truth matching IoU of 0.50. AP50, AP75, and mAP50:95 use the Ultralytics v8.0.114 legacy precision–recall interpolation and trapezoidal integration. The IoU grid for mAP50:95 contains ten thresholds with increments of 0.05.

GFLOPs estimate supported forward operations at a reference resolution of 640 × 640. The BG-O2D estimates exclude Grad-CAM backpropagation. Inference timing uses the matching checkpoints from the batch-1, two-thread prefetch benchmark. Model-specific timing boundaries are in the results CSV.

YOLOv8n reproduces AP75 of 89.77%. BG-O2D with P3, P4, and P5 reaches AP75 of 92.36% and mAP50:95 of 89.24%.
