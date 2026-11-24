# Broken-GCW UAV200 evaluation

This comparison reports 16 models on the same 200-image evaluation subset. The subset contains 45 confirmed UAV images with 46 broken-glass boxes, plus 155 damage-free test images whose capture method is unknown. Of the confirmed UAV images, 17 came from the validation split and 28 from the test split.

- [Results for all 16 models](uav200-results.csv)
- [Exact 200-image selection manifest](uav200-manifest.csv)
- [Download the 45 confirmed UAV photographs](https://github.com/zing-c/BGD-Yolo-v2/releases/download/brokengcw-legacy-ap-20261124/uav200-confirmed-images.zip)

The manifest records evaluation order, source split, filenames, dimensions, and annotated box counts. The photographs in the ZIP follow the val and test directories recorded in the manifest. Four archived images have location metadata removed. Each was verified pixel for pixel against the frame used for evaluation. The two affected MPO files are represented by that evaluated frame. The manifest records SHA-256 digests for the test files and the corresponding published photos.

Precision and recall are computed from the fixed operating point with score greater than 0.50 and prediction-to-ground-truth matching IoU of 0.50. Detectors that use NMS use NMS IoU of 0.70. The AP metrics use the project's legacy Ultralytics v8.0.114 evaluator.
