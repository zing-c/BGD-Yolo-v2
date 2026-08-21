"""Primary entry point for the complete repaired ZIP BGD-YOLO method.

The previous current-method runner remains available at
``experiments/run_gradcam_global_local.py`` for archived comparisons.
"""

import sys

from experiments.run_zip_bgd import main


if __name__ == "__main__":
    # Keep ``python train.py`` and ``python train.py --epochs ...`` convenient.
    # An explicit mode (train/val) is also passed through.
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "train")
    main()
