# 室外多位置误检修正案例：00784

该样本包含三个相互独立的玻璃破损位置。红色为预测框，黄色为 GT。

| 方法 | TP | FP | FN | IoU |
|---|---:|---:|---:|---:|
| best318.pt | 3 | 1 | 0 | 0.964 |
| P4 Grad-CAM + Detail Fusion | 3 | 0 | 0 | 0.980 |

融合方法移除了 YOLO 在错误位置产生的额外框，同时保留三个正确位置。候选阈值为 `conf > 0.2`，Grad-CAM 最大值约 `0.9999`。

[红框对比图](assets/p4_exterior_multibox_00784/12_baseline_vs_ours_comparison.jpg) · [全部候选框](assets/p4_exterior_multibox_00784/03_all_candidates_conf_gt_0_2.jpg) · [Grad-CAM](assets/p4_exterior_multibox_00784/04_p4_gradcam_all_candidates.jpg)
