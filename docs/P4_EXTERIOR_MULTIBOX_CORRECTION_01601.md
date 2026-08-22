# 室外多位置框修正案例：01601

该样本为室外拍摄的天窗/玻璃结构，标注中包含两个相互独立的破损玻璃位置，不是同一块玻璃的重复框。

| 方法 | TP | FP | FN | 结果 |
|---|---:|---:|---:|---|
| best318.pt | 1 | 0 | 1 | 只检测到左侧位置 |
| P4 Grad-CAM + Detail Fusion | 2 | 0 | 0 | 找回并正确定位两个位置 |

可视化中：黄色为 GT，绿色为预测框。候选阈值为 `conf > 0.2`，Grad-CAM 非零且最大值约 `0.9992`。

- [基线与融合结果对比](assets/p4_exterior_multibox_01601/12_baseline_vs_ours_comparison.jpg)
- [全部候选框](assets/p4_exterior_multibox_01601/03_all_candidates_conf_gt_0_2.jpg)
- [Grad-CAM](assets/p4_exterior_multibox_01601/04_p4_gradcam_all_candidates.jpg)
- [候选裁剪预览](assets/p4_exterior_multibox_01601/candidate_crops_preview/)

