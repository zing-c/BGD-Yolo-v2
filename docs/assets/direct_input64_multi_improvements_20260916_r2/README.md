# P3+P4+P5 三头 r2 改善案例

| 图片 | 严格结论 | 前后对比 | CAM | Detail 位置 / 截图 |
|---|---|---|---|---|
| 00601 | 外墙错误框 0.552→区域最高 0.396，正确玻璃保留；TP/FP/FN 1/1/0→1/0/0 | [PNG](00601/multi/before_after_comparison.png) | [JPG](00601/multi/actual_cam_overlay.jpg) | [位置](00601/multi/actual_detail_source_windows.jpg) / [错误区域截图](00601/multi/false_positive_detail_crops.png) |
| 00890 | 正确目标 0.679→0.742；TP/FP/FN 保持 1/0/0 | [PNG](00890/multi/before_after_comparison.png) | [JPG](00890/multi/actual_cam_overlay.jpg) | [位置](00890/multi/actual_detail_source_windows.jpg) / [全部截图](00890/multi/detail_crops_64/) |

图片为无外部白边 / 标题 / 图例版本；只在图内保留检测框、置信度标签、GT 和截图窗口。CAM 数值、检测坐标和置信度未修改。融合前是同一三头 checkpoint 的第一次全局 forward，不是独立训练 YOLO baseline；完整筛选的局限见[结果页](../../DIRECT_INPUT64_MULTI_IMPROVEMENT_VISUALS_20260916.md)。
