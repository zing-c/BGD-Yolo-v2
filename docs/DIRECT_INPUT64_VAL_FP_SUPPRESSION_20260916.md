# 当前单头：Val 新找到的错误位置误检抑制例子

日期：2026-09-16。使用[当前已完成 50 epoch 的 P3/P4/P5 单头](DIRECT_INPUT64_HEADS_TEST_RESULTS_20260916.md)各自固定的 Val best.pt，三个头分别扫描完整 **503 张 Val 图片，其中 96 张为无目标背景**。没有训练、修改权重或调阈值；不是正在训练的三头模型。

新找到 **P5 的两个“保留正确检测、去掉非破损位置误检”正例，以及一个无破损标注的背景误检抑制例子**。所有三张的 P5 Grad-CAM 均非零且数值有限；不能由此推断其他图片或其他头都具有有效热点。

## 先回答 00601：有误检，但没有像 00751 那样被消除

00601 来自 Test，不能混入下面的 Val 统计。当前 P3 在右下方非破损位置有一个 **0.557723** 的错误框，融合后仍为 **0.557723**，最终 conf=0.5 仍能检测到；TP/FP/FN 为 **1/1/0 → 1/1/0**。本图 P3 CAM 全零，使用框中心回退。

当前 P4、P5 在 00601 的融合前就只有正确框，前后均为 1/0/0；不能把不同头的结果当成“同一模型融合消除误检”的证据。00601 不作为误检改善正例。

[00601 三个当前单头比较 PNG](assets/direct_input64_head_visuals_20260916_pretty/00601/comparison.png) · [P3 原始 metadata](assets/direct_input64_head_visuals_20260916_pretty/00601/p3/metadata.json) · [此前 00751/P5 的真实抑制正例](DIRECT_INPUT64_FP_SUPPRESSION_SEARCH_20260916.md)

## 新增三张 Val 图片

置信度“前→后”指**同一个原始 decoded anchor** 的分数，不是从 NMS 后框中挑选一个邻近框。TP/FP/FN 使用 conf=0.5、NMS IoU=0.5、与标注 IoU≥0.5 的一对一匹配。

| Val 图片 / 当前单头 | 非破损位置误检 | 原误检 anchor 分数：前→后 | 融合后同一错误区域所有 raw anchor 最高分¹ | TP/FP/FN：前→后 |
|---|---|---:|---:|---|
| 00770 / P5 | 建筑窗户左侧狭窄墙边 / 窗框边缘 | 0.578938 → 0.328489 | 0.328489 | 1/1/0 → 1/0/0 |
| 00657 / P5 | 玻璃顶旁右侧带孔结构面板 | 0.672253 → 0.269189 | 0.363850 | 1/1/0 → 1/0/0 |
| 02211 / P5 | 外拍反光窗户 / 百叶帘区域，无破损 GT | 0.590294 → 0.303861 | 0.409391 | 0/1/0 → 0/0/0 |

¹ 区域定义为与原错误框 IoU≥0.5 的全部原始 decoded anchor。三个原错误框与任意 GT 的最大 IoU 都为 0；02211 没有 GT，单独标明为背景负例。区域最高分也低于 0.5，且融合后最终输出没有与旧错误框 IoU≥0.5 的框，因此不是仅由 NMS 挤掉原错误框。

### 00770：保留破损窗户，消除左侧错误位置框

![00770 P5 误检抑制](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/false_positive_suppression.jpg)

正确玻璃仍被检测到：其 NMS 后框置信度为 **0.996298 → 0.977489**，不是提升置信度的例子，而是去掉额外误检的例子。此图为建筑窗户；不强行将拍摄方向标成确定的外拍。

[无损对比 PNG](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/false_positive_suppression.png) · [完整流程 PNG](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/paper_overview.png) · [真实热力图](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/actual_cam_overlay.jpg) · [误检候选实际 Detail 输入](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/false_positive_detail_crops.jpg) · [全部 conf>0.2 框](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/all_candidates_conf_gt02.jpg) · [逐框核验 JSON](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/false_positive_audit.json)

### 00657：保留玻璃顶破损，消除旁边带孔面板误检

![00657 P5 误检抑制](assets/direct_input64_val_fp_suppression_20260916_r1/00657/p5/false_positive_suppression.jpg)

这张是向上拍摄的玻璃顶 / 天窗，不是外拍建筑立面。错误框覆盖右侧带孔结构面板，不是同一块破损玻璃的多个重叠框。

[无损对比 PNG](assets/direct_input64_val_fp_suppression_20260916_r1/00657/p5/false_positive_suppression.png) · [完整流程 PNG](assets/direct_input64_val_fp_suppression_20260916_r1/00657/p5/paper_overview.png) · [真实热力图](assets/direct_input64_val_fp_suppression_20260916_r1/00657/p5/actual_cam_overlay.jpg) · [误检候选实际 Detail 输入](assets/direct_input64_val_fp_suppression_20260916_r1/00657/p5/false_positive_detail_crops.jpg) · [全部 conf>0.2 框](assets/direct_input64_val_fp_suppression_20260916_r1/00657/p5/all_candidates_conf_gt02.jpg) · [逐框核验 JSON](assets/direct_input64_val_fp_suppression_20260916_r1/00657/p5/false_positive_audit.json)

### 02211：外拍反光窗户背景，原误检不再输出

![02211 P5 背景误检抑制](assets/direct_input64_val_fp_suppression_20260916_r1/02211/p5/false_positive_suppression.jpg)

从外部拍摄窗户，可见树木 / 建筑反光及窗帘。**没有破损玻璃标注**，融合后输出为空；不能描述为“同时保留正确破损玻璃框”，也不能把该背景例子与上面两个正目标例子混为一类。图中窗口是否存在细微破损仍以数据标注为依据，这里不额外创造标注。

[无损对比 PNG](assets/direct_input64_val_fp_suppression_20260916_r1/02211/p5/false_positive_suppression.png) · [完整流程 PNG](assets/direct_input64_val_fp_suppression_20260916_r1/02211/p5/paper_overview.png) · [真实热力图](assets/direct_input64_val_fp_suppression_20260916_r1/02211/p5/actual_cam_overlay.jpg) · [误检候选实际 Detail 输入](assets/direct_input64_val_fp_suppression_20260916_r1/02211/p5/false_positive_detail_crops.jpg) · [全部 conf>0.2 框](assets/direct_input64_val_fp_suppression_20260916_r1/02211/p5/all_candidates_conf_gt02.jpg) · [逐框核验 JSON](assets/direct_input64_val_fp_suppression_20260916_r1/02211/p5/false_positive_audit.json)

## 扫描范围与结论边界

| 已完成单头 | 扫描 Val 图片 | 融合前有 FP 的图片 | 融合前 FP 框 | 原 FP / 同区域 raw 最高分均跌破 0.5 的框 | 本页严格错误位置例子 |
|---|---:|---:|---:|---:|---|
| P3 | 503 | 13 | 18 | 0 | 0 |
| P4 | 503 | 11 | 14 | 0 | 0 |
| P5 | 503 | 16 | 21 | 6 | 2 个保留正确检测的正例 + 1 个背景负例 |

其余三个 P5 抑制框不是本页所要求的干净正例：01841 原框与 GT 最大 IoU 约 0.494，正确目标前后都漏检；01728 原框最大 GT IoU 约 0.298，仍有另一个 FP；01257 原框最大 GT IoU 约 0.354，正确目标仍漏检。它们不作为“非破损位置误检完全修正且保留正确检测”的素材。

固定 FP32、imgsz=640、实际 rectangular letterbox、batch=1、candidate>0.2、最终 conf=0.5、NMS IoU=0.5、无 alpha、重叠 sum。每张原本有 FP 的图才执行完整 CAM/Detail；因此**这不是完整融合模型的 Val 指标复测**，不统计原本无 FP 图片是否新产生 FP，不能将所选正例外推为整体误检全部消失。完整扫描耗时 53.15 秒，不是 inference benchmark。

**融合前为同一联合训练 checkpoint 的第一次全局 forward，不是独立训练的 YOLO / 318.pt。**三个头分别使用各自 checkpoint，不能跨头对比来证明融合改进。best.pt 在训练时按 Val 选定；本页只对已经固定的权重进行定性筛图，不重新选择 best、不用于超参数调优，也不是独立 Test 证据。

采用单头训练时 SHA256 匹配的本地运行版本进行筛查，随后在独立进程重新导出三张图、所有三个头；P5 的前后逐框坐标、置信度和计数均与筛查结果相符。实际 raw 回归坐标未变化，但分类分数变化会令 NMS 选择不同 anchor，所以最终保留框坐标可能略有变化。红底白字标签、粗红框只用于显示，不改框、分数或 CAM；检测区域内部不填色。

素材首页：[三个 Val 例子及全部实际 64×64 截图](assets/direct_input64_val_fp_suppression_20260916_r1/README.md)。每张图还保留 P3/P4/P5 比较：P3 在 00657/00770 的 CAM 为零，P4 在 02211 的误检仍存在，均如实展示。热点反映本模型实际响应，不保证最强点一定在正确玻璃上；不能把误检区域的热点解释成正确定位。

完整筛查证据：[Val JSON](results/direct_input64_val_fp_suppression_search_20260916.json)；实际检测、候选框、截图范围与 CAM 数值见各头 `metadata.json`，渲染保真核验见 [render_manifest.json](assets/direct_input64_val_fp_suppression_20260916_r1/render_manifest.json)。原始 NPZ 数组、运行源码和权重未随本次图片发布，仍保留在训练机本地。
