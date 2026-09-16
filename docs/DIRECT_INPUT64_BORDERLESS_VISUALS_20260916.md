# 当前单头可视化：无外部白边与说明文字

2026-09-16，按用户要求更新当前上传版。**单张图片不增加任何外部白边、标题、指标文字、图例或候选分数侧栏；对比 / 流程 / 截图拼图没有白色间隔和空白占位格。**图内仅保留预测框及 `broken glass: 0.979` 红底白字标签，标注框为绿色，实际 Detail 窗口为红色。靠近顶边的重叠标签可放在框内另一边缘，避免互相遮挡，不移动实际框。

本次直接替换以下四个已发布目录的显示图片，原来的 GitHub 链接仍有效；有白边 / 标题的旧版可从 [ca606ec 历史提交](https://github.com/zing-c/BGD-Yolo-v2/tree/ca606ec/docs/assets)查看。**检测数值、模型权重、浮点 CAM 数组、metadata、原生 64×64 PNG 截图均不改变。**仅 CPU 重绘，不训练、不重新推理。

| 素材组 | 已更新目录 | 范围 |
|---|---|---|
| 三头 r2 新案例 | [00601 / 00890 图片首页](assets/direct_input64_multi_improvements_20260916_r2/README.md) | 00601 消除外墙误检并保留正确玻璃；00890 正确框 0.679→0.742；同样无外白边 / 标题 / 图例 |
| 原六张 Test 示例 | [图片首页](assets/direct_input64_head_visuals_20260916_r1/README.md) | 00587 / 00586 / 00039 / 00601 / 01979 / 00970，三个当前独立单头 |
| 用户选定四张 | [图片首页](assets/direct_input64_head_visuals_20260916_pretty/README.md) | 00039 / 00586 / 00601 / 01979，三个当前独立单头 |
| Test 误检抑制 | [00751 首页](assets/direct_input64_fp_suppression_20260916_r1/README.md) | P5 迷彩衣服错误框 0.647→0.234，正确目标保留；其他头真实结果也保留 |
| Val 误检抑制 | [Val 图片首页](assets/direct_input64_val_fp_suppression_20260916_r1/README.md) | 00770 / 00657 的 P5 保留正确检测；02211 为无破损 GT 的背景例子 |

## 直接使用的无白边素材

00770 / P5，左为融合前，右为融合后：

![00770 无外部白边对比](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/false_positive_suppression.jpg)

[对比 PNG](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/false_positive_suppression.png) · [单张融合前](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/global_before_fusion.jpg) · [单张融合后](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/final_after_fusion.jpg) · [真实热力图](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/actual_cam_overlay.jpg) · [Detail 实际截图位置](assets/direct_input64_val_fp_suppression_20260916_r1/00770/p5/actual_detail_source_windows.jpg)

| 其他示例 | 无白边对比 / 流程 PNG |
|---|---|
| 00657 / P5 | [前后对比](assets/direct_input64_val_fp_suppression_20260916_r1/00657/p5/false_positive_suppression.png) · [误检候选实际截图拼图](assets/direct_input64_val_fp_suppression_20260916_r1/00657/p5/false_positive_detail_crops.png) |
| 02211 / P5 | [背景误检前后对比](assets/direct_input64_val_fp_suppression_20260916_r1/02211/p5/false_positive_suppression.png) |
| 00751 / P5 | [前后对比](assets/direct_input64_fp_suppression_20260916_r1/00751/p5/false_positive_suppression.png) · [实际误检候选截图](assets/direct_input64_fp_suppression_20260916_r1/00751/p5/false_positive_detail_crops.png) |
| 00601 / P4 | [完整流程](assets/direct_input64_head_visuals_20260916_pretty/00601/p4/paper_overview.png) |
| 01979 / P5 | [完整流程](assets/direct_input64_head_visuals_20260916_pretty/01979/p5/paper_overview.png) |
| 00039 / P4 | [完整流程](assets/direct_input64_head_visuals_20260916_pretty/00039/p4/paper_overview.png) |
| 00586 / P4 | [完整流程](assets/direct_input64_head_visuals_20260916_pretty/00586/p4/paper_overview.png) |

每个头 `paper_overview` 现在是六幅图紧邻的横条，顺序：**GT → 融合前 → 实际 CAM → Detail 截图位置 → 融合后 → 第一张实际 Detail 输入**。照片保留完整视野与宽高比；最后一张实际 64×64 截图按正方形 nearest-neighbor 放大，不拉伸成照片比例。

每张 `comparison` 的三行依次为 P3 / P4 / P5，四列依次为融合前 / CAM / Detail 位置 / 融合后。没有外部图例，布局说明在本 MD。截图拼图只排列实际截图，以整行排满的列数展示，不用空白格补齐；全部重复输入仍保存在 `detail_crops_64/`。

## 数值和零值 CAM 的说明保留在文档

候选图保留所有 pre-NMS conf>0.2 的红框，不在外部列分数；精确分数和原图坐标都在该组 `metadata.json`。最终检测图仍为 conf=0.5、NMS IoU=0.5。所有前后对比来自同一联合训练 checkpoint，不是独立 YOLO / 318.pt 基线。

零值 CAM 图现在显示干净原照片，不在照片上写 ZERO CAM，也不伪造热力图：该组 metadata 的 `cam.max=0` 和 `render_manifest.json` 的 `cam_is_zero=true` 明确保留。实际灰度 CAM PNG 及浮点 NPZ 不重写，不裁掉它们代表真实模型输入的区域。已知零值与限制见[六图可视化说明](DIRECT_INPUT64_HEADS_VISUALS_20260916.md)、[Test 00751 说明](DIRECT_INPUT64_FP_SUPPRESSION_SEARCH_20260916.md)、[Val 说明](DIRECT_INPUT64_VAL_FP_SUPPRESSION_20260916.md)。00601/P3 的 0.558 误检仍未消除。

同区域纹理对照从原来保存的 1000×460 JPEG 中，准确取出两幅真实图像的已知 390×390 区域，横向无缝拼接；只去除外部白边与标题，没有猜测图像内容、重造压缩图或改模型输入。两幅图左为实际原图 Detail，右为实际 YOLO 输入对应区域。原生截图文件仍为最可靠的像素来源。

每个目录的 `render_manifest.json` 记录零外边距、零间隔、未改 CAM / 坐标 / 分数、各组 metadata SHA256、原始 CAM NPZ SHA256、实际截图计数、显示文件像素尺寸及 SHA256。此次 GitHub 上传只包含显示图片、说明 MD 和渲染核验 JSON，不上传训练源码、NPZ 或权重二进制。
