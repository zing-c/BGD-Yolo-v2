# 00039 / 00586 / 00601 / 01979：无外部白边版可视化

按用户选定的四张 Test 图片重新排版，使用当前完成 50 epoch 的 P3、P4、P5 三个独立单头 Val best.pt 的**已保存结果**。这次只重绘显示样式，没有重新推理、调整方法、更新权重或修改 CAM。[无外白边素材及布局说明](../../DIRECT_INPUT64_BORDERLESS_VISUALS_20260916.md)：图片外部不加白边、文字或间隔，说明全部放在 MD。

图片位于 GitHub **main / docs / assets / direct_input64_head_visuals_20260916_pretty**。[完整实验与可视化说明](../../DIRECT_INPUT64_HEADS_VISUALS_20260916.md) · [旧版六张图片](../direct_input64_head_visuals_20260916_r1/README.md)

另找到：[00751 / P5 真实误检消除例子](../direct_input64_fp_suppression_20260916_r1/README.md)。迷彩衣服错误框 0.647→0.234，最终 conf=0.5 时消失，正确玻璃保留；该图是车内场景，不是外拍建筑。[完整筛查证据](../../DIRECT_INPUT64_FP_SUPPRESSION_SEARCH_20260916.md)

## 样式

- 检测框统一粗红线，不加白轮廓；小截图窗口控制线宽，不用加粗填满截图区域。
- 检测标签为常见的 **`broken glass: 0.979` 红底白字**，紧贴框边且不超出图片；顶边标签重叠时改在框内另一边缘，不改实际框。
- 去掉图片外的顶部标题、底部说明、图例、白边及拼图间隔；检测区域与截图区域内部不填色，保留真实纹理和热力图。标签中的 confidence 显示三位小数。
- 全部 conf>0.2 pre-NMS 候选框仍完整画出，不加图片外侧栏；每个候选精确分数保存在 metadata。
- 先缩放干净图像再画框和文字，避免原图尺寸较大时文字被一起缩小。
- 比较图及完整流程图同时提供高清无损 PNG 与浏览用 JPG。PNG 写入 300 dpi，实际像素尺寸仍以文件为准；不存在凭空增加原图细节。
- 原 CAM 为零时显示干净原照片，零值 / 框中心回退在本 MD、metadata 和渲染 JSON 中说明，不加图中文字或人为热点。

## 新增 00039：远处破损玻璃

![00039 清晰排版](00039/comparison.jpg)

[高清无损 PNG](00039/comparison.png) · [P4 六联流程 JPG](00039/p4/paper_overview.jpg) · [P4 六联流程 PNG](00039/p4/paper_overview.png) · [P4 全部 conf>0.2 候选框](00039/p4/all_candidates_conf_gt02.jpg) · [P4 全部实际 64×64 截图](00039/p4/detail_crops_64/)

P4 有效 CAM 的最强点位于标注内，正确框置信度 0.744→0.810。三个头均为 TP=1、FP=0、FN=0，融合前后计数相同；P5 的 CAM 为零。

## 新增 00586：外拍建筑窗户

![00586 清晰排版](00586/comparison.jpg)

[高清无损 PNG](00586/comparison.png) · [P4 六联流程 JPG](00586/p4/paper_overview.jpg) · [P4 六联流程 PNG](00586/p4/paper_overview.png) · [P4 全部 conf>0.2 候选框](00586/p4/all_candidates_conf_gt02.jpg) · [P4 全部实际 64×64 截图](00586/p4/detail_crops_64/)

三个头的 CAM 均有值；P4/P5 最强点在标注内，P3 最强点在标注外。P4 正确框置信度 0.916→0.958；P5 实际为 0.975→0.929。三个头均为 TP=1、FP=0、FN=0，不能把这些置信度变化说成消除了误检。

## 00601：三个单头比较

行：P3、P4、P5。列：融合前全局检测、真实 CAM、实际 Detail 截图位置、融合后检测。

![00601 清晰排版](00601/comparison.jpg)

[高清无损 PNG](00601/comparison.png) · [P4 六联流程 JPG](00601/p4/paper_overview.jpg) · [P4 六联流程 PNG](00601/p4/paper_overview.png) · [P4 全部 conf>0.2 候选框](00601/p4/all_candidates_conf_gt02.jpg) · [P4 全部实际 64×64 截图](00601/p4/detail_crops_64/)

建议框架图使用 **00601 / P4**：CAM 有真实有效响应，正确框置信度从 0.933 变为 0.979。前后 TP/FP/FN 相同，这不等于消除误检。P3 的额外错误位置框仍保留，P3/P5 的 CAM 为零。

## 01979：三个单头比较

![01979 清晰排版](01979/comparison.jpg)

[高清无损 PNG](01979/comparison.png) · [P5 六联流程 JPG](01979/p5/paper_overview.jpg) · [P5 六联流程 PNG](01979/p5/paper_overview.png) · [P5 全部 conf>0.2 候选框](01979/p5/all_candidates_conf_gt02.jpg) · [P5 全部实际 64×64 截图](01979/p5/detail_crops_64/)

建议框架图使用 **01979 / P5** 展示有效 CAM 和实际截图位置。P3/P4 的 CAM 为零。P5 正确框置信度实际从 0.995 变为 0.958；TP/FP/FN 相同，不能称为检测性能提升。

## 可用素材与真实性校验

每张图片的 `p3/`、`p4/`、`p5/` 都保留融合前后框、CAM 叠加图、灰度 CAM、原图截图范围、截图拼图、六联流程、全部 conf>0.2 原始 pre-NMS 候选框及 metadata。候选框与最终框口径不同：候选为 score>0.2、未做 NMS；最终框为 conf=0.5、NMS IoU=0.5。

`detail_vs_global_same_region.jpg` 只裁掉旧版对照图的外部标题 / 白边，保留两幅实际图像区域。`detail_crops_64/` 保存全部实际截图，包括重复输入；00039 共 30 张、00586 共 31 张、00601 共 44 张、01979 共 29 张，总计 134 张，PNG 文件与原导出字节完全一致。

十二组 `metadata.json` 也与原导出字节完全一致，保存真实原图坐标和未舍入置信度。显示图中的标签保留三位小数，坐标按面板尺寸缩放；这不修改原始数据。记录见 [render_manifest.json](render_manifest.json)。CAM 读取本地原始浮点数组后按原叠加公式绘制，不增强数值、移动热点或改变归一化。

融合前是**同一个联合 checkpoint 的第一次全局 forward**，不是独立训练的 YOLO 基线。这四张图片所有单头前后的 TP/FP/FN 均没有变化；用于方法过程的真实展示，不作为纠正误检 / 补回漏检的证据。另行完成的全 Test 误检筛查已找到上述 00751 / P5 正例，并区分置信度压低到 0.5 以下与 NMS 去框，不通过修改阈值制造正例。

本次发布仅含图片、MD 和 JSON；原始 NPZ、训练源码、渲染代码和权重文件不随这些图片上传。当前清晰版和 r1 显示图片均为无外白边版，原始数值 / 原生截图不变。[此前带标题的版式](https://github.com/zing-c/BGD-Yolo-v2/tree/ca606ec/docs/assets/direct_input64_head_visuals_20260916_pretty)和[更早白底检测标签版](https://github.com/zing-c/BGD-Yolo-v2/tree/e8319e7/docs/assets/direct_input64_head_visuals_20260916_pretty)仍可从 GitHub 历史版本查看。六联流程当前为横向无缝六图：GT、融合前、CAM、Detail 位置、融合后、第一张实际输入。
