# 实验结果与方法文档索引

更新日期：2026-09-16。本索引位于 GitHub main 分支的 docs 目录。

| 文档 | 内容与使用范围 |
|---|---|
| [破损区域框 / 整图面积比例分布](BROKEN_REGION_RATIO_DISTRIBUTION.md) | Train/Val/Test 共 2,118 个 GT 框，0.1 间隔的 10 个等宽 bins；柱间无缝、横轴两端留白、无图例；PNG/SVG/PDF、CSV、JSON 与可复现脚本 |
| [当前正确版本：P3/P4/P5 单头与 P3+P4+P5 三头 r2 总表](DIRECT_INPUT64_CURRENT_SINGLE_MULTI_RESULTS_20260916.md) | 四组 50 epoch 的 Val-best / Test、参数量、模块 GFLOPs、延迟口径和四个可下载 best.pt；三头不自动优于最佳单头 |
| [Test 00756：P3+P4+P5 三头完整可视化](DIRECT_INPUT64_MULTI_00756_VISUALIZATION_20260916.md) | 三头 r2 的真实聚合 Grad-CAM、Detail 64×64 输入、conf>0.2 候选、融合前后检测与逐头 CAM 数值；无外部白边和标题 |
| [三头 r2 改善可视化：00601 / 00890](DIRECT_INPUT64_MULTI_IMPROVEMENT_VISUALS_20260916.md) | 完整 Test 严格筛选；00601 外墙误检 0.552→区域最高 0.396 且正确玻璃保留，00890 正确框 0.679→0.742；真实 CAM / Detail 截图、无外部白边；同时披露整体工作点退化 |
| [三头 r2 完成记录](DIRECT_INPUT64_MULTIHEAD_RERUN_20260916.md) | 第 40 轮 Val-best、完整 Test、r1/r2 tensor 级确定性核验、配置与联合 best.pt 下载位置 |
| [当前图片统一无外白边 / 文字版](DIRECT_INPUT64_BORDERLESS_VISUALS_20260916.md) | 原六图、选定四图、Test 00751、Val 三图均替换显示图片；无外部标题、白边、图例、侧栏和拼图间隔；保留图内红框标签，原始数据 / 截图不变 |
| [Val 新例子：00770 / 00657 / 02211 的误检抑制](DIRECT_INPUT64_VAL_FP_SUPPRESSION_20260916.md) | 每个当前单头扫描完整 503 张 Val；P5 两个保留正确检测的错误位置正例、一个外拍窗户背景负例；含清晰红框、真实 CAM 和全部截图；同时说明 00601/P3 误检并未被消除 |
| [00751 / P5：误检框在 conf=0.5 下消失的真实例子](DIRECT_INPUT64_FP_SUPPRESSION_SEARCH_20260916.md) | 每个当前单头扫描完整 1598 张 Test；P5 迷彩衣服错误框 0.647→0.234、正确检测保留；含逐 anchor 核验、真实 CAM 与截图，车内场景，非独立 YOLO 基线对照 |
| [00039 / 00586 / 00601 / 01979：清晰排版可视化](assets/direct_input64_head_visuals_20260916_pretty/README.md) | 四张选定图片 × 当前三个独立单头；粗红框、贴框的 broken glass + conf 红底白字标签、无损 PNG 流程图、全部 134 张原生 Detail 截图；只调整样式，原始 CAM / 检测不变，旧版保留 |
| [当前 P3/P4/P5：Grad-CAM、Detail 截图与最终检测可视化](DIRECT_INPUT64_HEADS_VISUALS_20260916.md) | 六张 Test 图片 × 三个当前单头 best，真实热力图、全部 64×64 截图、conf>0.2 候选框、粗红最终框、同区域纹理对照；图片在 main/docs/assets/direct_input64_head_visuals_20260916_r1/ |
| [本轮 Direct input64 修正版 P3/P4/P5：完整 Test、时间、参数量与 GFLOPs](DIRECT_INPUT64_HEADS_TEST_RESULTS_20260916.md) | 2026-09-16 完成的三个独立单头，各 50 epoch；含 CAM H/W、增强来源、截图中心与累积梯度修正；单次完整 Test 时间、实际权重参数与 forward GFLOPs、具体权重位置、CSV 和运行版本 hash |
| [此前 xy/add 修正版单头与三头：Test + 统一推理时间](CORRECTED_DIRECT_HEADS_TEST_TIMING_20260915.md) | 旧运行版本；三轮统一延迟、GFLOPs 口径、Val 选权重、具体权重位置、P4 复测差异，不与本轮 input64 结果混用 |
| [Corrected 单头 Sum 原始结果](CORRECTED_SINGLE_HEAD_SUM_RESULTS.md) | 修正版单头的原始 Test 日志与训练细节；旧的单次时间不用于当前速度排名 |
| [Direct 单头与多头论文方法](DIRECT_SINGLE_MULTIHEAD_PAPER_METHOD.md) | 方法结构与实现说明；论文结果需结合最新修正页，不混入 legacy 数值 |
| [P4 Detail 与融合结构](P4_LEARNABLE_ALPHA_DETAIL_FUSION_ARCHITECTURE.md) | 早期可学习 alpha 变体的结构说明，不是当前无 alpha 实验 |
| [框架图可视化素材](P4_FRAMEWORK_VISUAL_ASSETS.md) | 可视化素材与使用说明 |

本轮报告配套 [单头精确结果 JSON](results/direct_input64_heads_test_20260916.json)、[三头 r2 status](results/direct_input64_multi_r2_status_20260916.json)、[当前四模型三轮计时 JSON](results/direct_input64_v5_controlled_timing_20260916.json)、四份训练 CSV与[完整三头可视化筛选 JSON](results/direct_input64_multi_improvement_search_20260916.json)。P3/P4/P5 单头和三头 r2 的四个关键 best.pt 已发布到 [docs/weights/direct_input64_20260916](weights/direct_input64_20260916/)；只上传 best，不上传重复 last。完整训练运行源码快照保留在训练机本地。旧三轮报告配套 [旧结果 JSON](results/corrected_direct_heads_test_timing_20260915.json) 和 [计时脚本快照](results/benchmark_corrected_direct_heads_20260915.py)。
