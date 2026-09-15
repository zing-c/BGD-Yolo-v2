# 实验结果与方法文档索引

更新日期：2026-09-15。本索引位于 GitHub main 分支的 docs 目录。

| 文档 | 内容与使用范围 |
|---|---|
| [Corrected Direct 单头与三头：Test + 统一推理时间](CORRECTED_DIRECT_HEADS_TEST_TIMING_20260915.md) | 最新 P3/P4/P5 单头与三头结果；三轮统一延迟、GFLOPs 口径、Val 选权重、具体权重位置、P4 复测差异 |
| [Corrected 单头 Sum 原始结果](CORRECTED_SINGLE_HEAD_SUM_RESULTS.md) | 修正版单头的原始 Test 日志与训练细节；旧的单次时间不用于当前速度排名 |
| [Direct 单头与多头论文方法](DIRECT_SINGLE_MULTIHEAD_PAPER_METHOD.md) | 方法结构与实现说明；论文结果需结合最新修正页，不混入 legacy 数值 |
| [P4 Detail 与融合结构](P4_LEARNABLE_ALPHA_DETAIL_FUSION_ARCHITECTURE.md) | 早期可学习 alpha 变体的结构说明，不是当前无 alpha 实验 |
| [框架图可视化素材](P4_FRAMEWORK_VISUAL_ASSETS.md) | 可视化素材与使用说明 |

最新报告配套 [精确结果 JSON](results/corrected_direct_heads_test_timing_20260915.json) 和 [计时脚本快照](results/benchmark_corrected_direct_heads_20260915.py)。权重路径为训练机器上的位置，不应误认为本次已发布对应权重二进制。
