# Grad-CAM Detect Head 实验结果

更新时间：2026-08-22

本文档集中记录 `best318` 框架下四个单 Detect 头实验、一个 P3+P4+P5 多头实验及其复跑，包括训练参数、验证指标、统一 test 指标、推理时间、权重下载位置和文件校验值。

## 实验设置

所有实验均从 `best318.pt` 之前的独立 YOLO 与 Detail 预训练权重初始化，不使用 `best318.pt` 初始化。

共同训练参数：

- epochs：50
- imgsz：640
- optimizer：AdamW
- lr0：0.0001
- lrf：0.01
- momentum：0.937
- weight decay：0.0005
- warmup epochs：3
- warmup momentum：0.8
- warmup bias lr：0.0
- batch：32
- AMP：开启
- candidate confidence：0.2
- patience：50

可学习 alpha 参数加入优化器，但单独放在无 weight decay 参数组中；其余训练参数不变。统一 test 表中的 `conf=0.5` 是最终检测评估阈值，候选区域阈值仍为 `candidate confidence=0.2`。

实验结构：

| 实验 | Grad-CAM/Detect 位置 | 融合方式 | W&B |
| --- | --- | --- | --- |
| 低层头 P5 | `Detect.cv3[2][1]` | `base + 0.5 × (fused - base)` | [1ghxfq1l](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/1ghxfq1l) |
| 中层头 P4 | `Detect.cv3[1][1]` | `base + 0.5 × (fused - base)` | [a5ucwbtv](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/a5ucwbtv) |
| 中层头 P4，可学习 alpha | `Detect.cv3[1][1]` | `base + sigmoid(alpha_logit) × (fused - base)`，alpha 初始值 0.5 | [cy2abxu9](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/cy2abxu9) |
| 高层头 P3 | `Detect.cv3[0][1]` | 原始直接替换分类 logits，无 alpha | [2mx54q9b](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/2mx54q9b) |
| P3+P4+P5 多头 | `Detect.cv3[0/1/2][1]` | 三个尺度均直接替换分类 logits，无 alpha | [dtj4tsb4](https://wandb.ai/zing-c-ningbo-university/BGD-YOLO/runs/dtj4tsb4) |
| P3+P4+P5 多头复跑 | `Detect.cv3[0/1/2][1]` | 与上一行完全相同，seed 0、deterministic | [7k3zhkgt](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/7k3zhkgt) |

多头实现对 P3、P4、P5 分别配置独立的 `Conv2d + BatchNorm + ReLU` 投影参数，并保留各 Detect 头原有的独立分类器；三个头共享一次 Detail 前向和一次多目标 Grad-CAM 反向计算，不共享新增投影权重。

这里的“三头独立权重”指三个尺度各自的新增投影层和 Detect 分类分支参数彼此独立，并不是复制三套完整模型。YOLO backbone/neck、Detail 网络等主体仍然共享，所有共享参数和三组独立分支参数一起保存在同一个 `best.pt` 中。

## 50 epoch 验证结果

下表按每次训练中最高 `mAP50-95` 的 epoch 汇总。项目的 `best.pt` 实际按验证 `mAP50` 选取，因此可学习 alpha 实验的官方 test 使用的是 epoch 14（CSV epoch 13）的 `best.pt`，不是下表中 epoch 50 的 `last.pt`。

| 实验 | 最佳 epoch | P | R | mAP50 | mAP75 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 低层头 P5，alpha=0.5 | 28 | 0.96998 | 0.94313 | 0.97394 | 0.94600 | 0.91971 |
| 中层头 P4，alpha=0.5 | 41（CSV epoch 40） | 0.96598 | 0.94211 | 0.97255 | 0.94352 | 0.91871 |
| 中层头 P4，可学习 alpha | 50（CSV epoch 49） | 0.98497 | 0.92417 | 0.97282 | 0.94928 | 0.91925 |
| 高层头 P3，直接融合 | 49 | 0.97059 | 0.93852 | 0.97250 | 0.94742 | **0.92072** |
| P3+P4+P5 多头直接融合 | 32（CSV epoch 31） | 0.96584 | 0.93797 | 0.97338 | 0.94120 | 0.91638 |
| P3+P4+P5 多头复跑 | 21（CSV epoch 20） | 0.97776 | 0.93748 | 0.96906 | 0.93823 | 0.91496 |

## 统一 Test 结果

统一测试设置：`split=test`、`imgsz=640`、`conf=0.5`、`iou=0.5`、`batch=1`、GPU 0。

| 模型 | P | R | mAP50 | mAP75 | mAP50-95 | inference time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 `best318.pt` | 0.954128 | 0.888889 | 0.926514 | 0.901398 | 0.872197 | 54.90 ms/图 |
| 低层头 P5，alpha=0.5 | 0.950123 | 0.895489 | 0.935050 | 0.916033 | 0.887504 | 53.33 ms/图 |
| 中层头 P4，alpha=0.5 | 0.945701 | 0.893162 | 0.932220 | 0.912353 | 0.882639 | **53.03 ms/图** |
| 中层头 P4，可学习 alpha（best.pt，alpha=0.498287） | 0.945510 | 0.889851 | 0.935196 | 0.916356 | 0.885936 | 52.15 ms/图 |
| 高层头 P3，直接融合 | 0.942222 | 0.905983 | 0.937386 | 0.907642 | 0.882735 | 53.11 ms/图 |
| P3+P4+P5 多头直接融合 | 0.955302 | **0.913360** | 0.943453 | **0.920430** | 0.891103 | 54.13 ms/图 |
| P3+P4+P5 多头复跑 | **0.959459** | 0.910256 | **0.946014** | 0.914087 | **0.891513** | 53.61 ms/图 |

相对原始 `best318.pt`：

- 低层头 P5 的 test mAP50-95 提高 `0.015307`。
- 中层头 P4 的 test mAP50-95 提高 `0.010442`、mAP75 提高 `0.010955`、mAP50 提高 `0.005706`，推理时间减少约 `1.87 ms/图`。
- 中层头 P4 可学习 alpha 的 test mAP50-95 提高 `0.013739`、mAP75 提高 `0.014958`、mAP50 提高 `0.008682`，推理时间减少约 `2.75 ms/图`。
- 高层头 P3 的 test mAP50-95 提高 `0.010537`。
- P3+P4+P5 多头的 test mAP50-95 提高 `0.018905`，且推理时间减少约 `0.77 ms/图`。
- P3+P4+P5 多头复跑的 test mAP50-95 提高 `0.019315`，且推理时间减少约 `1.29 ms/图`。

可学习 alpha 相对固定 P4 alpha=0.5：test mAP50 提高 `0.002976`、mAP75 提高 `0.004002`、mAP50-95 提高 `0.003297`，推理时间减少约 `0.88 ms/图`；P 降低 `0.000191`，R 降低 `0.003311`。alpha 从 `0.500000` 更新至官方 `best.pt` 的 `0.498287`，第 50 轮 `last.pt` 为 `0.496979`。

多头相对单头实验：

- 相对低层头 P5，mAP50-95 提高 `0.003598`，inference 增加约 `0.81 ms/图`。
- 多头复跑相对中层头 P4，mAP50-95 提高 `0.008874`，inference 增加约 `0.58 ms/图`。
- 多头复跑相对中层头 P4 可学习 alpha，mAP50-95 提高 `0.005577`，inference 增加约 `1.46 ms/图`。
- 相对高层头 P3，mAP50-95 提高 `0.008368`，inference 增加约 `1.02 ms/图`。
- 第二次相对第一次多头：P 提高 `0.004157`、mAP50 提高 `0.002562`、mAP50-95 提高 `0.000410`、inference 减少约 `0.52 ms/图`；R 降低 `0.003104`、mAP75 降低 `0.006343`。
- 第一次多头的 R 和 mAP75 更高，第二次多头的 P、mAP50、mAP50-95 和速度更好；两次多头的 test mAP50-95 均高于原始 `best318.pt` 和三个单头实验。

## 关键权重位置

GitHub Release：[best318-gradcam-heads-20260821](https://github.com/zing-c/BGD-Yolo-v2/releases/tag/best318-gradcam-heads-20260821)

### 低层头 P5，alpha=0.5

- GitHub 下载：[best318_low_p5_alpha05_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_low_p5_alpha05_best.pt)
- 本地位置：`experiments/runs/best318_low_detectconv_gradcam_alpha05_e50_lr1e4/weights/best.pt`
- 文件大小：685,018,627 字节
- SHA-256：`f1a976ed1e34a7c9da759cd772600e6900f90ac18ab394db1dcb42a8fa234a4b`
- 对应 test mAP50-95：`0.88750449316474`

### 中层头 P4，alpha=0.5

- GitHub 下载：[best318_mid_p4_alpha05_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_mid_p4_alpha05_best.pt)
- 本地位置：`experiments/runs/best318_mid_p4_detectconv_alpha05_e50_lr1e4/weights/best.pt`
- 文件大小：685,017,091 字节
- SHA-256：`1042eef87aa36bb2af7d0a97f84ff39934c0b51c1696e9cfde0309d09fcb6a30`
- 对应 test mAP50-95：`0.8826389520844394`
- 训练结果：[results.csv](https://github.com/zing-c/BGD-Yolo-v2/blob/experiment/best318-multihead-direct/experiments/runs/best318_mid_p4_detectconv_alpha05_e50_lr1e4/results.csv)
- 训练曲线：[results.png](https://github.com/zing-c/BGD-Yolo-v2/blob/experiment/best318-multihead-direct/experiments/runs/best318_mid_p4_detectconv_alpha05_e50_lr1e4/results.png)
- 统一 test 日志：`experiments/best318_mid_p4_detectconv_alpha05_e50_lr1e4.test.log`（本地）
- 统一 test 输出：`experiments/runs/best318_mid_p4_detectconv_alpha05_e50_lr1e4_test/`（本地）

### 中层头 P4，可学习 alpha

- 官方 test 权重（按验证 mAP50 选取）：[best318_mid_p4_learnable_alpha_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_mid_p4_learnable_alpha_best.pt)
- 第 50 轮权重：[best318_mid_p4_learnable_alpha_last.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_mid_p4_learnable_alpha_last.pt)
- 本地 `best.pt`：`experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/weights/best.pt`
- `best.pt` 文件大小：685,017,534 字节
- `best.pt` SHA-256：`e0a53a49f76bc9d4675b35191e7b5b4689497489e77febcc8bc57df1bcb4687b`
- `best.pt` 对应 epoch 14 验证：P `0.97789`、R `0.94306`、mAP50 `0.97334`、mAP75 `0.94276`、mAP50-95 `0.91233`、alpha `0.49828720`
- `best.pt` 对应 test mAP50-95：`0.8859358672939293`
- 本地 `last.pt`：`experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/weights/last.pt`
- `last.pt` 文件大小：685,017,534 字节
- `last.pt` SHA-256：`91b94bf4250e2cedaa9f72ca901411334b9925b5b3859b2a698b4301861cf809`
- 第 50 轮验证：P `0.98497`、R `0.92417`、mAP50 `0.97282`、mAP75 `0.94928`、mAP50-95 `0.91925`、alpha `0.49697876`
- 详细结果：[JSON](https://github.com/zing-c/BGD-Yolo-v2/blob/experiment/best318-multihead-direct/experiments/results/best318_mid_p4_learnable_alpha_20260822.json)
- 训练结果：[results.csv](https://github.com/zing-c/BGD-Yolo-v2/blob/experiment/best318-multihead-direct/experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/results.csv)
- 训练曲线：[results.png](https://github.com/zing-c/BGD-Yolo-v2/blob/experiment/best318-multihead-direct/experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/results.png)
- 统一 test 日志：[test.log](https://github.com/zing-c/BGD-Yolo-v2/blob/experiment/best318-multihead-direct/experiments/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2.test.log)

### 高层头 P3，直接融合

- GitHub 下载：[best318_high_p3_direct_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_high_p3_direct_best.pt)
- 本地位置：`experiments/runs/best318_high_detectconv_direct_e50_lr1e4/weights/best.pt`
- 文件大小：685,017,027 字节
- SHA-256：`75e20890c69e3da39d3c477d2877a384c025ad6eb36f2b4555ed2dfb995bf904`
- 对应 test mAP50-95：`0.8827347444334619`

### P3+P4+P5 多头直接融合，无 alpha

- GitHub 下载：[best318_multihead_p3_p4_p5_direct_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_multihead_p3_p4_p5_direct_best.pt)
- 本地位置：`experiments/runs/best318_multihead_all_detectconv_direct_e50_lr1e4/weights/best.pt`
- 文件大小：684,152,405 字节
- SHA-256：`b165bb813e7b76184c835bd1c713ad6dc1f137eed0533c4506cbfe6dc3e516de`
- 对应 test mAP50-95：`0.89110251959343`
- 训练结果：`experiments/runs/best318_multihead_all_detectconv_direct_e50_lr1e4/results.csv`
- 统一 test 输出：`experiments/runs/best318_multihead_all_detectconv_direct_e50_lr1e4_test_retry/`

### P3+P4+P5 多头复跑，无 alpha

- GitHub 下载：[best318_multihead_p3_p4_p5_direct_rerun2_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_multihead_p3_p4_p5_direct_rerun2_best.pt)
- 本地位置：`experiments/runs/best318_multihead_all_detectconv_direct_e50_lr1e4_rerun2/weights/best.pt`
- 文件大小：684,152,405 字节
- SHA-256：`8cc1d899c10fa9cd9860bca542fa8bc0e85dd446c32766ef9db84b8c5faf0daa`
- 对应 test mAP50-95：`0.8915125356452475`
- 训练结果：`experiments/runs/best318_multihead_all_detectconv_direct_e50_lr1e4_rerun2/results.csv`
- 统一 test 日志：`experiments/best318_multihead_all_detectconv_direct_e50_lr1e4_rerun2.test.log`
- 统一 test 输出：`experiments/runs/best318_multihead_all_detectconv_direct_e50_lr1e4_rerun2_test/`

## 代码位置

- P4 可学习 alpha 结构图细节与生成 Prompt：[P4_LEARNABLE_ALPHA_DETAIL_FUSION_ARCHITECTURE.md](P4_LEARNABLE_ALPHA_DETAIL_FUSION_ARCHITECTURE.md)
- P4 框架图真实测试样本与分阶段可视化素材：[P4_FRAMEWORK_VISUAL_ASSETS.md](P4_FRAMEWORK_VISUAL_ASSETS.md)
- 单头实验归档分支：[archive/best318-gradcam-heads-20260821](https://github.com/zing-c/BGD-Yolo-v2/tree/archive/best318-gradcam-heads-20260821)
- 多头无 alpha 实验分支：[experiment/best318-multihead-direct](https://github.com/zing-c/BGD-Yolo-v2/tree/experiment/best318-multihead-direct)
- 主要运行脚本：`experiments/run_318_fusion_alpha.py`
- 可学习 alpha 实现提交：[f50e6f7](https://github.com/zing-c/BGD-Yolo-v2/commit/f50e6f7)、[36bd84a](https://github.com/zing-c/BGD-Yolo-v2/commit/36bd84a)
- 可学习 alpha 结果归档提交：[7ac02a7](https://github.com/zing-c/BGD-Yolo-v2/commit/7ac02a7)

四个单头实验与两次多头 P3+P4+P5 直接融合实验均已完成全部 50 epochs、统一 test、权重 SHA-256 校验和 GitHub Release 上传。
