# Grad-CAM Detect Head 实验结果

更新时间：2026-08-21

本文档集中记录 `best318` 框架下两个已完成的单 Detect 头实验，包括训练参数、验证指标、统一 test 指标、推理时间、权重下载位置和文件校验值。

## 实验设置

两组实验均从 `best318.pt` 之前的独立 YOLO 与 Detail 预训练权重初始化，不使用 `best318.pt` 初始化。

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

实验结构：

| 实验 | Grad-CAM/Detect 位置 | 融合方式 | W&B |
| --- | --- | --- | --- |
| 低层头 P5 | `Detect.cv3[2][1]` | `base + 0.5 × (fused - base)` | [1ghxfq1l](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/1ghxfq1l) |
| 高层头 P3 | `Detect.cv3[0][1]` | 原始直接替换分类 logits，无 alpha | [2mx54q9b](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/2mx54q9b) |

## 50 epoch 最佳验证指标

| 实验 | 最佳 epoch | P | R | mAP50 | mAP75 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 低层头 P5，alpha=0.5 | 28 | 0.96998 | 0.94313 | 0.97394 | 0.94600 | 0.91971 |
| 高层头 P3，直接融合 | 49 | 0.97059 | 0.93852 | 0.97250 | 0.94742 | **0.92072** |

## 统一 Test 结果

统一测试设置：`split=test`、`imgsz=640`、`conf=0.5`、`iou=0.5`、`batch=1`、GPU 0。

| 模型 | P | R | mAP50 | mAP75 | mAP50-95 | inference time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 `best318.pt` | **0.954128** | 0.888889 | 0.926514 | 0.901398 | 0.872197 | 54.9 ms/图 |
| 低层头 P5，alpha=0.5 | 0.950123 | 0.895489 | 0.935050 | **0.916033** | **0.887504** | 53.33 ms/图 |
| 高层头 P3，直接融合 | 0.942222 | **0.905983** | **0.937386** | 0.907642 | 0.882735 | **53.11 ms/图** |

相对原始 `best318.pt`：

- 低层头 P5 的 test mAP50-95 提高 `0.015307`。
- 高层头 P3 的 test mAP50-95 提高 `0.010537`。
- 低层头 P5 的综合 test mAP50-95 最好，因此没有触发更换 alpha 继续训练的条件。

## 关键权重位置

GitHub Release：[best318-gradcam-heads-20260821](https://github.com/zing-c/BGD-Yolo-v2/releases/tag/best318-gradcam-heads-20260821)

### 低层头 P5，alpha=0.5

- GitHub 下载：[best318_low_p5_alpha05_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_low_p5_alpha05_best.pt)
- 本地位置：`experiments/runs/best318_low_detectconv_gradcam_alpha05_e50_lr1e4/weights/best.pt`
- 文件大小：685,018,627 字节
- SHA-256：`f1a976ed1e34a7c9da759cd772600e6900f90ac18ab394db1dcb42a8fa234a4b`
- 对应 test mAP50-95：`0.88750449316474`

### 高层头 P3，直接融合

- GitHub 下载：[best318_high_p3_direct_best.pt](https://github.com/zing-c/BGD-Yolo-v2/releases/download/best318-gradcam-heads-20260821/best318_high_p3_direct_best.pt)
- 本地位置：`experiments/runs/best318_high_detectconv_direct_e50_lr1e4/weights/best.pt`
- 文件大小：685,017,027 字节
- SHA-256：`75e20890c69e3da39d3c477d2877a384c025ad6eb36f2b4555ed2dfb995bf904`
- 对应 test mAP50-95：`0.8827347444334619`

## 代码位置

- 单头实验归档分支：[archive/best318-gradcam-heads-20260821](https://github.com/zing-c/BGD-Yolo-v2/tree/archive/best318-gradcam-heads-20260821)
- 多头无 alpha 实验分支：[experiment/best318-multihead-direct](https://github.com/zing-c/BGD-Yolo-v2/tree/experiment/best318-multihead-direct)
- 主要运行脚本：`experiments/run_318_fusion_alpha.py`

多头 P3+P4+P5 直接融合实验仍在训练中，因此本文档暂不记录其中间权重；训练、统一 test 和权重校验全部完成后再补充最终结果。
