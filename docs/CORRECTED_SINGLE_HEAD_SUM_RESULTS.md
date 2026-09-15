# P3/P4/P5 Corrected Single-head Direct（Sum）实验结果

更新日期：2026-09-14

2026-09-15 补充：[最新单头与三头 Test + 三轮统一推理时间复测](CORRECTED_DIRECT_HEADS_TEST_TIMING_20260915.md)。当前速度比较以新页为准；P4 复测 AP 有微小差异，新页保留说明，本页原始数字不覆盖。

本文档记录修正 Detail scatter 后，P3、P4、P5 三个 Detect 内部单头 Direct 模型的统一实验结果。
这三组实验均不使用固定或可学习 alpha，重叠 Detail 特征采用 `sum` 累加，并使用 validation
指标选择 `best.pt`，test 集只用于最后一次报告。

> 本页是当前单头 Direct 尺度消融的权威结果。旧文档中的 legacy 单头结果没有同时修正
> scatter 写入和坐标次序，不能与本页混为论文主结果。

## 1. 修正内容

旧实现存在两个会影响 Detail 融合有效性的代码问题：

1. `detail_feature[..., ...].add(current_box_output)` 是非原地运算，返回值没有被保存，因此写入的
   Detail scatter 可能仍为全零；正确实现使用 `add_`。
2. Grad-CAM 中心返回顺序是 `[x, y]`，而 BCHW 张量空间索引顺序是 `[y, x]`；修正后分别使用
   `center_y` 定位高度维、`center_x` 定位宽度维。

修正实现位于 `experiments/run_318_fusion_alpha.py` 的
`patch_corrected_single_head_direct()`，训练入口使用 `--corrected-direct-fusion`。融合分类 logit
直接替换所选尺度的原分类 logit，不引入 alpha：

\[
F_l'=\operatorname{ReLU}\!\left(\operatorname{BN}_l\left(
\operatorname{Conv}_{3\times3,l}([F_l;D_l])\right)\right),
\qquad z_l'=\operatorname{Conv}_{cls,l}(F_l').
\]

框回归/DFL 分支保持原 YOLO 输出不变。

## 2. Detail 输出与尺度适配

P3、P4、P5 三组实验使用的是**同一个** `Detail_Net_attn_block` 结构和同一个预训练权重
`run/detail_net_attn.pt`。对每个 `64×64` RGB crop，Detail model 的原始输出始终是
`3×4×4`，并不存在独立的 P5 `2×2` Detail model。

为了让同一个 `64×64` 原图区域在不同 stride 的 YOLO 特征图上覆盖相同物理范围，scatter 前
按目标尺度调整 Detail feature：

\[
k_{h,l}=\max\!\left(1,\operatorname{round}\frac{64}{s_{y,l}}\right),
\qquad
k_{w,l}=\max\!\left(1,\operatorname{round}\frac{64}{s_{x,l}}\right),
\]

其中 (s_{y,l}=H_{in}/H_l\)、(s_{x,l}=W_{in}/W_l\)。在 `640×640` 输入下：

| 单头 | Hook 位置 | 全局特征 | Stride | Detail 原始输出 | Scatter 局部块 | 适配操作 |
|---|---|---:|---:|---:|---:|---|
| P3 / high | `Detect.cv3[0][1]` | `64×80×80` | 8 | `3×4×4` | `3×8×8` | 双线性上采样 |
| P4 / mid | `Detect.cv3[1][1]` | `64×40×40` | 16 | `3×4×4` | `3×4×4` | 恒等映射 |
| P5 / low | `Detect.cv3[2][1]` | `64×20×20` | 32 | `3×4×4` | `3×2×2` | `2×2` 平均池化 |

设 CAM 中心得到的输入坐标为 `(x, y)`，映射到第 (l) 层后，在 BCHW 张量中写入：

\[
D_l[b,:,y_0:y_0+k_{h,l},x_0:x_0+k_{w,l}]\mathrel{+}=d_i^{(l)}.
\]

本实验的重叠区域使用 `sum`，即多个候选覆盖同一网格时直接累加，不除以覆盖次数。

## 3. 共同训练与测试设置

初始化不是 `best318.pt`，而是融合前的两个独立 checkpoint：

- YOLO：`yolo-runs/train/train10/weights/best.pt`
- Detail：`run/detail_net_attn.pt`
- 融合投影：随机初始化

三个模型均联合微调 YOLO、Detail 和融合层。共同训练设置如下：

| 参数 | 设置 |
|---|---|
| Dataset | `experiments/bg_local.yaml` |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 32 |
| Optimizer | AdamW |
| Initial LR | `1e-4` |
| Final LR factor | `0.01` |
| Momentum | `0.937` |
| Weight decay | `5e-4` |
| Warmup | 3 epochs；momentum `0.8`；bias LR `0.0` |
| AMP | enabled |
| Candidate confidence | `0.2` |
| Alpha | 无 |
| Scatter reduction | `sum` |
| Total parameters | 4,209,276 |

正式 test 协议统一为：`split=test`、1598 images、234 instances、`imgsz=640`、
`conf=0.5`、NMS IoU `0.5`、batch 1、RTX 6000 Ada。

## 4. Validation 选权重

当前 fork 的 `best.pt` fitness 只使用 validation mAP50，而不是 test 指标。因此下表第一部分给出
实际选入 `best.pt` 的 validation 行；“最高 mAP50-95”仅供分析，并未用于选权重。

### 4.1 实际 `best.pt` 对应的 validation 行

| 模型 | Epoch | P | R | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| P3 correct sum | 7 | 0.97849 | 0.93365 | **0.97294** | 0.94732 | 0.91585 |
| P4 correct sum | 36 | 0.97535 | 0.93749 | **0.97393** | 0.94193 | 0.91317 |
| P5 correct sum | 32 | 0.97666 | 0.93839 | **0.97339** | 0.94048 | 0.91590 |

### 4.2 训练期最高 validation mAP50-95 行

| 模型 | Epoch | P | R | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| P3 correct sum | 3 | 0.98013 | 0.93531 | 0.97154 | 0.94414 | **0.92009** |
| P4 correct sum | 47 | 0.98472 | 0.93128 | 0.97263 | 0.94569 | **0.91931** |
| P5 correct sum | 50 | 0.97662 | 0.93365 | 0.97249 | 0.94099 | **0.91876** |

## 5. 首次 Test 记录（旧时间不作同场速度排名）

所有 P3/P4/P5 test 数字均来自各自 validation-selected `best.pt`；没有用 test 集挑选 epoch 或
超参数。

| 模型 | P | R | mAP50 | mAP75 | mAP50-95 | Inference |
|---|---:|---:|---:|---:|---:|---:|
| Pure YOLO 640 baseline | 0.932735 | 0.888889 | 0.926368 | 0.898112 | 0.870250 | 1.68 ms/image* |
| P3 correct sum | 0.941964 | **0.901709** | 0.932006 | 0.905852 | 0.877765 | **51.45 ms/image** |
| P4 correct sum | **0.963123** | 0.892884 | **0.938131** | 0.910592 | 0.884545 | 55.97 ms/image |
| P5 correct sum | 0.950226 | 0.897436 | 0.934471 | **0.916394** | **0.884661** | 53.57 ms/image |

`*` Pure YOLO latency 来自既有统一测试记录；融合方法延迟包含 Grad-CAM backward、候选 crop、
Detail forward 和第二次分类计算，不能只按参数量推断。

相对 Pure YOLO 640 baseline 的绝对提升（百分点）为：

| 模型 | ΔmAP50 | ΔmAP75 | ΔmAP50-95 |
|---|---:|---:|---:|
| P3 correct sum | +0.564 | +0.774 | +0.752 |
| P4 correct sum | **+1.176** | +1.248 | +1.430 |
| P5 correct sum | +0.810 | **+1.828** | **+1.441** |

结论：P4 的 Precision 和 mAP50 最高；P5 的 mAP75 和 mAP50-95 最高；P3 的 Recall 最高。
P5 与 P4 的 mAP50-95 只差 `0.000116`（0.0116 个百分点），应视为基本持平，不宜宣称 P5
显著优于 P4。

## 6. P5 感受野与融合风险

P5 的 stride 为 32、空间尺寸仅 `20×20`，原生感受野和语义抽象程度都高于 P3/P4。因此把
`3×4×4` Detail feature 平均池化为 `3×2×2` 后，细微纹理有被压缩、被强语义特征稀释的风险；
同时，P5 Grad-CAM 本身也比 P3/P4 粗糙。这是合理的尺度/语义失配风险。

但 `2×2` scatter 在几何上是正确的：P5 的一个网格对应输入约 `32×32`，所以 `64×64` crop
应覆盖 `2×2` 网格。若强行写入 `4×4`，实际对应约 `128×128`，会改变局部区域的物理范围。
当前 Test 中 P5 mAP50-95 与 P4 持平、mAP75 反而更高，未出现明显融合失败。因为 Detail 路径
不修改框回归分支，较高 mAP75 应解释为融合分类分数更有利于保留/排序原本定位较准的框，而不是
Detail 直接改善了坐标回归。

## 7. 权重与可复查产物

| 模型 | `best.pt` 本地位置 | 大小（bytes） | SHA-256 |
|---|---|---:|---|
| P3 correct sum | `experiments/runs/best318_high_p3_detecthead_corrected_xy_scale8_sum_e50_lr1e4_r1/weights/best.pt` | 685,017,411 | `75525a957aeef9d85c4abb6240633b2e931339a6643c444603ba0e1efb488604` |
| P4 correct sum | `experiments/runs/best318_mid_p4_detecthead_correctedscatter_direct_e50_lr1e4_r1/weights/best.pt` | 685,018,819 | `83ae89e54500be2e7ac8722c66ffa925563b4f53366cad44c9415cf391b5e130` |
| P5 correct sum | `experiments/runs/best318_low_p5_detecthead_corrected_xy_scale2_direct_e50_lr1e4_r2/weights/best.pt` | 685,017,411 | `7921171241048cf56cda758d3a995d79f0e653b4e91b3d79c8d7408901e527a8` |

训练日志：

- P3：`experiments/best318_high_p3_detecthead_corrected_xy_scale8_sum_e50_lr1e4_r1.train.log`
- P4：`experiments/best318_mid_p4_detecthead_correctedscatter_direct_e50_lr1e4_r1.train.log`
- P5：`experiments/best318_low_p5_detecthead_corrected_xy_scale2_direct_e50_lr1e4_r2.train.log`

最终测试目录与日志：

- P3：`experiments/runs/best318_high_p3_detecthead_corrected_xy_scale8_sum_e50_final_test`；
  `experiments/best318_high_p3_detecthead_corrected_xy_scale8_sum_e50_lr1e4_r1.final_test.log`
- P4：`experiments/runs/best318_mid_p4_detecthead_correctedscatter_direct_e50_final_test`；
  `experiments/best318_mid_p4_detecthead_correctedscatter_direct_e50_lr1e4_r1.final_test.log`
- P5：`experiments/runs/best318_low_p5_detecthead_corrected_xy_scale2_direct_e50_r2_final_test`；
  `experiments/best318_low_p5_detecthead_corrected_xy_scale2_direct_e50_lr1e4_r2.final_test.log`

## 8. Sum 与 Mean 补充消融

相同修正版 P4 使用 `mean` 聚合时，Test mAP50-95 为 `0.883400`；本页 P4 `sum` 为
`0.884545`，高 `0.001145`（0.1145 个百分点）。因此目前保留 `sum` 作为 P3/P4/P5 单头
尺度消融的统一设置。

## 9. 复现实例

以下命令中的 `high` 可替换为 `mid` 或 `low`：

```bash
python -u experiments/run_318_fusion_alpha.py train \
  --corrected-direct-fusion --scatter-reduce sum \
  --candidate-conf 0.2 --head-select high \
  --epochs 50 --optimizer AdamW --lr0 0.0001 --lrf 0.01 \
  --momentum 0.937 --weight-decay 0.0005 \
  --warmup-epochs 3.0 --warmup-momentum 0.8 --warmup-bias-lr 0.0 \
  --batch 32 --workers 8 --device 0 --amp --patience 50 \
  --name corrected_p3_sum_e50
```
