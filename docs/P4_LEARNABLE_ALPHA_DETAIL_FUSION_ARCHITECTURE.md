# P4 可学习 Alpha 细节检测与融合结构图说明

更新时间：2026-08-22

本文档用于绘制当前 `best318` 框架下 **P4 单头、Grad-CAM 引导细节截图、可学习 alpha logit 融合** 的论文结构图。图中应简化 YOLO 主干，重点展开候选生成、Grad-CAM 定位、Detail 特征提取、空间回填和 P4 分类融合。

> 实际训练权重中的检测器是三尺度 YOLOv8n，包含 P3、P4、P5，stride 分别为 8、16、32。旧训练目录中的部分 `args.yaml` 与实际序列化权重不一致，结构图应以实际权重中的三尺度模型为准。

## 1. 推荐的整图布局

论文主图建议从左到右分为五部分：

1. 简化 YOLO 检测器；
2. 候选框与 Grad-CAM 定位；
3. 多分支 Detail Encoder；
4. Detail 特征空间回填；
5. P4 分类 logit 与可学习 alpha 融合。

```text
Input Image
    │
    ▼
Simplified YOLOv8 Detector
    ├── P3 / P4 / P5 preliminary predictions
    │          │
    │          └── candidate boxes: class confidence > 0.2
    │
    └── P4 classification feature: 64×40×40
               │
               └── Grad-CAM heatmap
                         │
Candidate boxes ─────────┘
          │
Grad-CAM-guided localization
          │
64×64 detail crops from original image
          │
Three-route Detail Encoder
          │
N×3×4×4 detail features
          │
Coordinate-aware Scatter-and-Add
          │
3×40×40 sparse detail map
          │
Concat with P4 classification feature
          │
Conv3×3(67→64) + BN + ReLU
          │
P4 original Conv1×1 classifier
          │
Detail-enhanced P4 logit
          │
Learnable alpha logit interpolation
          │
Final P4 logit + unchanged P3/P5 outputs
          │
Decode and NMS
```

## 2. YOLO 部分的简化画法

结构图不需要展开完整 Backbone 和 PAN-FPN，只需保留：

```text
Input 3×640×640
        ↓
YOLOv8 Backbone + Neck
        ↓
P3: 64×80×80, stride 8
P4: 128×40×40, stride 16
P5: 256×20×20, stride 32
        ↓
Decoupled Detect Head
```

每个 Detect level 包含独立的框回归分支 `cv2` 和分类分支 `cv3`。类别数为 1，`reg_max=16`，因此每个位置的原始输出为 64 个回归通道和 1 个分类 logit。三尺度共 `80²+40²+20²=8400` 个位置。

当前方法只修改 P4 的分类 logit：

- P3 分类输出保持不变；
- P5 分类输出保持不变；
- P3、P4、P5 的框回归分支均保持不变；
- Detail 分支不直接修改边界框坐标。

## 3. P4 Grad-CAM 位置

Hook 位于：

```text
Detect.cv3[1][1]
```

它是 P4 分类分支的第二个 `Conv3×3 + BatchNorm + SiLU`，位于最终 `Conv1×1` 分类器之前：

```text
P4 feature: 128×40×40
        ↓
Conv3×3 + BN + SiLU, 128→64
        ↓
Conv3×3 + BN + SiLU, 64→64  ← Grad-CAM hook
        ↓
Conv1×1, 64→1
        ↓
P4 class logit: 1×40×40
```

Hook 激活为：

\[
A^{P4}\in\mathbb{R}^{B\times64\times40\times40}.
\]

Grad-CAM 的目标是每张图所有 8400 个检测位置的单类别置信度之和：

\[
S_b=\sum_{n=1}^{8400}p_{b,n}.
\]

经典 Grad-CAM 通道权重和热图为：

\[
w_k=\frac{1}{HW}\sum_{i,j}\frac{\partial S}{\partial A_{kij}},
\]

\[
M=\operatorname{ReLU}\left(\sum_k w_kA_k\right).
\]

所得热图从 `40×40` 上采样到输入图像尺寸并归一化到 `[0,1]`。

## 4. 候选框与细节截图

初步 YOLO 解码结果的形状为：

\[
Y^{pre}\in\mathbb{R}^{B\times5\times8400},
\]

其中包含 4 个框坐标和 1 个类别置信度。通过以下条件选择候选：

\[
p_{b,n}>0.2.
\]

这里的 `0.2` 是内部候选阈值，不是最终测试的 `conf=0.5`。候选在最终 NMS 之前产生。

每个候选框的截图流程如下：

1. 将 Grad-CAM 热图限制在该候选框范围内；
2. 在候选框内滑动方形窗口；
3. 选择窗口内 CAM 响应和最大的位置；
4. 以最大响应窗口的中心作为细节中心；
5. 若候选框小于搜索窗口，则退化为候选框几何中心；
6. 将 letterbox 坐标转换回原始图像坐标；
7. 从原图裁剪固定 `64×64` RGB 图像块；
8. 越界区域使用黑色填充；
9. 除以 255 后进行 ImageNet 均值和标准差归一化。

归一化参数为：

\[
\mu=(0.485,0.456,0.406),\qquad
\sigma=(0.229,0.224,0.225).
\]

若一个 batch 共有 `N` 个候选，则 Detail 输入为：

\[
I_{detail}\in\mathbb{R}^{N\times3\times64\times64}.
\]

结构图中建议将这一模块命名为 **Grad-CAM-guided Detail Crop Generator**。

## 5. Detail Feature Extraction Network

当前 `Detail_Net_attn_block` 是特征提取器，不输出最终类别概率。输入首先经过 `MaxPool2d(2,2)`：

```text
N×3×64×64 → N×3×32×32
```

之后进入 Large、Medium、Small 三条并行感受野路线。

### 5.1 Large Route

```text
3×32×32
  ↓ Residual Conv3×3, 3→16
16×32×32
  ↓ MaxPool2
16×16×16
  ↓ Residual Conv3×3, 16→32
32×16×16
  ↓ MaxPool2
32×8×8
  ↓ Residual Conv3×3, 32→64
64×8×8
  ↓ MaxPool2
64×4×4
  ↓ Residual Conv3×3, 64→128
128×4×4
  ↓ Multi-Dilation Attention
128×4×4
  ↓ Conv1×1, 128→1
1×4×4
```

### 5.2 Medium Route

```text
3×32×32
  ├── Main: Conv6×6, stride 4, 3→16
  └── Shortcut: MaxPool4 + Conv1×1, 3→16
  ↓ Add + ReLU
16×8×8
  ↓ Residual Conv3×3, 16→32
32×8×8
  ↓ MaxPool2
32×4×4
  ↓ Residual Conv3×3, 32→64
64×4×4
  ↓ Multi-Dilation Attention
64×4×4
  ↓ Conv1×1, 64→1
1×4×4
```

### 5.3 Small Route

```text
3×32×32
  ├── Main: Conv13×13, stride 8, 3→32
  └── Shortcut: MaxPool8 + Conv1×1, 3→32
  ↓ Add + ReLU
32×4×4
  ↓ Residual Conv3×3, 32→64
64×4×4
  ↓ Multi-Dilation Attention
64×4×4
  ↓ Conv1×1, 64→1
1×4×4
```

每个 Residual Conv 块表示：

```text
Main: Conv3×3 + BN
Shortcut: Conv1×1 + BN
Main + Shortcut → ReLU
```

## 6. Multi-Dilation Attention

三条路线分别使用相同结构的多膨胀率注意力模块：

```text
Input feature F
  ├── Conv3×3, dilation 1
  ├── Conv3×3, dilation 3
  ├── Conv3×3, dilation 5
  └── Conv3×3, dilation 7
            ↓
      Channel Concat
            ↓
         Conv1×1
            ↓
          Sigmoid
            ↓
 Element-wise multiplication with F
            ↓
      Attended feature
```

公式为：

\[
Q(F)=[\operatorname{Conv}_{d=1}(F);\operatorname{Conv}_{d=3}(F);
\operatorname{Conv}_{d=5}(F);\operatorname{Conv}_{d=7}(F)],
\]

\[
W(F)=\sigma(\operatorname{Conv}_{1\times1}(Q(F))),
\]

\[
F_{att}=F\odot W(F).
\]

## 7. 三路 Detail 特征输出

三条路线分别压缩为一个通道：

\[
F_L,F_M,F_S\in\mathbb{R}^{N\times1\times4\times4}.
\]

随后进行通道拼接、BatchNorm 和 ReLU：

\[
F_D=\operatorname{ReLU}\left(
\operatorname{BN}([F_L;F_M;F_S])
\right),
\]

\[
F_D\in\mathbb{R}^{N\times3\times4\times4}.
\]

图中可以表示为：

```text
Large Route  → 1×4×4 ─┐
Medium Route → 1×4×4 ─┼→ Concat → BN(3) → ReLU → 3×4×4
Small Route  → 1×4×4 ─┘
```

## 8. Detail 特征空间回填

为 P4 建立全零稀疏 Detail 特征图：

\[
F_{scatter}^{P4}\in\mathbb{R}^{B\times3\times40\times40}.
\]

P4 stride 为 16，因此候选中心从图像坐标映射到 P4 网格：

\[
(x_{P4},y_{P4})=(x/16,y/16).
\]

每个候选的 `3×4×4` Detail 特征被放置到对应中心附近的 `4×4` 区域。多个候选重叠时采用逐元素相加，而不是覆盖：

\[
F_{scatter}(x,y)=\sum_{n\in\mathcal{N}(x,y)}F_D^{(n)}(x,y).
\]

结构图中建议命名为 **Coordinate-aware Scatter-and-Add**。该操作保留候选的空间位置，不应画成 Global Average Pooling 或简单向量拼接。

## 9. P4 Detail–YOLO 特征融合

P4 Hook 特征为：

\[
F_Y\in\mathbb{R}^{B\times64\times40\times40},
\]

稀疏 Detail 特征图为：

\[
F_D^{map}\in\mathbb{R}^{B\times3\times40\times40}.
\]

首先进行通道拼接：

\[
F_{cat}=[F_Y;F_D^{map}]
\in\mathbb{R}^{B\times67\times40\times40}.
\]

然后使用新建的投影模块：

```text
67×40×40
  ↓ Conv3×3, stride 1, padding 1, 67→64
  ↓ BatchNorm2d(64)
  ↓ ReLU
64×40×40
```

即：

\[
F_{fusion}=\operatorname{ReLU}\left(
\operatorname{BN}(\operatorname{Conv}_{3\times3}([F_Y;F_D^{map}]))
\right).
\]

投影层参数量为：

- `Conv2d(67,64,3,padding=1)`：38,656；
- `BatchNorm2d(64)` 可训练参数：128；
- 投影模块合计：38,784。

融合特征继续经过 P4 原有分类器 `Detect.cv3[1][2]`：

```text
Conv1×1, 64→1
```

得到 Detail 增强分类 logit：

\[
z_{detail}\in\mathbb{R}^{B\times1\times40\times40}.
\]

这里复用 P4 原来的最终分类器，没有新建第二个 `Conv1×1` 分类器。

## 10. 可学习 Alpha Logit 融合

原始 P4 分类 logit 为：

\[
z_{base}\in\mathbb{R}^{B\times1\times40\times40}.
\]

最终 P4 logit 在 Sigmoid 之前融合：

\[
z_{out}=z_{base}+\alpha(z_{detail}-z_{base}),
\]

等价于：

\[
z_{out}=(1-\alpha)z_{base}+\alpha z_{detail}.
\]

使用一个全局可学习标量 `beta`，并通过 Sigmoid 限制 alpha：

\[
\alpha=\sigma(\beta),\qquad 0<\alpha<1.
\]

- 初始 `alpha=0.5`，因此初始 `beta=0`；
- 官方 `best.pt` 中 `beta=-0.006851`，`alpha=0.498287`；
- 第 50 轮 `last.pt` 中 `alpha=0.496979`；
- alpha 对所有图像、候选、空间位置和通道共用；
- alpha 位于 logit 层，不是 Sigmoid 后的概率融合；
- alpha 参数进入无 weight decay 优化器参数组。

结构图中应画成：

```text
Original P4 logit z_base ─────── ×(1-alpha) ──┐
                                               ├─ Add → z_out
Detail-enhanced logit z_detail ── ×alpha ──────┘

                         alpha = Sigmoid(beta)
                         beta: learnable scalar
```

## 11. 最终输出

融合后的 `z_out` 替换 P4 原始分类 logit，之后与未修改的 P3、P5输出共同进入：

```text
DFL box decoding
        ↓
Classification sigmoid
        ↓
Final confidence filtering
        ↓
NMS
        ↓
Final detections
```

内部候选阈值为 `0.2`；已有统一 test 使用的最终检测阈值为 `0.5`，NMS IoU 为 `0.5`。

## 12. 可微与不可微路径

图中建议用黑色实线表示可微路径，用橙色虚线表示不可微路径，用红色虚线表示用于生成 Grad-CAM 的内部反向路径。

不可微步骤包括：

- CAM 转 CPU/NumPy；
- 候选阈值筛选；
- 最大响应窗口和中心选择；
- 坐标变换；
- OpenCV 原图裁剪；
- Detail 特征 scatter 位置选择。

最终检测损失仍可更新：

- YOLO Backbone、Neck 和 Detect；
- P4 分类分支；
- Detail Encoder；
- `Conv(67→64)+BN` 投影模块；
- 可学习 `beta`。

Grad-CAM 内部反向只用于得到热图，随后清空这次内部反向产生的梯度，不直接作为优化器更新梯度。

## 13. 推理过程

该方法不是两次完整 YOLO 前向，实际过程为：

```text
One YOLO backbone/neck/head forward
        ↓
Preliminary decoding and candidate selection
        ↓
One Grad-CAM backward pass
        ↓
One batched Detail Encoder forward for all crops
        ↓
P4 classification-logit modification
        ↓
Second Detect decoding only
        ↓
Final NMS
```

若没有候选框，模型直接返回原始 YOLO 输出。

## 14. 参数量

| 部分 | 参数量 |
| --- | ---: |
| YOLOv8n 主体 | 3,011,043 |
| Detail Encoder 主体 | 1,159,449 |
| P4 `Conv+BN` 融合投影 | 38,784 |
| 可学习 `beta` | 1 |
| 总参数量 | 4,209,277 |

## 15. 绘图时不要画错的内容

- 实际检测器为 P3、P4、P5 三尺度，不是两尺度；
- P3 是 stride 8 的高分辨率尺度，主要对应小目标；
- Detail Encoder 输出 `3×4×4` 特征，不是 Detail 分类概率；
- Detail 分支直接修改分类 logit，不直接修改边界框；
- alpha 位于 Sigmoid 之前的 logit 层；
- 当前只有一个全局 alpha，不是每个候选或每个位置一个 alpha；
- 截图来自原始图像，不是 GPU 特征上的 ROIAlign；
- 定位、裁剪和位置选择不是端到端可微操作；
- 只执行一次 YOLO Backbone/Neck 前向，但额外执行一次 Grad-CAM 反向和一次最终 Detect 解码；
- 代码 scatter 时第一个中心坐标被用于第一个空间维，`x/y` 行列顺序在发表前仍建议单独审计；概念图中写作 `coordinate mapping` 即可。

## 16. 结构图推荐标签

可直接使用以下英文标签：

- `Simplified YOLOv8 Detector`；
- `Preliminary Predictions`；
- `Candidate Selection (confidence > 0.2)`；
- `P4 Classification Feature (64×40×40)`；
- `Grad-CAM Localization`；
- `Grad-CAM-guided Detail Crop Generator`；
- `64×64 Detail Crops`；
- `Three-route Detail Encoder`；
- `Multi-Dilation Attention (d=1,3,5,7)`；
- `Detail Feature (3×4×4 per candidate)`；
- `Coordinate-aware Scatter-and-Add`；
- `Sparse Detail Map (3×40×40)`；
- `Channel Concatenation (67×40×40)`；
- `Conv3×3 (67→64) + BN + ReLU`；
- `Detail-enhanced P4 Logit`；
- `Learnable Logit Interpolation`；
- `alpha = sigmoid(beta)`；
- `Final Detection`。

## 17. 论文矢量结构图生成 Prompt

下面的 Prompt 适合用于生成横向论文方法结构图。为了减少图片模型生成错误文字，建议先生成布局，再用 PowerPoint、Figma、Illustrator 或 draw.io 重写全部标签和公式。

```text
Create a clean, publication-quality horizontal neural-network architecture diagram for a computer-vision research paper. White background, flat vector graphics, precise alignment, thin dark-gray arrows, restrained blue-green-orange-purple color palette, no 3D effects, no photorealism, 16:9 landscape composition, high resolution.

The method is a Grad-CAM-guided detail feature fusion detector with a learnable scalar alpha at the P4 classification-logit level. Simplify the YOLO section and expand the detail and fusion sections.

Layout from left to right:

1. Input and simplified detector:
- Show an input image block labeled “Input Image, 3×640×640”.
- Connect it to one compact blue block labeled “Simplified YOLOv8 Detector, Backbone + Neck”.
- Show three compact outputs: “P3, 64×80×80, stride 8”, “P4, 128×40×40, stride 16”, and “P5, 256×20×20, stride 32”.
- Do not expand the YOLO backbone layers.
- Show a preliminary detection block receiving P3, P4, and P5 outputs.
- From preliminary predictions, show “Candidate Selection, class confidence > 0.2”.

2. Grad-CAM localization:
- Expand only the P4 classification branch as “Conv3×3 + BN + SiLU, 128→64”, then “Conv3×3 + BN + SiLU, 64→64”.
- Mark the output of the second convolution with a red hook labeled “Grad-CAM Hook: Detect.cv3[1][1]”.
- Label the hooked tensor “P4 Classification Feature, 64×40×40”.
- Draw a red dashed backward arrow from the preliminary class scores to a “Grad-CAM” block.
- Show the Grad-CAM heatmap and candidate boxes entering a block labeled “Maximum Heat-response Localization inside Each Candidate Box”.
- Then show “Crop 64×64 RGB Patch from Original Image”.
- Use orange dashed arrows for candidate selection, coordinate localization, and cropping to indicate non-differentiable operations.

3. Three-route Detail Encoder, shown as a large purple inset:
- Input: “N×3×64×64 Detail Crops”.
- First show “MaxPool2, 64×64→32×32”.
- Split into three parallel branches.
- Large Route: residual convolution stages 3→16→32→64→128 with spatial sizes 32×32→16×16→8×8→4×4, followed by Multi-Dilation Attention and Conv1×1 128→1, output 1×4×4.
- Medium Route: parallel main and shortcut downsampling, Conv6×6 stride 4 and MaxPool4 shortcut, then residual stages 16→32→64, output spatial size 4×4, followed by Multi-Dilation Attention and Conv1×1 64→1, output 1×4×4.
- Small Route: parallel Conv13×13 stride 8 and MaxPool8 shortcut, then residual 32→64, followed by Multi-Dilation Attention and Conv1×1 64→1, output 1×4×4.
- Inside each Multi-Dilation Attention block, show four parallel Conv3×3 paths with dilation rates 1, 3, 5, and 7, followed by Concat, Conv1×1, Sigmoid, and element-wise multiplication with the input feature.
- Merge the three 1×4×4 outputs using “Concat + BN + ReLU” to produce “Detail Feature, N×3×4×4”.

4. Spatial detail reconstruction:
- Show a green block labeled “Coordinate-aware Scatter-and-Add”.
- Illustrate multiple 3×4×4 candidate features being placed at their candidate-center positions on a zero P4 grid.
- Output label: “Sparse Detail Map, 3×40×40”.
- Preserve spatial positions; do not use global pooling or ROIAlign.

5. P4 detail fusion:
- Concatenate “P4 Classification Feature, 64×40×40” with “Sparse Detail Map, 3×40×40”.
- Label the concatenated tensor “67×40×40”.
- Pass it through “Conv3×3, 67→64 + BatchNorm + ReLU”.
- Output “Fused Feature, 64×40×40”.
- Pass it through the reused original P4 classifier “Conv1×1, 64→1”.
- Output “Detail-enhanced P4 Logit z_detail, 1×40×40”.
- In parallel, show the original P4 branch producing “Original P4 Logit z_base, 1×40×40”.

6. Learnable alpha logit interpolation:
- Draw a clear formula block before the classification sigmoid:
  z_out = (1 − alpha) z_base + alpha z_detail
  alpha = sigmoid(beta)
  beta is one globally learnable scalar, initialized so alpha = 0.5.
- Show two weighted arrows, z_base multiplied by (1−alpha) and z_detail multiplied by alpha, followed by an Add node.
- Label the output “Final P4 Classification Logit”.
- Merge it with unchanged P3 and P5 outputs, followed by “DFL Decode + Classification Sigmoid + NMS” and “Final Detections”.

Visual conventions:
- Blue: simplified YOLO components.
- Red: Grad-CAM hook and backward path.
- Orange dashed arrows: non-differentiable candidate localization and image cropping.
- Purple: Detail Encoder and multi-dilation attention.
- Green: scatter, feature concatenation, and logit fusion.
- Solid arrows: differentiable forward paths.
- Make tensor dimensions prominent but keep typography clean.
- Use only the specified modules and dimensions.

Important negative constraints:
- Do not draw a second full YOLO forward pass.
- Do not draw ROIAlign.
- Do not show the Detail Encoder producing a classification probability; it produces a 3×4×4 feature map per candidate.
- Do not show the detail branch modifying bounding-box regression.
- Do not put alpha after sigmoid; alpha mixes logits before sigmoid.
- Do not create per-candidate or per-pixel alpha values; alpha is one global scalar.
- Do not replace P3 or P5 logits in this single-head P4 model.
- Do not add Transformer, self-attention, cross-attention, FPN fusion, segmentation, or modules not described above.
```

## 18. 简化概览图 Prompt

如果只需要论文首页或答辩使用的简化图，可以使用：

```text
Draw a clean horizontal vector architecture diagram on a white background for a computer-vision paper. Show five major stages: (1) a simplified YOLOv8 detector producing P3, P4, and P5 outputs; (2) P4 Grad-CAM plus confidence-based candidate boxes generating 64×64 detail crops; (3) a three-route Detail Encoder with large, medium, and small receptive-field branches and multi-dilation attention rates 1, 3, 5, and 7, producing a 3×4×4 feature per candidate; (4) coordinate-aware scatter-and-add producing a sparse 3×40×40 P4 detail map; and (5) P4 feature fusion by concatenating 64×40×40 YOLO features with the 3×40×40 detail map, followed by Conv3×3 67→64, BatchNorm, ReLU, the original Conv1×1 classifier, and learnable pre-sigmoid logit interpolation z_out=(1−alpha)z_base+alpha*z_detail, alpha=sigmoid(beta). Use solid arrows for differentiable paths, red dashed arrows for Grad-CAM backward, orange dashed arrows for non-differentiable localization and cropping. Keep YOLO compact and make the Detail Encoder and fusion module visually dominant. Do not use ROIAlign, do not add a second full YOLO forward, and do not show the detail branch modifying box regression.
```

## 19. 对应代码

- P4 Hook、融合与可学习 alpha：`experiments/run_318_fusion_alpha.py`；
- Detail Encoder：`Resnet6.py` 中的 `Detail_Net_attn_block`；
- Detail 前向兼容修正：`experiments/eval_best318_legacy.py` 中的 `patch_legacy_detail_forward`；
- 三尺度 YOLO 配置：`ultralytics/models/v8/yolov8.yaml`；
- 指标和权重：[Grad-CAM Detect Head 实验结果](GRADCAM_DETECT_HEAD_EXPERIMENT_RESULTS.md)。
