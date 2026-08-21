# BGD-YOLO 全局—局部特征融合：第一版方法设计

> 文档状态：V1 代码已实现并通过静态检查，尚未运行训练/验证/前向测试
> 对应项目入口：`train.py`
> 当前主要实现：`gradcam_fusion.py::GradCAMGlobalLocalYOLO`、`global_local_fusion.py`
> 局部细节网络：`Resnet6.py::Detail_Net_attn` / `Detail_Net_attn_block`
> 具体代码改动：`experiments/GRADCAM_GLOBAL_LOCAL_V1_CODE_CHANGES.md`

## 1. 本文档的目的

本文档只确定下一阶段代码修改的方法、边界和验收条件，不在本次文档提交中修改训练或推理逻辑。

第一版的核心目标是把当前的“YOLO 分数与 Detail 分数线性平均”改成真正的空间特征融合：

```text
整图输入
  ↓
YOLO 第一次前向：得到候选框和 P3/P4/P5 全图特征
  ↓
对候选框做 NMS + Top-K
  ↓
Grad-CAM 在候选框内寻找最大响应点
  ↓
从同一张模型输入图中裁剪局部区域
  ↓
Detail 编码器提取局部特征
  ↓
将局部特征按原空间位置回填到 P3 尺度的空特征图
  ↓
门控残差融合全局特征与局部特征
  ↓
YOLO Detect Head 第二次预测
```

## 2. 当前实现与目标实现的区别

### 2.1 当前代码实际做法

当前 `val.py` 的主路径大致为：

1. YOLO 输出候选 anchor；
2. 根据阈值筛选 anchor；
3. 在候选框左上、右上、左下、右下和中心裁剪 5 个 `64×64` patch；
4. `Detail_Net_attn` 为每个 patch 输出一个 logit；
5. 对 5 个 logit 求均值并执行一次 Sigmoid；
6. 使用固定系数对 YOLO 分数与 Detail 分数进行线性平均；
7. 训练时再尝试把融合概率写回 YOLO raw logits。

因此，当前代码属于“候选框局部分类分数融合”，不是“全局—局部特征图融合”。虽然 `val.py` 导入了 Grad-CAM 相关模块，但当前主路径没有使用 Grad-CAM 来决定裁剪位置。

### 2.2 第一版目标做法

第一版不再把单个局部 patch 特征直接与整图特征拼接，也不再以固定 `0.5` 比例平均两个模型的最终概率。局部特征必须先恢复到全图特征坐标系，形成 `L_canvas`，然后再与 YOLO 的 `G_full` 融合。

## 3. 第一版采用的总体方案

第一版采用“P3 单尺度稀疏回填 + 门控残差融合 + Detect 二次预测”。

选择 P3 的原因：

- P3 的 stride 通常为 8，保留的空间细节最多；
- `64×64` 图像区域在 P3 上大约对应 `8×8` 个位置，适合放置局部纹理特征；
- 第一版只融合一个尺度，容易验证坐标、显存和收益；
- P3 方案稳定后，再考虑同时融合 P4/P5。

第一版暂不把 Grad-CAM 当作可微训练模块。Grad-CAM 只负责产生裁剪中心，中心坐标在进入裁剪步骤前 `detach`。检测损失仍然能够训练 Detail 编码器、回填投影层、门控层以及按阶段解冻的 YOLO 层，但不会穿过 `argmax` 反向传播到“选点过程”。

## 4. 张量定义与空间对齐

设输入图像张量为：

```text
I: [B, 3, H_img, W_img]
```

YOLO 第一次前向在进入 Detect Head 前得到：

```text
G3: [B, C3, H3, W3]    # P3，stride≈8
G4: [B, C4, H4, W4]    # P4，stride≈16
G5: [B, C5, H5, W5]    # P5，stride≈32
```

对于 `640×640` 输入，P3 通常约为：

```text
G3: [B, C3, 80, 80]
```

Detail 编码器接收局部图像：

```text
patches: [N, 3, 64, 64]
```

现有 `Detail_Net_attn_block` 因为开头还有一次 `2×2` 池化，实际输出预计为：

```text
L_patch: [N, 3, 4, 4]
```

代码注释中部分位置写成 `8×8`，第一版实现前必须用单元测试确认真实尺寸，不能依赖旧注释。`L_patch` 会根据裁剪区域映射到 P3 后的实际大小进行双线性插值，再贴入局部特征画布。

## 5. 候选框生成与数量控制

第一次 YOLO 前向得到预测后，按以下顺序选择 Detail 候选框：

1. 使用较低的第一阶段置信度阈值保留候选，避免过早损失召回；
2. 解码候选框；
3. 执行 class-aware NMS；
4. 按置信度选择每张图最多 `K` 个候选；
5. 保存候选框与其原始 anchor/类别分数的对应关系，供 Grad-CAM 构造 target。

第一版建议默认值：

```yaml
candidate_conf: 0.15
candidate_iou: 0.60
candidate_topk: 5
```

这些参数必须进入配置，不能继续写死在 `_predict_once()` 内。

必须先 NMS 再运行 Detail 和 Grad-CAM，否则相邻 anchor 会围绕同一目标产生大量重复裁剪，显著增加时间和显存。

## 6. Grad-CAM 局部区域定位

### 6.1 Grad-CAM 的输入与目标

第一版对每个保留候选框构造一个目标分数：

```text
target_score = 对应候选框的类别分数
```

目标层优先选择 P3 或进入 Detect Head 前的 P3 特征。对目标分数关于目标特征图求梯度：

```text
weights = GAP(∂target_score / ∂G3)
CAM = ReLU(sum(weights * G3, channel))
```

CAM 归一化后，只在候选框映射到 CAM 的有效区域内寻找最大值：

```text
center = argmax(CAM inside candidate box)
```

不能直接在整张 CAM 上找最大值，否则可能选到当前候选框以外的另一个目标。

### 6.2 异常回退

出现以下情况时使用候选框几何中心：

- CAM 包含 NaN/Inf；
- CAM 全部为零或近似常数；
- 候选框映射后为空；
- 目标层无法取得梯度；
- Grad-CAM 功能被配置关闭。

保留 `box_center` 模式作为必要基线，用于验证性能提升究竟来自 Grad-CAM，还是仅仅来自 Detail 网络和二次预测。

### 6.3 计算限制

标准 Grad-CAM 需要梯度。第一版必须使用 `torch.autograd.grad()` 获取所需梯度，不能在模型 forward 内调用普通 `loss.backward()`，也不能随意 `zero_grad()` 清除正常训练梯度。

每个候选框单独计算 Grad-CAM 的代价较高，因此第一版限制 `candidate_topk`，并记录 Grad-CAM 耗时和显存。若速度不可接受，后续版本再比较：

- 一张图将多个候选分数求和后生成一张 CAM；
- 无反向传播的 feature-response map；
- ROIAlign + 可学习注意力定位；
- 离线预计算 CAM 中心。

## 7. 局部裁剪规则

### 7.1 训练、验证和推理必须使用同一坐标系

局部裁剪统一从当前模型真正接收的 `batch['img']` 张量产生，不能在训练时根据 `im_file` 重新读取未增强的原图。

原因是训练图像可能已经执行 Mosaic、随机缩放、平移和翻转；第一次 YOLO 预测框属于增强后图像坐标。如果再从磁盘原图裁剪，预测框与局部 patch 会错位。

### 7.2 裁剪方式

第一版默认以 Grad-CAM 最大响应点为中心裁剪固定 `64×64` 区域：

- 边界外使用 padding；
- 使用 tensor 操作完成；
- 使用与 Detail 预训练一致的 RGB 和 mean/std 归一化；
- 保存裁剪框坐标，供后续回填；
- 验证可视化时再把坐标转换回原图，不参与模型计算。

后续消融可比较固定裁剪、相对候选框尺度裁剪和双尺度裁剪，但第一版不同时引入这些变量。

## 8. Detail 特征提取

第一版使用 `Detail_Net_attn_block` 作为特征编码器，而不是使用 `Detail_Net_attn` 的最终二分类 logit。

现有 `Detail_Net_attn_block` 将大、中、小三个分支各压缩为 1 个通道，再拼接为局部特征：

```text
L_patch: [N, 3, 4, 4]
```

这样可以复用原有多尺度 Detail 结构，同时让后续融合层决定如何将 3 通道局部响应映射到 YOLO 的通道空间。

第一版修改时必须先修复 `Detail_Net_attn_block.forward()` 小分支中的变量错误：小分支卷积后当前代码错误地对 `x_M` 执行 ReLU，正确对象应为 `x_S`。还需要增加输出 shape、有限值和梯度测试。

Detail 权重加载必须打印并检查 missing/unexpected keys。允许从 `Detail_Net_attn` 的预训练权重加载名称和形状一致的编码层，但不能继续用 `strict=False` 静默忽略所有不匹配。

## 9. 局部特征回填

### 9.1 建立画布

在 P3 坐标系建立：

```text
L_sum:  [B, 3, H3, W3]
Count:  [B, 1, H3, W3]
Mask:   [B, 1, H3, W3]
```

对每个局部 patch：

1. 将裁剪区域从输入图坐标映射到 P3 坐标；
2. 根据映射后的宽高插值 `L_patch`；
3. 累加到 `L_sum` 的对应区域；
4. 在 `Count` 对应区域加 1。

全部候选处理完成后：

```python
L_canvas = L_sum / Count.clamp_min(1)
Mask = (Count > 0).to(L_canvas.dtype)
```

使用平均而不是后一个 patch 覆盖前一个 patch，避免结果依赖候选框处理顺序。

### 9.2 为什么必须有 Mask

未回填区域的零值表示“该位置没有局部特征”，而 Detail 编码器也可能真的输出零。加入 `Mask` 后，门控模块能够区分这两种含义，并确保无局部信息区域尽量保持原 YOLO 特征。

## 10. 门控残差融合

先把 3 通道局部画布投影到 P3 的通道数：

```python
L = local_proj(L_canvas)                  # [B, C3, H3, W3]
gate_input = torch.cat([G3, L, Mask], 1) # [B, 2*C3+1, H3, W3]
gate = torch.sigmoid(gate_conv(gate_input))
F3 = G3 + alpha * gate * L
```

其中：

- `local_proj`：`1×1 Conv + Norm + Activation`；
- `gate_conv`：`3×3 Conv`，输出 `C3` 通道门控；
- `gate`：同时具有空间和通道选择能力；
- `alpha`：可学习标量，第一版建议初始化为 `0.01`；
- `F3`：融合后的 P3 全图特征。

`alpha` 使用很小的初始值，是为了让训练开始时模型行为接近已训练好的 YOLO，避免随机初始化的局部投影和门控立即破坏全局特征。

这里的 Sigmoid 只生成 `0～1` 门控权重，与分类输出 Sigmoid 不同，不构成重复 Sigmoid。

无局部特征的位置需要显式保持：

```python
F3 = G3 + alpha * gate * L * Mask
```

或者在 `local_proj` 前后再次应用 Mask，防止投影层 bias 在空白区域产生非零响应。

## 11. 第二次 Detect 预测

第一次前向保存 Detect Head 的输入列表：

```text
[G3, G4, G5]
```

融合后复用同一个 Detect Head：

```python
second_features = [F3, G4, G5]
pred_second = detect_head([f.clone() for f in second_features])
```

第一版最终输出第二次预测结果。第一次预测只负责产生候选框和 Grad-CAM target，不再用固定 `0.5` 权重与 Detail 分类概率做最终平均。

复用同一个 Detect Head 可以控制参数量，但需要避免其 forward 对输入列表的原地修改，因此每次调用应传入新的列表或 clone。

## 12. 第一版训练方案

### 阶段 0：建立可靠基线

- 固定同一数据划分；
- 保存原始 YOLO 指标和速度；
- 保存当前“5 点裁剪 + 分数平均”指标作为旧方法对照；
- 确认 Detail 预训练权重的类别含义、归一化和输入尺寸。

### 阶段 1：只训练新增融合路径

冻结：

- YOLO backbone/neck；
- YOLO Detect Head；
- 可选：先冻结 Detail 编码器前部。

训练：

- `Detail_Net_attn_block` 的可训练部分；
- `local_proj`；
- `gate_conv`；
- `alpha`。

训练损失使用第二次预测的原 YOLO detection loss。第一阶段不再使用当前 `ratio → probability → logit` 的替换逻辑。

### 阶段 2：小学习率联合微调

当阶段 1 能稳定收敛后，解冻：

- P3 相关 neck 后部层；
- Detect Head；
- Detail 编码器后部层。

建议新增模块学习率高于已训练 YOLO 层，YOLO 层使用较小学习率。第一版代码必须打印实际可训练参数名称和数量，若没有匹配到任何参数则立即报错。

### 阶段 3：可选全局微调

只有阶段 2 在验证集上稳定优于基线时才考虑全模型小学习率微调。阶段 3 不属于第一版必须完成的范围。

## 13. 损失设计

第一版主损失：

```text
L_total = L_detect_second
```

如果后续发现第一次预测在联合微调时退化，可加入辅助损失：

```text
L_total = L_detect_second + λ_first * L_detect_first
```

第一版暂不增加独立 Detail 分类损失，避免同时改变过多变量。如果后续加入 Detail 分类辅助头，必须使用 `BCEWithLogitsLoss` 或正确实现的 focal-with-logits，并明确全流程中 logit 与 probability 的边界。

## 14. 计划修改的文件

以下是文档确认后第一版预计修改的文件。实际修改前仍会再次检查工作区状态。

| 文件 | 计划修改 |
|---|---|
| `val.py` | 拆分第一次 YOLO 前向、候选筛选、Grad-CAM 定位、tensor 裁剪、回填、融合和第二次 Detect；移除融合模式下的概率替换逻辑 |
| `Resnet6.py` | 修复 `Detail_Net_attn_block` 小分支变量错误；明确特征输出接口与 shape |
| `train.py` | 改成清晰的第一版训练入口；参数集中配置；去除机器相关的硬编码路径或改成命令行参数 |
| `ultralytics/yolo/engine/trainer.py` | 安全保存 Detail/fusion 模块和配置；普通 YOLO 不具有这些模块时不能崩溃 |
| 新增 `global_local_fusion.py` | 放置回填、Mask、局部投影和门控残差融合模块，避免继续把全部逻辑堆在 `val.py` |
| 新增测试文件 | 测试 Detail shape、坐标映射、重叠平均、空候选、梯度和第二次 Detect 输入输出 |
| `.gitignore` | 忽略数据集、权重、运行结果、CAM 缓存和可视化文件，不上传数据集 |

不计划修改或上传数据集。

## 15. 第一版实现顺序

第一版代码按以下小步骤实现，每一步都可单独检查：

1. 修复 `Detail_Net_attn_block` 并验证输出 shape；
2. 实现独立的 `GlobalLocalFusion` 模块；
3. 实现局部特征回填和重叠平均单元测试；
4. 在不启用 Grad-CAM 时，用候选框中心完成整条两次预测链路；
5. 加入 Grad-CAM 最大响应点定位及异常回退；
6. 删除/绕开融合模式中的旧分数替换训练逻辑；
7. 实现阶段 1 冻结策略并打印可训练参数；
8. 跑最小 batch 前向、反向和 checkpoint 保存测试；
9. 跑验证集并与纯 YOLO、旧分数融合进行对照；
10. 确认后提交并推送 GitHub。

先用 `box_center` 跑通不是改变最终方法，而是为了将“特征融合是否正确”和“Grad-CAM 是否正确”两个问题分开验证。最终实验仍包含 Grad-CAM 模式。

## 16. 必须通过的测试

### 16.1 单元测试

- `Detail_Net_attn_block([N,3,64,64])` 输出尺寸与文档一致；
- 无候选框时 `F3` 与 `G3` 完全一致；
- 单个 patch 能贴到正确 P3 区域；
- 两个 patch 重叠时取平均，不依赖处理顺序；
- 边界 patch 不越界；
- `Mask=0` 区域不改变全局特征；
- `gate` 数值位于 `[0,1]`；
- detection loss 能给 `local_proj`、`gate_conv` 和 Detail 编码器产生有限梯度；
- 普通 YOLO 在没有 Detail/fusion 模块时仍能训练和保存。

### 16.2 最小集成测试

- batch=1 的训练 forward/backward 成功；
- batch>1 时候选框不会串到其他图片；
- 训练、验证、predict 三条入口使用同一裁剪和融合逻辑；
- 空候选、单候选、Top-K 候选都能运行；
- checkpoint 保存后可严格恢复新增模块；
- CPU 与 CUDA 至少完成一次最小前向（环境支持时）。

### 16.3 可视化检查

每张抽样图片保存：

- 第一次 YOLO 候选框；
- 候选框内 Grad-CAM 热力图；
- Grad-CAM 最大点；
- 实际 `64×64` 裁剪区域；
- P3 上的回填 Mask；
- 门控图均值或热力图；
- 第二次 YOLO 结果。

这些可视化只用于调试，默认不提交 Git。

## 17. 对照实验与消融

第一版至少保留以下开关：

| 实验 | 目的 |
|---|---|
| 纯 YOLO | 基线 |
| YOLO + 旧 5 点 Detail 分数融合 | 与当前实现对照 |
| YOLO + 中心裁剪 + 回填特征融合 | 验证特征融合本身 |
| YOLO + Grad-CAM 裁剪 + 回填特征融合 | 验证 Grad-CAM 的额外贡献 |
| 融合无 gate / 有 gate | 验证门控作用 |
| `alpha=1` / 可学习小 `alpha` | 验证稳定性设计 |

报告指标至少包括：

- Precision、Recall；
- AP50、AP75、mAP50-95；
- FP/image、FN/image；
- 第一阶段候选数/image；
- Grad-CAM、Detail、融合和总推理时间；
- 峰值 GPU 显存。

## 18. 第一版明确不做的内容

为避免第一版范围失控，以下内容暂不同时实现：

- 同时在 P3、P4、P5 三个尺度融合；
- 多个不同尺寸的局部 patch；
- Transformer/Cross-Attention 融合；
- Grad-CAM 选点过程的高阶可微训练；
- ONNX/TensorRT 导出；
- 数据集重构和自动生成全部 Detail 数据；
- 全模型从头训练。

这些内容可在第一版正确性和收益得到验证后逐项加入。

## 19. 主要风险与处理方式

### 风险 1：Grad-CAM 速度和显存开销过大

处理：NMS 后只保留 Top-K；记录独立耗时；必要时将训练定位替换为缓存 CAM 或 feature-response，推理仍保留 Grad-CAM 对照。

### 风险 2：Grad-CAM 最大点不等于缺陷细节

处理：只在候选框内找峰值；保存可视化；与几何中心进行消融，不预设 Grad-CAM 一定更好。

### 风险 3：回填坐标错误

处理：统一使用模型输入 tensor 坐标；先用人工构造 patch 和 Mask 测试；训练逻辑禁止重新读取原图裁剪。

### 风险 4：局部噪声破坏 YOLO 特征

处理：残差保留 `G3`；Mask 限制注入区域；`alpha` 小值初始化；先冻结 YOLO 训练新增模块。

### 风险 5：第一阶段漏检无法被 Detail 恢复

处理：降低候选阈值并用 NMS/Top-K 控制预算。该级联系统主要改善误检与候选质量，不能保证恢复 YOLO 完全没有提出的目标。

### 风险 6：旧 Detail 权重与特征编码器不完全匹配

处理：显式报告所有 missing/unexpected keys；只加载名称和 shape 一致的编码层；其余层重新初始化并进行阶段 1 训练。

## 20. 第一版验收标准

代码层面满足以下条件才算第一版完成：

1. 主路径真正调用 Grad-CAM 或明确配置的中心回退模式；
2. 局部特征先回填为完整 P3 画布，再与 `G3` 融合；
3. 不存在 `G_full` 与单个 `L_patch` 直接 `cat`；
4. 最终检测来自融合特征的第二次 Detect，而不是固定概率平均；
5. 训练裁剪与增强后的输入图严格对齐；
6. 训练不再使用 `ratio` 替换 raw class probability 的旧逻辑；
7. 新增模块能够获得有限非零梯度；
8. checkpoint 可以保存和恢复 YOLO、Detail 编码器、融合模块与配置；
9. 数据集、权重和运行结果不会上传 GitHub；
10. 完成最小测试和至少一组基线对照后，再提交并推送到 `zing-c/BGD-Yolo-v2`。

效果层面不预先承诺 mAP 一定提高。第一版首先验收方法是否被真实、正确地调用，再依据相同测试集上的对照实验决定是否保留 Grad-CAM 和当前 Detail 结构。

## 21. 第一版核心伪代码

```python
# ----- Pass 1: YOLO global detection -----
first_raw, (G3, G4, G5) = yolo_forward_before_detect(images)
first_pred = detect_head([G3.clone(), G4.clone(), G5.clone()])

candidates = select_candidates(
    first_pred,
    conf=candidate_conf,
    iou=candidate_iou,
    topk=candidate_topk,
)

# ----- Grad-CAM localization and local encoding -----
patches = []
crop_boxes = []
for candidate in candidates:
    cam = gradcam_for_candidate(candidate.score, G3)
    center = max_point_inside_box(cam.detach(), candidate.box)
    center = fallback_to_box_center_if_invalid(center, candidate.box)

    patch, crop_box = crop_from_current_input_tensor(
        images,
        center=center,
        size=64,
    )
    patches.append(patch)
    crop_boxes.append(crop_box)

L_patch = detail_encoder(torch.stack(patches))

# ----- Sparse full-map reconstruction -----
L_canvas, mask = paste_and_average_on_p3(
    L_patch,
    crop_boxes,
    image_size=images.shape[-2:],
    p3_size=G3.shape[-2:],
)

# ----- Gated residual fusion -----
L = local_proj(L_canvas)
L = L * mask
gate = torch.sigmoid(gate_conv(torch.cat([G3, L, mask], dim=1)))
F3 = G3 + alpha * gate * L

# ----- Pass 2: final detection -----
second_raw = detect_head([F3.clone(), G4.clone(), G5.clone()])
return second_raw
```

## 22. 最终方法表述建议

如果第一版实现和实验通过，论文或报告中的方法可以表述为：

> 首先利用 YOLO 对整幅图像进行初步检测并提取多尺度全局特征；随后针对 NMS 后的候选目标，在候选框约束范围内利用 Grad-CAM 定位最具判别性的局部响应区域，并通过多分支注意力细节编码器提取局部纹理特征；之后将局部特征依据其原始空间坐标稀疏回填到 YOLO 的 P3 特征尺度，通过带有效区域掩码的门控残差模块完成全局语义与局部细节融合；最后将融合特征再次送入检测头，获得细化后的检测结果。

这个表述对应的是实际空间特征融合，而不只是两个分类概率的后处理平均。
