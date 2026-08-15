# BGD-Yolo 项目结构、实现链路与代码/方法审查报告

> 审查日期：2026-08-15  
> 审查范围：当前工作区 `/home/user/projects/czy/BGD-Yolo_v2_copy`  
> 目标方法：YOLO 初步检测 → Grad-CAM/特征响应定位 → 局部裁剪 → `Detail_Net_attn` 二次判断

## 1. 结论摘要

这个项目的研究方向是合理的：先用 YOLO 保证目标级召回，再用专门的局部纹理模型复核透明、反光或细粒度目标，可以用于降低误检并增强细节判别。不过，**当前新版主代码和上述方法描述并不一致，而且存在数个会直接影响训练正确性和指标可信度的严重问题**。

最重要的结论如下：

1. 当前实际主链路是 `train.py → val.py::BGD_YOLO → Detail_Net_attn`。新版 `val.py` 没有真正运行 Grad-CAM；它从每个 YOLO 候选框固定裁剪左上、右上、左下、右下、中心 5 个 64×64 patch。因此当前方法应称为“YOLO 候选框内五点局部复核”，而不是“Grad-CAM 定位”。真正使用 Grad-CAM 的是较早的 `BGD-Yolo-Detect.py`。
2. 联合训练时把 YOLO 原始输出的前 4 个通道当作全部框回归通道，但 YOLOv8 DFL 框回归实际占 `4 × reg_max = 64` 个通道。代码会错误改写另外 60 个框分布通道，直接污染框回归训练。
3. `Detail_Net_attn` 返回 logits，但独立训练的 Focal Loss 把它当成概率做 clamp + BCE；验证又直接使用 `logit > 0.5`。损失与阈值定义均不一致。
4. `BGD_YOLO.val()` 的预处理在本次复核的当前版本中仅除以 255 一次；此前审查时看到的重复归一化现已修复。
5. 联合训练在增强后的 YOLO batch 上预测，却重新从磁盘读取未增强原图裁剪细节 patch。开启 Mosaic、翻转、缩放等增强时，预测框坐标与裁剪图像不是同一个视觉样本。
6. 常规 `predict()` 没有给模型提供当前 batch 元数据，因此细节分支通常不会执行；目前缺少可靠的端到端推理入口。
7. 细节数据集存在重复与泄漏：验证集内部有 25 组完全相同的图像，另有 9 张验证图与训练图内容完全相同。

在修复以上问题之前，不建议直接用当前联合训练结果证明方法优于 YOLO 基线。

## 2. 项目结构与各部分职责

### 2.1 总体结构

项目不是一个干净的单一应用包，而是：

- 一份修改过的 Ultralytics YOLOv8 源码副本；
- 多代 Grad-CAM、细节网络和联合训练实验脚本；
- 本地细节 patch 数据集、标签、模型权重；
- 大量历史训练输出、可视化图片和 W&B 日志。

主要目录/文件如下：

```text
BGD-Yolo_v2_copy/
├── ultralytics/                  # 修改过的 YOLOv8（日志显示版本 8.0.114）
│   ├── nn/tasks.py               # YOLO 前向、模型解析；保存 batch 供自定义前向使用
│   └── yolo/
│       ├── data/base.py          # 数据读取，被修改为额外读取 ori_img
│       ├── engine/trainer.py     # checkpoint 中额外保存 detail_model
│       └── utils/loss.py         # YOLOv8 检测损失
├── val.py                        # 当前新版 BGD_YOLO 核心整合实现
├── train.py                      # 当前单阶段联合训练入口
├── train_v2.py                   # 两阶段冻结/解冻实验和验证入口
├── BGD-Yolo-Detect.py            # 较早的 Grad-CAM + 单局部 patch + Detail 实现
├── gradcam.py                    # Grad-CAM、热力图选点、数据生成工具
├── gradcam_negative_samples.py   # Grad-CAM/特征响应负样本生成实验
├── Resnet6.py                    # ResNet6、Detail_Net、Detail_Net_attn 等多个结构
├── Resnet_Train.py               # 当前细节模型独立训练入口
├── Resnet_val.py                 # 细节模型验证、阈值/温度、错误样本分析
├── detailModel/                  # 其他 Inception/Detail 模型实验
├── BGD_image/                    # 64×64 细节 patch 数据
├── BGD_lable/                    # CSV 二分类标签
├── run/                          # 细节模型权重和错误分析
├── yolo-runs/                    # 大量 YOLO/BGD 历史实验
├── wandb/                        # 本地 W&B 日志
└── visualize_tp_fp_fn/           # TP/FP/FN 可视化结果
```

### 2.2 建议认定的“当前主代码”

- YOLO + Detail 训练：`train.py`
- BGD 整合类：`val.py` 中的 `BGD_YOLO`
- Detail 模型定义：`Resnet6.py` 中的 `Detail_Net_attn`
- Detail 独立训练：`Resnet_Train.py`
- Detail 独立评估：`Resnet_val.py`
- 旧版 Grad-CAM 方案：`BGD-Yolo-Detect.py`

`BGD_detect.py`、`BGD_val.py`、`val_v1.py`、`dataLoader.py` 等更像中间版本或未完成原型，不应继续与主实现并行维护。

## 3. 项目当前实际是如何工作的

### 3.1 细节模型数据准备

细节模型使用 CSV：

- 训练：`BGD_lable/train/mixed_lable.csv` + `BGD_image/mixed_data`
- 验证：`BGD_lable/val/lable.csv` + `BGD_image/val`

所有图像都是 64×64 RGB patch。训练有效标签共 1676 条：

- 正样本 1198（71.48%）
- 负样本 478（28.52%）

验证共 616 条：

- 正样本 355（57.63%）
- 负样本 261（42.37%）

训练/验证类别比例有明显变化。`gradcam.py` 的正负标签逻辑实际判断“裁剪中心点是否落在 GT 框内”，不是注释所写的“裁剪区域与 GT 框是否相交”。`gradcam_negative_samples.py` 的一部分逻辑直接把生成 patch 标为 0，需要保证输入确实来自无缺陷图，否则会制造假负样本。

### 3.2 `Detail_Net_attn` 结构

模型输入为 `(B, 3, 64, 64)`，当前权重对应 4,495,185 个参数。结构大致为：

1. 首先执行 2×2 MaxPool，把输入变为 32×32。
2. 三条并行尺度分支：
   - Large：多层 3×3 残差卷积和池化，输出约 `(B,128,4,4)`；
   - Medium：大步长卷积后继续残差卷积，输出约 `(B,64,4,4)`；
   - Small：13×13、stride 8 的卷积快速获取大感受野，输出约 `(B,64,4,4)`。
3. 每个分支使用 `detail_atten`：分别做 dilation 1/3/5/7 的 3×3 卷积，拼接后用 1×1 卷积和 Sigmoid 生成空间/通道门控，再与原特征相乘。
4. 三个分支展平并拼接为 4096 维，经过 `4096 → 512 → 1` 的全连接层输出一个 logit。

这个模型本质上是“多尺度残差 CNN + 多膨胀率门控注意力”，并不是标准 ResNet。代码中还保留了多个相近变体，容易出现“权重对应哪个类”不清楚的问题。

### 3.3 旧版 Grad-CAM 链路

`BGD-Yolo-Detect.py` 的流程与原始方法描述基本一致：

```text
YOLO 前向
  → 对指定层注册 activation/gradient hook
  → 对检测输出反向传播并生成 Grad-CAM
  → 在候选框内寻找响应最高的 64×64 窗口中心
  → 映射回原图并裁剪一个局部 patch
  → Detail_Net_attn 复核
  → 使用 sqrt(YOLO分数 × Detail分数) 融合
```

该实现主要限制是 batch 处理不完整、需要反向传播、速度慢、部署困难，并且旧代码里 Detail 输出的概率/logit语义也发生过变化。

### 3.4 新版实际链路（`val.py`）

当前 `train.py` 调用的新版流程是：

```text
YOLO 输出所有尺度的 raw feature maps
  → 解码为全部 anchor 的 xywh + class probability
  → 在 NMS 之前保留 class score > conf_threshold 的 anchor
  → 将预测框映射回磁盘原图
  → 每个框固定取 5 个 64×64 patch
       左上 / 右上 / 左下 / 右下 / 中心
  → ImageNet mean/std 归一化
  → Detail_Net_attn 分别输出 5 个 logits
  → 对 5 个 logits 求均值，再 Sigmoid 得到 detail score
  → 验证时：0.5 × YOLO score + 0.5 × detail score
  → 训练时：尝试把 raw class logits 替换为 detail score，再使用 YOLO 分类损失反传
```

如果预测框宽或高小于 64，代码把同一个中心 patch 重复 5 次。Grad-CAM 的 hook 注册在新版中被注释，所以 `target_layers`、`cam_mode` 等参数当前基本不产生效果。

## 4. 代码问题清单

### 4.1 P0：必须先修复的问题

#### P0-1：联合训练错误修改 YOLOv8 的 DFL 框回归通道

位置：`val.py:772-794`

YOLOv8 raw head 的通道顺序是：

```text
[4 × reg_max 个 bbox distribution logits] + [nc 个 class logits]
```

当前 `reg_max=16` 时，bbox 部分是前 64 个通道。但代码写成：

```python
bbox_logits = raw_feat[:, :4, ...]
cls_logits = raw_feat[:, 4:, ...]
```

这会把第 4～63 通道的 60 个框分布 logits 当成类别 logits 做 Sigmoid、比例缩放和 Logit 逆变换，框回归损失因此被污染。应使用 `box_channels = m.reg_max * 4`，从该位置切分。

#### P0-2：Detail 独立训练的损失把 logits 当成概率

位置：`Resnet_Train.py:38-60`、`Resnet6.py:456-463`

`Detail_Net_attn` 最后一层没有 Sigmoid，输出是 logits；但 `FocalLoss` 先把输出 clamp 到 `[1e-6, 1-1e-6]`，再调用 `binary_cross_entropy`。负 logits 会被压到极小正数，大于 1 的 logits 会被压到接近 1，clamp 区域梯度为 0。

正确方案：使用 `binary_cross_entropy_with_logits` 或 `BCEWithLogitsLoss` 实现 focal loss；训练和验证都明确约定模型输出为 logits。

#### P0-3：Detail 验证阈值错误

位置：`Resnet_Train.py:97`、`Resnet_val.py:213-230`

logit 的 0.5 概率对应阈值应为 0，而不是 0.5；或者先 `sigmoid(logits)` 再与 0.5 比较。

直接加载现有 `exp3_4_1.pt` 重算，清除验证内部重复和训练泄漏后（n=582）：

| 判定方式 | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| 现代码：logit > 0.5 | 91.24% | 94.03% | 90.33% | 92.14% | 96.35% |
| 正确默认：logit > 0 | 91.92% | 91.04% | 95.17% | 93.06% | 96.35% |

这说明权重有判别能力，但当前阈值偏向 Precision、牺牲了 Recall。最终阈值应在独立验证集上按业务代价校准，而不是固定采用 0 或 0.5。

#### 已修复：外部 BGD 验证预处理曾重复除以 255

当前复核位置：`val.py:1207-1211`

当前代码只执行一次 `/255`，这一问题已经不存在。仍建议为预处理增加输入范围断言，防止不同训练/验证入口以后再次产生不一致。

#### P0-5：训练预测图与 Detail 裁剪图不一致

位置：`val.py:478-588`

YOLO 接收的是经过 Mosaic、随机缩放、平移、翻转、HSV 等增强后的 `batch['img']`；Detail 分支却根据 `im_file` 再次读取磁盘原图。预测框来自增强图，patch 来自原图，二者在训练时通常无法对齐。当前日志显示 `mosaic=1.0`、`fliplr=0.5`、`scale=0.5`、`translate=0.1`，因此不是边缘情况。

训练时必须从同一个增强后的 tensor 裁剪，推荐使用 `torchvision.ops.roi_align` 或 `grid_sample`；验证/推理也尽量走同一条预处理路径。

#### P0-6：普通 `predict()` 通常不会执行 Detail 分支

当前 Detail 逻辑依赖 `self.batch`/`model.batch` 中的 `im_file`、`ratio_pad` 等字段。验证器显式设置了 `model.batch`，训练时 `BaseModel.forward(dict)` 也会保存 batch；常规 predictor 没有这一操作。因此 `BGD_YOLO.predict()` 很可能退化成纯 YOLO，或者在先验证后预测时误用残留 batch。

应提供一个明确的 `predict`/`postprocess` 实现：接收当前原图、letterbox 元数据、NMS 后候选框，完成裁剪、Detail 推理、分数融合和最终 NMS。

#### P0-7：`detail_mode=False` 会访问未定义的 `self.detail_model`

位置：`val.py:81-140`

`self.detail_model` 只在 `if self.detail_mode:` 中创建，但后面无条件执行：

```python
self.model.detail_model = self.detail_model
```

因此关闭细节模式会触发 `AttributeError`。应在构造开始时设为 `None`，并只在启用时注册。

#### P0-8：全局 Trainer 保存逻辑破坏普通 YOLO 训练

位置：`ultralytics/yolo/engine/trainer.py:428-455`

`save_model()` 无条件读取 `self.model.detail_model.state_dict()`。任何没有 Detail 分支的普通 YOLO 模型在保存 checkpoint 时都会失败。必须用能力判断，或为 BGD 建立独立 Trainer，不能全局修改基类并假设所有模型都有 Detail。

### 4.2 P1：高优先级设计/工程问题

#### P1-1：训练融合与推理融合不一致

验证时使用 `0.5 × detail + 0.5 × YOLO`，联合训练写回 raw logits 时却几乎完全替换为 Detail 分数。训练目标和推理公式不一致；而且 `ratio = detail / old` 与 `current_prob × ratio` 会让 YOLO 分类梯度接近抵消，选中 anchor 上主要只训练 Detail 分支。

建议显式定义可微融合层，例如：

```text
fused_logit = yolo_logit + β × detail_logit
```

或让一个很小的 MLP 学习 `(yolo_logit, detail_logit, box_size)` 的融合，并使用同一公式完成训练和推理。

#### P1-2：在 NMS 前处理大量重复 anchor，验证时还没有上限

位置：`val.py:461-472`

当前会处理所有超过阈值的 raw anchors。相邻 anchors 往往对应同一目标，导致重复磁盘读取/裁剪/Detail 推理。训练只在超过 100 个时 Top-K，验证没有限制，`detail_max_boxes` 又没有真正使用。

推荐流程：低阈值解码 → class-aware NMS/Top-K → 对保留框执行 Detail → 重打分 → 必要时第二次 NMS。

#### P1-3：固定 64 像素和五点均值不适合所有目标尺度

- 64×64 是绝对像素，不随原图分辨率或框尺寸变化；
- 小框会把同一中心 patch 重复计算 5 次；
- 如果“任意一个局部出现缺陷即为正”，对 5 个 logits 求均值会稀释局部强响应。

建议采用相对框尺寸的上下文扩张、ROIAlign 到固定网络输入；小框只取一个中心 patch；多 patch 使用 max、temperature-controlled log-sum-exp、noisy-OR 或可学习 MIL attention pooling，并通过消融确定。

#### P1-4：验证集重复与训练泄漏

数据审计结果：

- 验证集 616 张中有 25 组完全重复内容；
- 9 张验证 patch 与训练 patch 内容完全相同，且均为负样本；
- 按内容去重并移除训练泄漏后剩 582 张。

更重要的是，patch 可能来自同一原始大图、同一玻璃或同一视频相邻帧。仅按 patch 文件随机拆分仍可能泄漏。应保存 `source_image_id / object_id / video_id / capture_session`，按来源分组切分 train/val/test。

#### P1-5：模型初始化/权重加载过于宽松

`load_state_dict(..., strict=False)` 会静默忽略不匹配层。项目里有很多近似同名的 Detail 结构和权重，很容易加载错模型却继续运行。至少应打印并断言 missing/unexpected keys；checkpoint 应保存 `model_class`、结构配置、预处理版本和类别定义。

#### P1-6：验证每个 batch 都 deepcopy YOLO ModuleList

位置：`val.py:372-374`

推理状态下每次前向都深拷贝整个 `self.model.model`，没有必要，显著增加 CPU/GPU 内存和延迟，也可能破坏 hook、模块引用和序列化行为。应删除并定位当初需要拷贝的真实原因。

#### P1-7：实现硬编码为单类别

位置：`val.py:652-665`

只更新 `x[0][:, 4, anchor]`，即第一个类别。若未来扩展多类，该逻辑会悄悄出错。应记录候选 anchor 的预测 class id，只更新对应类别，或明确断言 `nc == 1`。

#### P1-8：两阶段训练脚本冻结了错误的模型变体

`train_v2.py` 解冻的关键词 `conv_for_yolo_low`、`conv_l/m/s` 属于 `Detail_Net_attn_block`，当前 `BGD_YOLO` 实例化的却是 `Detail_Net_attn`。当前结构中这些参数名不存在，Stage 1 可能没有任何可训练参数。脚本还把 `stage1=False`、`stage2=False` 写死，却无条件执行最后的验证。

#### P1-9：设备、性能和部署问题

- `ultralytics/yolo/utils/loss.py:134` 硬编码 `cuda:0`，不支持 CPU、多卡或非 0 号 GPU；
- `ultralytics/yolo/data/base.py` 对每个样本额外 `cv2.imread` 一次保存 `ori_img`，但主链路并未使用，增加 I/O；
- Detail 前向里使用 OpenCV、Python 循环、整数坐标和磁盘读取，不能自然导出 ONNX/TensorRT；
- Grad-CAM 需要反向传播，部署成本更高；
- 历史日志中的联合训练约 14 秒/训练 batch，验证约 10 秒/batch，说明当前流水线开销很大。

### 4.3 P2：可维护性和可复现性问题

1. 大量路径写死为 `/home/bme-2020/czy/...`，当前工作区无法直接运行训练/验证。
2. 当前环境缺少 `opencv-python`、`torchvision`、`pytorch-grad-cam`；`requirements.txt` 虽包含前两者，但缺少 `pytorch-grad-cam`、`scikit-learn` 和代码中直接导入的 Triton 说明。
3. `Resnet6.py` 导入了未使用的 `triton.ops.blocksparse.softmax`，给模型增加了不必要的强依赖。
4. `val.py` 中 `head_select`、`detail_position`、`detail_num`、`iou_threshold`、`detail_max_boxes`、`united_train` 等参数当前没有实际控制主算法。
5. Debug 可视化字典被创建但从未填充，相关代码实际上不会输出预期可视化。
6. `dataLoader.py` 是未完成原型，静态检查发现 `self`、`detections`、`total_loss`、`train_loader` 等未定义名称，不能作为有效训练入口。
7. `jointTraing.py` 是空文件。
8. 源码、数据、权重和约 8.1 GB 的 `yolo-runs` 混在一个仓库；没有清晰的配置、版本化实验清单和最小复现说明。
9. Git 工作区有大量改动，其中许多文件表现为等量增删，疑似换行符变化；真实算法修改很难从 diff 中识别。
10. 静态检查在排除 `ultralytics/tests/examples` 后仍报告 230 个 F/E9 类问题，虽然多数是未使用导入，但包含会真正运行失败的未定义名称。

## 5. 对方法本身的评价

### 5.1 合理之处

1. 级联设计适合细粒度、透明、反光目标：YOLO 提供目标级定位和较高召回，局部模型专门学习边缘、裂纹、纹理或光学细节。
2. Detail 模型只处理少量候选时，可以用较高局部分辨率而不必把整图检测输入无限放大。
3. 使用 hard negatives 训练 Detail 分支，有机会降低 YOLO 在背景纹理和高亮区域上的误报。
4. 多尺度 Detail 分支和膨胀卷积注意力能够同时看局部纹理与稍大范围上下文，方向上有意义。

### 5.2 方法层面的风险

1. **级联无法自动恢复第一阶段漏检。** 如果只处理 `YOLO score > 0.5` 的框，YOLO 漏掉的目标永远不会进入 Detail。该系统更适合降低 FP，而不是提高召回。若希望兼顾召回，应把第一阶段阈值降低并采用 Top-K/预算控制。
2. **Grad-CAM 不一定是最佳候选定位器。** 它解释的是 YOLO 当前决策，不保证热区就是最有判别力的真实细节；错误检测的热区尤其可能强化背景偏差。
3. **固定局部裁剪可能丢失上下文。** 玻璃/透明目标的判断常依赖边界、反射连续性或更大结构，仅看 64×64 可能不够。
4. **两个模型的分数未必可直接线性平均。** YOLO score 与 Detail score 的校准分布、训练目标不同，固定 `alpha=0.5` 没有统计依据。
5. **独立训练产生域差异。** Detail 独立训练的 patch 来源与联合推理时预测框产生的 patch 分布可能不同；应使用 out-of-fold YOLO 预测生成训练 patch，避免只用理想 GT 区域。

## 6. 推荐的改进方案

### 6.1 推荐的近期可落地版本

优先实现一个稳定、易验证的两阶段系统，先不做联合反传：

```text
原图
  → YOLO 低阈值推理（例如 0.05～0.15）
  → NMS + 每图 Top-K
  → 对每个框按相对尺度扩张并 ROIAlign
  → Detail_Net_attn 输出 detail_logit
  → 校准后的融合器输出 fused_logit
  → 最终阈值 / 可选第二次 NMS
```

具体建议：

1. 第一阶段先冻结，不改写 YOLO raw feature maps，避免破坏 DFL。
2. 从 NMS 后候选框裁剪，显著减少重复计算。
3. 使用 ROIAlign 或 `grid_sample` 从当前模型输入 tensor 裁剪，训练/验证/推理保持一致。
4. 每个框使用“紧框 patch + 扩张上下文 patch”两尺度，比固定五点更容易解释和部署。
5. Detail 使用 `BCEWithLogitsLoss`；若类别不平衡，再正确加入 focal/pos_weight。
6. 融合优先使用 `fused_logit = yolo_logit + β × detail_logit + b`，在独立校准集学习 `β,b`；也可训练一个两层小 MLP。
7. 根据业务目标选择阈值：若漏检代价高，优化 Recall/F2；若人工复核成本高，优化 Precision/F0.5。

### 6.2 如果必须保留 Grad-CAM

1. 只对 NMS 后且处于不确定区间的候选（例如 0.1～0.6）计算 CAM。
2. 明确 CAM target：对应候选框的目标类别 logit，而不是所有框/所有类别的总和。
3. 选择与小目标分辨率匹配的 FPN 层，可做 LayerCAM/多层 CAM 消融。
4. 在框内找最大响应窗口时加入最小覆盖、边界约束和多峰抑制；同一框可取 1～3 个非重叠峰。
5. 把 CAM 作为可解释的“选点策略”，不要声称其天然提高准确率；必须与中心裁剪、五点裁剪、随机/网格裁剪作公平消融。

### 6.3 更进一步的替代方案

如果目标是更快、更容易端到端训练，优先考虑：

- 从 YOLO P2/P3 特征对候选框做 ROIAlign，再接一个轻量二分类 head；
- 增加 P2 小目标检测头或提高输入分辨率；
- 使用可学习 deformable attention/ROI attention 代替 Grad-CAM；
- 把 Detail 判别作为检测 head 的辅助损失，多任务联合训练；
- 若缺陷只占目标局部，采用 MIL：多个 patch → attention pooling → object-level label。

这些方案不需要每个候选做反向传播生成 CAM，更适合部署。

## 7. 必须补做的实验与消融

为了证明方法有效，至少需要同一数据划分、同一 YOLO 权重下比较：

| 实验 | 目的 |
|---|---|
| YOLO baseline | 基线 |
| YOLO + 中心单 patch + Detail | 最小两阶段收益 |
| YOLO + 五点 patch + mean | 对应当前新版 |
| YOLO + 五点 patch + max/LSE/attention pooling | 验证聚合方式 |
| YOLO + Grad-CAM 单峰 + Detail | 验证 Grad-CAM 是否真的优于几何裁剪 |
| Detail_Net 无 attention / 有 attention | 验证注意力贡献 |
| 固定 64 / ROIAlign 相对尺度 / 双尺度上下文 | 验证裁剪策略 |
| 固定 0.5 融合 / 几何均值 / 学习融合 / 校准融合 | 验证分数融合 |

检测系统应报告：Precision、Recall、AP50、AP75、mAP50-95、FP/image、FN/image、候选数/image、端到端 latency、GPU 显存；Detail 模型应报告 ROC-AUC、PR-AUC、F1、混淆矩阵和校准误差。需要给出至少 3 个随机种子的均值和标准差。

数据划分必须按原始图/目标/视频/采集批次分组，最后测试集只使用一次。阈值、alpha、CAM 层和 pooling 都只能在验证集选择。

## 8. 推荐修复顺序

1. 修正 Detail logits 损失和验证阈值；清洗并按来源重划细节数据集。
2. 为当前已经修正的单次 `/255` 预处理增加输入范围单元测试。
3. 暂停当前 raw-logit 联合训练；先实现 NMS 后 ROIAlign + Detail 的可靠推理链路。
4. 统一训练/验证/推理的裁剪与坐标系统，不再在 forward 中读磁盘原图。
5. 完成 YOLO baseline 与中心/五点/Grad-CAM 三组消融。
6. 依据消融结果决定是否保留 Grad-CAM。若增益不明显，使用几何/ROI 特征方案。
7. 最后再实现可微融合或联合微调，并修复 64 通道 DFL 切分。
8. 拆分代码为 `configs/`、`models/`、`datasets/`、`pipelines/`、`tools/`、`tests/`；将权重和 runs 移出 Git。

## 9. 本次实际检查与验证

- 对主要 Python 文件执行了语法编译检查：未发现语法错误。
- `Detail_Net_attn` CPU 前向通过：输入 `(2,3,64,64)`，输出 `(2,1)`。
- `exp3_4_1.pt` 与当前 `Detail_Net_attn` 严格匹配：0 missing keys、0 unexpected keys。
- 对细节训练/验证 CSV、图像存在性、尺寸、类别分布、内容哈希重复进行了审计。
- 在不依赖 torchvision/OpenCV 的读取路径下，直接加载现有权重重新计算了清洗前后指标。
- 运行了 Ruff F/E9 静态检查；发现 230 项，其中大部分是未使用代码，也包括 `dataLoader.py` 的未定义名称。
- 未完成端到端 YOLO 运行：当前环境缺少 OpenCV、torchvision、pytorch-grad-cam，而且主训练脚本依赖工作区外的 `/home/bme-2020/czy/yolov10/data.yaml` 和数据集路径。

## 10. 最终判断

**方法方向可以继续，但当前代码还不能作为严谨、可复现的最终实现。** 最值得保留的是“YOLO 候选生成 + 局部细节复核”的核心思想；最需要重新决定的是“Grad-CAM 是否真的必要”。

建议先把方法收敛为一个稳定的 NMS 后 ROI/局部二次分类系统，修复数据、损失、坐标和融合问题，再通过消融决定采用 Grad-CAM、五点裁剪还是可学习 ROI 特征。只有在同一干净测试集上证明 Grad-CAM 相对简单裁剪有稳定增益，论文中才应把它写成核心贡献。
