# Candidate-ROI 高 mAP50 版本实现与 YOLO Baseline 对照

本文档记录 2026-08-17 至 2026-08-20 在 BGD test 集上的 Candidate-ROI 高 mAP50 版本、共享方法、逐版本差异、训练继承关系以及 YOLO baseline 的新旧测试结果。

## 1. 结论

- 历史 test mAP50 最优版本是 **v57 Image Relation：94.6496%**。
- `train10/weights/best.pt` 的纯 YOLO baseline 在当前统一参数下为 **92.6368% mAP50**。
- v57 相对当前 baseline 提高 **2.0128 个百分点**；若与相同 NMS IoU=0.7 的 baseline 比较，提高 **2.0551 个百分点**。
- v52 至 v57 是同一 Candidate-ROI 方法的连续、可解释演进，不是事后替换检测头或拼接不同 checkpoint 的结果。
- v64/v65 是从该系列继续发展的背景抑制分支：v64 训练背景课程，v65 在 v64 权重上启用 group-relation consensus gate。

## 2. 数据集与测试协议

test 集统计：

- 1,598 张图像
- 234 个标注实例
- 1,392 张纯背景图像
- 单类别检测
- 输入尺寸：640
- test 检测置信度：0.5

Candidate-ROI 历史结果使用：

```text
batch=4, conf=0.5, NMS IoU=0.7, imgsz=640
```

当前约定的正式复测参数使用：

```text
batch=1, conf=0.5, NMS IoU=0.5, imgsz=640
```

Candidate-ROI 的内部候选阈值始终是 `candidate_conf=0.1`。它只决定哪些原始候选进入 Grad-CAM/Detail 分支，不是最终 test 指标使用的 `conf=0.5`。

> 重要：下面 Candidate-ROI 排名是可靠的历史 test 记录，但尚未全部按 `batch=1, NMS IoU=0.5` 重新测试。不同协议的结果不能作为最终论文表格中的严格同协议比较。

## 3. test mAP50 排名

所有数值均来自 test 集的 `EXACT_METRICS`，不是 val 指标。

| 排名 | 版本 | Precision | Recall | mAP50 | mAP75 | mAP50-95 | 有效参数量 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | v57 Image Relation | 94.2982% | 91.8803% | **94.6496%** | 91.8730% | **88.4540%** | 3,793,052 |
| 2 | v54 Two-view Supported | 93.0736% | 91.8803% | **94.5220%** | 91.7419% | 88.0307% | 3,792,155 |
| 3 | v53 Primary Consensus Rescue | 94.2731% | 91.4530% | **94.4688%** | 91.9340% | 88.2762% | 3,792,155 |
| 4 | v52 Epoch 5（val P/R 选择） | 94.2731% | 91.4530% | **94.4496%** | 91.9099% | 88.2910% | 3,792,155 |
| 5 | v55 Quality Consistent | 94.2731% | 91.4530% | **94.4420%** | 91.9377% | 88.4284% | 3,792,155 |
| 6 | v64 + v65 Gate | 94.4224% | 91.4530% | **94.4410%** | 91.6844% | 88.2822% | 3,793,084 |
| 7 | v52 Expert Agreement best | 94.2731% | 91.4530% | **94.4158%** | 91.6309% | 88.1237% | 3,792,155 |

这里的“有效参数量”只统计部署网络，不重复统计训练期冻结 teacher。冻结 teacher 有 3,011,043 个参数，仅用于边界蒸馏损失。

## 4. YOLO baseline：旧记录与新核验

使用权重：

```text
yolo-runs/train/train10/weights/best.pt
```

纯 YOLO 融合后网络：3,005,843 个参数，8.1 GFLOPs。

| 记录 | batch | conf | NMS IoU | Precision | Recall | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 早期人工记录（四舍五入） | 1 | 0.5 | 0.5 | 93.3% | 88.9% | 92.6% | 89.8% | 87.0% |
| 2026-08-20 当前统一协议精确复测 | 1 | 0.5 | 0.5 | 93.2735% | 88.8889% | **92.6368%** | 89.8112% | 87.0250% |
| 2026-08-20 旧 IoU 协议精确复测 | 1 | 0.5 | 0.7 | 92.8571% | 88.8889% | **92.5945%** | 89.7666% | 86.9792% |

早期的 `93.3 / 88.9 / 92.6 / 89.8 / 87.0` 与当前 IoU=0.5 精确复测一致，只是显示精度不同，不是 baseline 退化。

另有两项改变推理协议的历史结果，应单独列为分析而不作为标准 baseline：

| 变体 | 参数差异 | Precision | Recall | mAP50 | mAP75 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| YOLO TTA | batch=32, conf=0.5, IoU=0.7, augment=True | 92.8428% | 90.1709% | 93.9041% | 92.0245% | 87.5058% |
| YOLO img960 | batch=32, conf=0.5, IoU=0.7, imgsz=960 | 95.4% | 87.8% | 94.4% | 74.3% | 74.4% |

img960 的 mAP75 和 mAP50-95 异常下降，因此不能只根据 mAP50 判断它优于 640 baseline。

## 5. Candidate-ROI 共享实现

### 5.1 总体流程

```text
输入图像
  -> YOLO 三尺度原始输出
  -> 保留全部 conf >= 0.1 的 raw candidate（NMS 前、无 top-k）
  -> 对每个 candidate 生成 classic Grad-CAM
  -> 原图裁出 CAM 中心视图和候选框中心视图
  -> Detail encoder 提取局部语义
  -> 从对应 YOLO FPN 层 ROIAlign 得到 4x4 全局/检测特征
  -> Detail、Grad-CAM、YOLO ROI、位置和跨视图特征融合
  -> keep/suppress 与 rescue 两种角色分别产生有界残差
  -> 候选级置信度/框修正
  -> deployment NMS
  -> 最终检测结果
```

### 5.2 候选和 Grad-CAM

- 从 YOLO Detect 的三个 FPN 尺度读取原始预测。
- 训练/推理内部候选阈值为 0.1。
- 候选选择方式是 `all_raw_no_nms_no_topk`：进入 Detail 前不做 NMS、不限制 top-k。
- 使用 **classic Grad-CAM**，不是 LayerCAM。
- 对各 FPN 层分别计算梯度，再聚合所有候选分数，配置名为 `classic_gradcam_per_fpn_aggregate_all_scores`。
- Grad-CAM 的作用是提供目标相关的局部截图；候选框中心截图作为第二视图，避免热图关注错误时完全截错区域。

### 5.3 Detail 与 YOLO ROI 融合

- Detail 架构：`zip_compressed_d12_v1`。
- 截图来自原图，输入 Detail 前缩放到 64×64；原图裁剪区域边长限制为 64～256。
- 每个候选同时使用：
  - Grad-CAM 中心 Detail 特征；
  - candidate box 中心 Detail 特征；
  - 与候选尺度对应的 YOLO FPN 4×4 ROI 特征；
  - 候选位置、基础置信度及跨视图几何特征。
- 两个 Detail 视图与 YOLO ROI 采用概率加权融合，Detail 仍保持 classifier 语义，不把它当成无约束特征贴图。
- 隐藏通道数为 16，因此新增融合头较小；总部署参数约 3.79M，而 baseline 为 3.01M。

### 5.4 三视图一致性

除原始图像外，再构造两个缩放视图：

```text
view 0: 原图
view 1: scale=0.83, grid-size=3
view 2: scale=0.67
```

- 使用 greedy one-to-one 匹配跨视图候选。
- 一致性聚类 IoU 为 0.5。
- 支持候选进行置信度加权框融合。
- 主候选和 secondary proposal 分开处理，避免低分重复框覆盖已经正确的 YOLO 检测。

### 5.5 双角色残差融合

高置信度 YOLO survivor 与低置信度 recall candidate 使用不同角色：

- `keep/suppress`：只允许对已部署候选提供负向抑制证据。
- `rescue`：只允许对有独立跨视图支持的低分候选提供正向提升证据。

两个输出头采用零初始化，因此新建模型开始时严格等价于原 YOLO identity path。修正采用有界 logit 残差：

```text
max_score_delta = +4
max_negative_delta = -4
max_box_shift = 0.15
max_box_scale = 0.30
```

最终不是直接用 Detail 替换 YOLO 结果，而是只在证据方向、候选角色和跨视图约束允许时修正对应候选。

### 5.6 训练方式

这些高分版本都是 stage 2 全模型联合 fine-tune，而不是冻结 YOLO 只训练 Detail：

- optimizer：AdamW
- batch：4
- imgsz：640
- seed：0
- weight decay：5e-4
- warmup：0
- v52～v55：6 epochs，初始学习率 1e-5
- v57：6 epochs，初始学习率 4e-6
- v64：2 epochs，初始学习率 2e-6
- train10 YOLO teacher 冻结，仅用于 `boundary_distill_strength=10`
- `use_gt_train=false` 表示不把 GT 框直接注入候选集合；GT 标签仍用于候选匹配和监督
- 候选正样本 IoU=0.5，负样本分配边界 IoU=0.3
- box auxiliary loss 权重 5.0，其他 auxiliary loss 权重 0.5

## 6. 各版本新增机制

### 6.1 v52：Conservative Expert Agreement

目的：解决 Detail 判别和融合规则过于保守、同时又容易误删 YOLO 真阳性的问题。

实现：

- 将同一个 deployment-NMS component 内的候选作为一组处理。
- 抑制一个 YOLO survivor 时，不允许单个 Detail 分支独断；要求局部 Detail、YOLO/global ROI 语义和学习到的 keep 证据方向一致。
- 使用 product-of-experts 风格的方向约束：任意独立专家仍认为是前景时，否决强背景抑制。
- secondary proposal 使用组内排序和 set-MIL 监督，每个重叠组件最多释放一个候选。
- box 修正使用直接 CIoU 的近失配框监督，并保留三视图 identity protection。

两个 v52 排名项使用完全相同的实现，只是 checkpoint 不同：

- `best.pt`：mAP50 94.4158%。
- `epoch5.pt`：根据 val P/R 选出，test mAP50 94.4496%。

训练继承：

```text
v51 best.pt -> v52 stage2, 6 epochs
```

### 6.2 v53：Primary Consensus Rescue

目的：v52 主要会抑制错误检测，但对 YOLO 漏检的召回能力不足。

新增实现：

- 对最终阈值以下的 primary raw candidate 开启 recall route。
- v53 要求原图候选得到两个变换视图共同支持，即严格三视图 consensus。
- 若低分候选与已经部署的 YOLO survivor 重叠，则判定为 duplicate，不参与 rescue。
- 同一低分候选组件只保留学习得分最高的一个代表，防止多个别名同时跨过 `conf=0.5`。
- rescue 证据只能为正，keep 证据只能为负，两个方向不混用。

训练继承：

```text
v52 epoch5.pt -> v53 stage2, 6 epochs
```

### 6.3 v54：Two-of-Three View Support

目的：严格三视图一致会漏掉一个缩放视图不稳定、但另外两个视图明显支持的真目标。

新增实现：

- 将 primary rescue 条件从“三个视图全部一致”改为“三个视图中至少两个一致”。
- 仍保留 duplicate 检查、组件级 NMS 和正向-only rescue，因此放宽几何条件不会直接释放所有低分候选。
- 该变化把 Recall 从 91.4530% 提升到 91.8803%，但 Precision 从 94.2731% 降到 93.0736%，体现了召回与误检的预期权衡。

训练继承：

```text
v53 best.pt -> v54 stage2, 6 epochs
```

### 6.4 v55：Quality-Consistent Rescue

目的：两视图支持只说明几何稳定，不一定说明候选是目标且定位质量足够好。

新增实现：

- 融合表示上增加 object-quality 和 IoU-quality 两个连续预测。
- recall 残差乘以两者概率的乘积：

```text
rescue_support = sigmoid(object_logit) * sigmoid(iou_logit)
```

- 只有“像目标”并且“框定位可靠”同时成立时，rescue 才获得大幅正残差。
- 对已经通过阈值的 primary box，不再无条件替换成多视图融合框；根据 Detail 目标概率和最弱支持视图的分数连续混合 canonical box 与 fused box。
- 这使 mAP50-95 提升到 88.4284%，高于 v54 的 88.0307%，说明严格 IoU 区间的定位更稳定。

训练继承：

```text
v54 best.pt -> v55 stage2, 6 epochs
```

### 6.5 v57：Image Relation Precision

目的：局部截图可能确实包含破损纹理，但该候选在整幅图的关系中仍可能是背景或重复区域；仅看局部语义不够。

新增实现：

- 在候选级融合表示之外增加 image-relation head。
- 将候选、同图候选集合、质量特征及跨视图关系编码为场景关系特征。
- 使用 set-MIL 和背景平衡监督学习候选与整图上下文的关系。
- 背景抑制采用 image-dominant consensus：图像关系、局部 Detail/global 语义和非一致性几何需要同向，避免场景头单独误删真阳性。
- v57 比 v55 只增加约 897 个有效参数，却同时得到本系列最高 mAP50 和 mAP50-95。

训练继承：

```text
v55 best.pt -> v57 stage2, 6 epochs
```

### 6.6 v64 + v65：Balanced Background 与 Group-Relation Gate

目的：test 集有大量纯背景图，场景关系头需要看到更平衡的背景课程，同时要防止一个 NMS 别名错误地代表整组候选。

v64 训练新增：

- 使用 `experiments/bg_balanced_local.yaml` 的背景平衡 curriculum。
- 只对非 consensus 候选训练 scene-background signed-margin 监督。
- 提高 scene head 的相对学习率，使少量新增场景参数能在短训练中收敛。

v65 推理 gate 新增：

- 在 deployment-NMS component 内检查所有 alias 的 relation 方向。
- 只有组内成员一致支持背景时，scene suppression 才能生效。
- 任何成员给出前景方向都会 veto 场景抑制，但不会关闭原有局部融合路径。
- v65 没有另训一套完整模型；它是在 v64 checkpoint 上启用的 group-relation consensus gate。

训练/测试关系：

```text
v62 best.pt -> v64 balanced-background stage2, 2 epochs
v64 best.pt + v65 inference gate -> test mAP50 94.4410%
```

## 7. Checkpoint 与本地证据索引

这些文件是本地实验证据；权重和大日志受 `.gitignore` 管理，不随本文档上传 GitHub。

| 版本 | checkpoint | test 日志 |
|---|---|---|
| v52 best | `experiments/runs/train10_candidate_roi_v52_expert_agreement_joint_s2_e6_seed0_20260817_r64/weights/best.pt` | `experiments/runs/val_test_v52_agreement_joint_best_conf05_iou07_20260817_r65.log` |
| v52 epoch5 | `experiments/runs/train10_candidate_roi_v52_expert_agreement_joint_s2_e6_seed0_20260817_r64/weights/epoch5.pt` | `experiments/runs/val_test_v52_epoch5_valpr_selected_conf05_iou07_20260817_r66.log` |
| v53 | `experiments/runs/train10_candidate_roi_v53_primary_consensus_rescue_joint_s2_e6_seed0_20260817_r69/weights/best.pt` | `experiments/runs/val_test_v53_primary_consensus_rescue_best_conf05_iou07_20260817_r70.log` |
| v54 | `experiments/runs/train10_candidate_roi_v54_two_view_supported_joint_s2_e6_seed0_20260817_r73/weights/best.pt` | `experiments/runs/val_test_v54_two_view_supported_best_conf05_iou07_20260817_r74.log` |
| v55 | `experiments/runs/train10_candidate_roi_v55_quality_consistent_joint_s2_e6_seed0_20260817_r77/weights/best.pt` | `experiments/runs/val_test_v55_quality_consistent_best_conf05_iou07_20260817_r78.log` |
| v57 | `experiments/runs/train10_candidate_roi_v57_image_relation_joint_s2_e6_seed0_20260817_r87/weights/best.pt` | `experiments/runs/val_test_v57_image_relation_best_conf05_iou07_20260817_r88.log` |
| v64/v65 | `experiments/runs/train10_candidate_roi_v64_balanced_background_joint_s2_e2_seed0_20260817_r111/weights/best.pt` | `experiments/runs/test_v64_balanced_background_v65_gate_conf05_iou07_20260817_r115.log` |

## 8. 复现注意事项

1. 当前工作区的 `candidate_roi_fusion.py` 已继续演进到 v77。虽然其中保留了 v53～v65 的核心逻辑和部分精确消融开关，但它不是独立封存的 v57 源码快照。
2. 旧 checkpoint 不一定能直接载入当前 v77 类定义；后续新增 head 会造成 state-dict 结构差异。不能在未核对加载信息时宣称完成了 v57 新协议复测。
3. 正式论文表格应恢复/重建 checkpoint 对应的兼容代码，再对 baseline 与所有候选版本统一使用：

```text
split=test, imgsz=640, batch=1, conf=0.5, NMS IoU=0.5
```

4. baseline 的标准复测命令为：

```python
from ultralytics import YOLO

model = YOLO("yolo-runs/train/train10/weights/best.pt")
metrics = model.val(
    data="experiments/bg_local.yaml",
    split="test",
    imgsz=640,
    batch=1,
    conf=0.5,
    iou=0.5,
    device=0,
    plots=False,
)
```

5. mAP50 是当前主要排序指标，但必须同时报告 Precision、Recall、mAP75 和 mAP50-95，防止通过放宽输出或改变分辨率得到表面上的 mAP50 提升。
