# Direct input64 修正版 P3 / P4 / P5：完整 Test、时间、参数量与 GFLOPs

更新：2026-09-16。三组均完成 50 epoch 和完整 Test，状态 complete，无训练或测试退出错误。

本页只记录本轮 `geometry_v5_input64_sum_e50_lr1e4_r1` 的三个**独立单头实验**，不是一套 P3+P4+P5 多头模型，也不是此前仅修复 xy / add 的旧结果。没有 alpha；重叠 Detail 回填为 sum。

P3+P4+P5 三头 r2 也已完成并与本页一起更新，统一比较、可下载四个 best.pt 见[当前单头与三头总表](DIRECT_INPUT64_CURRENT_SINGLE_MULTI_RESULTS_20260916.md)；三头真实改善图片见[00601 / 00890 可视化](DIRECT_INPUT64_MULTI_IMPROVEMENT_VISUALS_20260916.md)。

## 1. 最终 Test 指标

使用各自 **Val mAP50 选择的 best.pt**；没有使用 Test 选择 epoch。以下数值为 0–1，保留九位小数；精确浮点数见 [结果 JSON](results/direct_input64_heads_test_20260916.json)。

| 独立实验 | Precision | Recall | mAP50 | mAP75 | mAP50–95 | Test inference（ms/张） |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 0.957983331 | 0.893162393 | 0.941816639 | 0.910698774 | 0.884631232 | 8.759546 |
| P4 | 0.950450450 | 0.901709402 | 0.935256143 | 0.914600685 | 0.881924249 | 8.575551 |
| P5 | 0.951301142 | 0.897435897 | 0.943038918 | 0.919773816 | 0.890624975 | 8.753099 |

本轮三组中 P5 的 mAP50 / mAP75 / mAP50–95 最高；这是一种子实验，不代表统计显著性或所有场景都最好。

## 2. Val 选权重记录

来自各组训练 CSV 中选出 best 的行，保留 CSV 的精度。epoch 以下从 1 开始；原 CSV 从 0 开始。这里不是第 50 轮指标，也不是 standalone 恢复后再次验证的数值。

| 实验 | Best epoch | Val P | Val R | Val mAP50 | Val mAP75 | Val mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3 | 45 | 0.95709 | 0.94076 | 0.97293 | 0.93775 | 0.91610 |
| P4 | 9 | 0.97067 | 0.94119 | 0.97379 | 0.94204 | 0.91472 |
| P5 | 38 | 0.98232 | 0.92163 | 0.97446 | 0.93407 | 0.91530 |

完整 50 epoch CSV：[P3](results/direct_input64_p3_training_20260916.csv)、[P4](results/direct_input64_p4_training_20260916.csv)、[P5](results/direct_input64_p5_training_20260916.csv)。

## 3. 时间统计与测试环境

| 实验 | 预处理 ms/张 | inference ms/张 | loss 计时段 ms/张 | 后处理 ms/张 |
|---|---:|---:|---:|---:|
| P3 | 0.105217 | 8.759546 | 0.003458 | 0.152848 |
| P4 | 0.103827 | 8.575551 | 0.003451 | 0.150166 |
| P5 | 0.108616 | 8.753099 | 0.003523 | 0.155220 |

loss 计时段是 standalone 验证器空分支 / 计时开销，不代表 Test 计算训练 loss。

| 项目 | 设置 |
|---|---|
| GPU | 同一块 NVIDIA RTX 6000 Ada Generation；Test 在所有训练完成后顺序执行 |
| Python / Torch | bgd 环境，Python 3.11，Torch 2.7.0+cu128 |
| 数据 | `experiments/bg_local.yaml`，split=test；1598 images / 234 instances，1392 background images |
| 输入 | imgsz=640，rectangular letterbox；并非每张都严格 640×640 |
| batch / precision | 1 / FP32，half=False |
| workers / OMP、MKL threads | 4 / 4；OpenBLAS=1 |
| candidate confidence | 原 YOLO pre-NMS score >0.2，train / val / test 一致 |
| 最终 confidence / NMS IoU | 0.5 / 0.5 |
| Test 次数 | 表 1 为训练结束后的每模型一次完整 Test；之后另做每模型三次统一预热复测，见下文 |
| 时间边界 | 验证器 inference，含模型内部 YOLO、CAM、原图 crop、Detail、scatter、融合及再次解码；不含 loader、外部预处理与最终 NMS |

表中时间是**整个 Test 集的单次平均 inference**，不是只统计含有效 Detail crop 的图片，也不是完整请求端到端耗时。无候选样本跳过 CAM / Detail 的既有分支仍保留。不能将本页约 8.6–8.8 ms 与 [旧版三轮约 54 ms](CORRECTED_DIRECT_HEADS_TEST_TIMING_20260915.md) 直接混用或据此声称固定加速倍数。

差异已经定位到实现，而不只是计时名称：旧 ZIP `_predict_once` 在每张验证图片上执行 `copy.deepcopy(self.model.model)`，末尾执行 `torch.cuda.empty_cache()`，还会全模型清理 / 重挂 hook；当前 geometry-v5 复用同一模型，只对目标层注册并在 `finally` 移除临时 hook，也不逐图清空 CUDA allocator。CAM backward、原图 crop、Detail 与融合仍在当前 inference 计时范围内。

当前版本每模型三轮完整 Test 受控复测已完成：P3 **8.098744±0.227742**、P4 **7.885968±0.005193**、P5 **7.942459±0.027016** ms/张；三头 r2 为 **7.972821±0.011996** ms/张。三轮均做 20 张真实图预热、CUDA 同步，且无其他 GPU 计算进程，证明约 8 ms 不是单次日志偶然值。完整数据见[统一结果页](DIRECT_INPUT64_CURRENT_SINGLE_MULTI_RESULTS_20260916.md)和[计时 JSON](results/direct_input64_v5_controlled_timing_20260916.json)。论文可称为实现优化后的实测延迟，但不能把旧 / 新数值解释为网络理论 FLOPs 固定加速。

正式启动 2026-09-15 20:27:53 JST；所有训练及最终 best Val 恢复完成于 2026-09-16 00:33:16 JST；全部 Test 完成于 00:35:09 JST（北京时间 9 月 15 日 23:35:09）。

## 4. 参数量

| 实验 | 训练 / 未 fuse 总参数 | 参数量 M | YOLO Conv–BN fuse 后部署总参数 |
|---|---:|---:|---:|
| P3 | 4,209,276 | 4.209276 | 4,204,076 |
| P4 | 4,209,276 | 4.209276 | 4,204,076 |
| P5 | 4,209,276 | 4.209276 | 4,204,076 |

各组包含一套 YOLO、historical wide Detail、67→64 融合投影。三个 Detect 分类分支 hook 的通道均为 64，因此单头参数量相同；分辨率不同导致计算量不同。部署 fold YOLO Conv–BN 后参数差异不是另一次训练或权重丢失。

## 5. GFLOPs：实际权重的模块 forward 核验

测量固定 `[1,3,640,640]` 输入、FP32；THOP，1 MAC = 2 FLOPs。原始 Detail 输入 `[1,3,64,64]`，输出始终 `[1,3,4,4]`。本次直接加载实际 best EMA，分别测 YOLO、一个 Detail crop、额外融合投影及再次调用分类卷积。

主表采用与实际 standalone Test 一致的 **YOLO Conv–BN fused 部署**口径。N 为实际进入 Detail 的 crop 数。

| 实验 | YOLO fused forward | 额外融合与分类器 forward | 固定部分 A | N 个 crop 后的模块 forward GFLOPs | 示例 N=1 |
|---|---:|---:|---:|---|---:|
| P3 | 8.0851968 | 0.4980736 | 8.5832704 | 8.5832704 + 0.043610496 × N | 8.626880896 |
| P4 | 8.0851968 | 0.1245184 | 8.2097152 | 8.2097152 + 0.043610496 × N | 8.253325696 |
| P5 | 8.0851968 | 0.0311296 | 8.1163264 | 8.1163264 + 0.043610496 × N | 8.159936896 |

单个 Detail crop 为 **0.043610496 GFLOPs**；不是 0，也不是每张都固定只有一个 crop。若一个样本没有有效融合，额外融合层跳过，此时执行路径不能机械按 A+0 统计，native YOLO 模块部分为 8.0851968。

补充未 fuse 架构口径：YOLO forward 8.1941504；P3/P4/P5 固定部分分别为 **8.6922240 / 8.3186688 / 8.2252800 GFLOPs**，均另加 `0.043610496 × N`。不能把未 fuse 参数 / FLOPs 与 fused 部署数值混成一个表而不说明。

**上述不是完整方法总 GFLOPs**：不包括 Grad-CAM backward、CAM 归一化 / 放大、全 1 卷积窗口搜索、Detail 输出 resize GEMM、scatter sum、第二次 decode、NMS，以及 THOP 未注册的 functional 操作（例如 attention elementwise operations）。完整方法 total GFLOPs 在结果 JSON 中明确为 null，不伪造数值。启动日志 8.1 GFLOPs 只是 YOLO 主干 forward，不是含 Detail 的完整模型。

固定 640×640 FLOPs 也不是实际 rectangular Test 的逐图平均计算量。N=1 只是示例，不是实测 Test 平均 crop 数。

## 6. 本轮方法与训练设置

| 项目 | 设置 |
|---|---|
| 正式训练 | 各 50 epochs，三个独立单头任务，seed=0，deterministic=True |
| optimizer / lr0 / lrf | AdamW / 0.0001 / 0.01 |
| momentum / weight decay | 0.937 / 0.0005 |
| warmup | 3 epochs，momentum=.8，bias lr=0 |
| train imgsz / batch / AMP | 640 / 32 / True |
| 初始化 | YOLO `yolo-runs/train/train10/weights/best.pt` + Detail `run/detail_net_attn.pt` 独立预训练；不从旧 joint / 318.pt 或中断实验续训 |
| 网络 | historical wide Detail，相同原图原生 64×64 RGB，ImageNet normalize；原始 Detail 输出 3×4×4 |
| hook | Detect 内 `cv3[0][1]` / `cv3[1][1]` / `cv3[2][1]`，全局 64ch |
| 回填支持 | 固定输入 64×64 context：P3=8×8、P4=4×4、P5=2×2 |
| 融合 | 3ch sparse feature + 64ch global，Conv3×3 67→64 + BN + ReLU，再用原 Detect classifier；仅替换所选层 cls logits |
| 重叠 | 原位 `.add_()` sum；无 alpha、无 confidence weighting |
| 坐标 | 实际原图来源、整数 padding、Mosaic、平移 / 缩放 / flip、实际 clamp 后 crop 中心；画布边缘只 clip、不移动 / 拉伸 |
| CAM | 保留原 all-sigmoid-score sum target、mean-gradient weighting、ReLU、全零时有效框中心回退；没有修改方法规避零值 |
| 增强 | 默认 Mosaic、HSV、scale、translate、fliplr 保留；close_mosaic=0 |

固定输入支持是上下文注入，不是原图 crop 每个像素与全局特征逐像素对齐；中心有最多半个各层网格的量化误差。CAM 全零样本确实存在并逐 epoch 记录，不能把“坐标检查通过”解释为“热力图总能定位破损区域”。

## 7. 权重具体位置与核验

以下为训练机本地路径，相对项目根目录 `/home/user/projects/czy/BGD-Yolo_v2_copy`。本次更新已上传三个单头 best.pt；不上传重复的 last.pt。

| 实验 | GitHub best.pt | 训练机原路径 | Best epoch | 文件大小 bytes |
|---|---|---|---:|---:|
| P3 | [p3_best.pt](weights/direct_input64_20260916/p3_best.pt) | `experiments/runs/best318_high_p3_detecthead_geometry_v5_input64_sum_e50_lr1e4_r1/weights/best.pt` | 45 | 10,303,029 |
| P4 | [p4_best.pt](weights/direct_input64_20260916/p4_best.pt) | `experiments/runs/best318_mid_p4_detecthead_geometry_v5_input64_sum_e50_lr1e4_r1/weights/best.pt` | 9 | 10,303,029 |
| P5 | [p5_best.pt](weights/direct_input64_20260916/p5_best.pt) | `experiments/runs/best318_low_p5_detecthead_geometry_v5_input64_sum_e50_lr1e4_r1/weights/best.pt` | 38 | 10,303,029 |

每个同目录还有 `last.pt`。完成训练后 optimizer 已 strip；文件大小约 9.83 MiB，不能沿用训练中含 raw / optimizer 的约 61 MB，也不能用历史旧版约 685 MB 的文件大小解释本轮参数量。best 与 last 的 SHA256 及字节数均在结果 JSON。

best SHA256：

- P3：`03bbecbb8257ad933c8e381c8941e2b1f23dadcbad8f15ea5dcf172789357f7d`。
- P4：`ca1379869b5d1d5f99ab1d86bc895207a5227858d6e0be888e03e7ad0b412cff`。
- P5：`74af0b70567be09acd2473659427d843c1d7b063be39d5825efa94563d5fbaf0`。

以 JSON 的完整 SHA256 为核验依据。

## 8. 数据与版本核验

- [精确结果 JSON](results/direct_input64_heads_test_20260916.json)：Test、Val、每阶段时间、模型参数、fused / unfused GFLOPs 分解、配置、权重 hash 与路径。
- JSON 的 `training_code_sha256` 记录本轮实际运行代码版本；扩展多头前已核对全部匹配。
- 完整运行源码快照与只读 GFLOPs 核验脚本保留在训练机本地，分别为 `docs/results/direct_input64_v5_runtime_20260916/` 与 `docs/results/profile_input64_heads_20260916.py`；本次上传的是结果、图像与关键 best.pt，不发布完整训练运行归档。
- 机器本地完整状态：`experiments/analysis/geometry_v5_input64_e50_r1/status.json`；完整训练和 Test 日志位于同目录。

main 本次更新包含 docs 下的结果 MD、JSON、训练 CSV、可视化和四个关键 best.pt，不覆盖 main 的训练代码。复现仍需要相同运行版本、本地环境、历史代码 ZIP 与数据。P3+P4+P5 多头 r2 已完成，见[统一结果页](DIRECT_INPUT64_CURRENT_SINGLE_MULTI_RESULTS_20260916.md)。
