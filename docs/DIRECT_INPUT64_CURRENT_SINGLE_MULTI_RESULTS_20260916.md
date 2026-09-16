# Direct input64 当前正确版本：单头与三头完整结果

更新：2026-09-16。这里把当前 geometry-v5 / input64 / sum / 无 alpha 的三个独立单头和 P3+P4+P5 联合三头 r2 放在同一张表中。四组均训练 50 epoch，`best.pt` **只按 Val mAP50 选择**，然后在固定 Test 上测试；没有用 Test 选择 epoch 或超参数。

## Test 主结果

数据为完整 Test：1,598 张图片、234 个标注实例，其中 1,392 张背景图。评估均为 imgsz=640、FP32、batch=1、candidate conf>0.2、最终 conf=0.5、NMS IoU=0.5。

| 模型 | P | R | mAP50 | mAP75 | mAP50–95 | 参数量（训练 / fused） |
|---|---:|---:|---:|---:|---:|---:|
| P3 单头 | 0.957983331 | 0.893162393 | 0.941816639 | 0.910698774 | 0.884631232 | 4,209,276 / 4,204,076 |
| P4 单头 | 0.950450450 | 0.901709402 | 0.935256143 | 0.914600685 | 0.881924249 | 4,209,276 / 4,204,076 |
| P5 单头 | 0.951301142 | 0.897435897 | **0.943038918** | **0.919773816** | **0.890624975** | 4,209,276 / 4,204,076 |
| P3+P4+P5 三头 r2 | 0.954159140 | **0.901709402** | 0.937602885 | 0.915409993 | 0.888222115 | 4,286,844 / 4,281,644 |

当前这一个 seed 下，P5 单头的三个 AP 指标最高；三头不是自动优于最佳单头。三头 r2 与 r1 的 50 轮 CSV、Val 指标、Test 指标和全部 543 个 `best.pt` state tensor 均逐值相同，说明同 seed 确定性复跑成功，但不能代替不同 seed 的均值与标准差。

## Val 选权重

| 模型 | Best epoch（从 1 开始） | Val P | Val R | Val mAP50 | Val mAP75 | Val mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 45 | 0.95709 | 0.94076 | 0.97293 | 0.93775 | 0.91610 |
| P4 | 9 | 0.97067 | 0.94119 | 0.97379 | 0.94204 | 0.91472 |
| P5 | 38 | 0.98232 | 0.92163 | **0.97446** | 0.93407 | 0.91530 |
| 三头 r2 | 40 | **0.98259** | 0.93365 | 0.97409 | **0.93812** | **0.91807** |

训练曲线：[P3 CSV](results/direct_input64_p3_training_20260916.csv) · [P4 CSV](results/direct_input64_p4_training_20260916.csv) · [P5 CSV](results/direct_input64_p5_training_20260916.csv) · [三头 r2 CSV](results/direct_input64_multi_r2_training_20260916.csv)。三头完成状态与精确 Test 原始记录见 [r2 status JSON](results/direct_input64_multi_r2_status_20260916.json)。

## 延迟：为什么当前约 8–9 ms，而旧版约 54 ms

当前版本已在同一 RTX 6000 Ada 上完成每模型三轮完整 Test 受控复测：FP32、batch=1、workers=8、CPU threads=8，每轮先用 20 张真实图片预热并排除预热时间；每次 forward 前后 CUDA synchronize，期间没有其他 GPU 计算进程。

| 模型 | 三轮 inference ms/张 | 均值 ± 样本标准差 | Detail crops/张 |
|---|---|---:|---:|
| P3 | 8.355049 / 8.021552 / 7.919630 | 8.098744 ± 0.227742 | 1.431164 |
| P4 | 7.880268 / 7.890429 / 7.887206 | **7.885968 ± 0.005193** | 1.437422 |
| P5 | 7.927806 / 7.973636 / 7.925935 | 7.942459 ± 0.027016 | 1.396120 |
| 三头 r2 | 7.959147 / 7.981569 / 7.977748 | 7.972821 ± 0.011996 | 1.293492 |

这里的 inference 包含模型内部 YOLO、Grad-CAM、原图 crop、Detail、scatter、融合和再次 decode，不含 loader、验证器外部 preprocess 与最终 NMS。完整逐轮指标、同步 forward p50/p95、GPU 快照和每图 crop 统计见[受控计时 JSON](results/direct_input64_v5_controlled_timing_20260916.json)。三轮内每个模型的精度指标完全一致。

旧版约 54 ms 与当前 8–9 ms 的主要区别不是统计口径改名，而是执行实现已优化：旧 ZIP `_predict_once` 在每张验证图片上 `copy.deepcopy(self.model.model)`，末尾还执行 `torch.cuda.empty_cache()`，并全模型清理 / 重挂 hook；geometry-v5 改为复用同一模型，只注册并在 `finally` 移除目标层临时 hook，也不逐图清空 CUDA allocator。算法仍执行 CAM backward、原图 Detail crop 与融合，但删除了这些与预测结果无关的逐图管理开销。

当前 geometry-v5 在同一口径下稳定约 7.89–8.10 ms/张，证明 8 ms 不是日志漏计或偶然单轮值。旧版受控均值约 54 ms 来自保留逐图 deepcopy / empty-cache 开销的旧执行实现；可以报告为“实现优化前后实测延迟”，但必须同时说明代码路径改变，不能写成网络结构理论 FLOPs 带来的固定 6.x 倍加速。

## 模块 forward GFLOPs 口径

固定 640×640、FP32、YOLO Conv–BN fused、THOP、1 MAC=2 FLOPs。以下只统计可直接 profile 的模块 forward，`N` 为本图实际 Detail crop 数：

| 模型 | 模块 forward GFLOPs |
|---|---:|
| P3 | 8.5832704 + 0.043610496 × N |
| P4 | 8.2097152 + 0.043610496 × N |
| P5 | 8.1163264 + 0.043610496 × N |
| 三头 | 8.7389184 + 0.043610496 × N |

三头的一次 Detail 编码由三个头共享，不是三倍 `N`。上述数值不包括 Grad-CAM backward、CAM normalize / resize、全 1 卷积窗口搜索、scatter、再次 decode 和 NMS，因此不能称为动态方法的完整总 GFLOPs；完整值保持 `null`，不把 Ultralytics 自动打印的 8.1 GFLOPs 冒充全方法值。

## 可下载 best.pt

这次将四个关键 `best.pt` 一起上传；只上传 best，不上传重复的 last。GitHub 文件名、训练机原路径与 SHA256 如下：

| 模型 | GitHub 权重 | 训练机原路径 | SHA256 |
|---|---|---|---|
| P3 | [p3_best.pt](weights/direct_input64_20260916/p3_best.pt) | `experiments/runs/best318_high_p3_detecthead_geometry_v5_input64_sum_e50_lr1e4_r1/weights/best.pt` | `03bbecbb8257ad933c8e381c8941e2b1f23dadcbad8f15ea5dcf172789357f7d` |
| P4 | [p4_best.pt](weights/direct_input64_20260916/p4_best.pt) | `experiments/runs/best318_mid_p4_detecthead_geometry_v5_input64_sum_e50_lr1e4_r1/weights/best.pt` | `ca1379869b5d1d5f99ab1d86bc895207a5227858d6e0be888e03e7ad0b412cff` |
| P5 | [p5_best.pt](weights/direct_input64_20260916/p5_best.pt) | `experiments/runs/best318_low_p5_detecthead_geometry_v5_input64_sum_e50_lr1e4_r1/weights/best.pt` | `74af0b70567be09acd2473659427d843c1d7b063be39d5825efa94563d5fbaf0` |
| 三头 r2 | [multi_r2_best.pt](weights/direct_input64_20260916/multi_r2_best.pt) | `experiments/runs/best318_multihead_p3p4p5_detecthead_geometry_v5_input64_sum_e50_lr1e4_r2/weights/best.pt` | `aec33ba87c3f534740a4501b6a545fcf9d7e2c27f080207b7613e7b4b06bb2c0` |

三个单头是三份独立联合权重；三头模型内部共享一套 YOLO 与 Detail、具有三个独立融合投影，但保存为**一个**联合 checkpoint。

## 可视化与边界

三头新增两个完整 Test 严格核验案例见[三头改善可视化](DIRECT_INPUT64_MULTI_IMPROVEMENT_VISUALS_20260916.md)：00601 消除建筑外墙错误框并保留正确玻璃；00890 的正确目标置信度从 0.679 提高到 0.742。此前单头真实改善仍见 [Test 00751](DIRECT_INPUT64_FP_SUPPRESSION_SEARCH_20260916.md) 和 [Val 00770 / 00657 / 02211](DIRECT_INPUT64_VAL_FP_SUPPRESSION_20260916.md)。

这些“融合前”都是**同一个联合 checkpoint 的第一次全局 forward**，不是独立训练的纯 YOLO baseline。完整三头 Test 逐图检查也发现局部退化，因此可视化只说明具体机制案例，不能替代上面的完整 Test mAP。
