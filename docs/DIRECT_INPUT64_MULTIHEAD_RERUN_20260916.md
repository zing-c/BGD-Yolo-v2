# Direct input64 三头：同配置复跑 r2 完成结果

状态：50 epoch 训练和完整 Test 均已完成，退出码均为 0。启动 2026-09-16 05:20:09 JST；训练完成 07:58:40；Test 完成 07:59:20。r2 为 fresh 训练，不从 r1、单头或中断 checkpoint 续训。

## r2 正式结果

`best.pt` 只按 Val mAP50 选择，来自第 40 轮（从 1 开始）：Val P=0.98259、R=0.93365、mAP50=0.97409、mAP75=0.93812、mAP50–95=0.91807。

完整 Test 使用 FP32、batch=1、candidate conf>0.2、最终 conf=0.5、NMS IoU=0.5：

| Precision | Recall | mAP50 | mAP75 | mAP50–95 | 单次 inference |
|---:|---:|---:|---:|---:|---:|
| 0.954159140 | 0.901709402 | 0.937602885 | 0.915409993 | 0.888222115 | 8.987700 ms/张 |

精确原始记录：[r2 status JSON](results/direct_input64_multi_r2_status_20260916.json)；完整 50 轮：[训练 CSV](results/direct_input64_multi_r2_training_20260916.csv)。后续同环境三轮完整 Test 复测为 7.959147 / 7.981569 / 7.977748，即 **7.972821±0.011996 ms/张**；三轮精度完全一致。最终延迟口径见[当前单头与三头总表](DIRECT_INPUT64_CURRENT_SINGLE_MULTI_RESULTS_20260916.md)。

## 与 r1 的确定性复跑核验

r1 与 r2 的 50 轮 CSV 字节一致，Val / Test 指标逐值一致；两个 `best.pt` 内全部 543 个 state tensor 逐值一致，最大绝对差为 0。checkpoint 文件 SHA 不同是序列化容器元数据不同，不表示 tensor 不同。r2 的单次 inference 8.987700 ms、r1 为 8.843240 ms，这种单次波动不能解释为模型不同。

这是同 seed=0、deterministic=True 的复现，证明本环境的确定性；它不是不同随机种子的统计重复，论文仍不能用 r1/r2 代替多 seed 均值±标准差。

## 配置

- 一个联合模型，共享 YOLO / Detail；P3/P4/P5 三个独立融合投影；一个联合 checkpoint。
- 50 epochs；imgsz=640；batch=32；AMP=True；workers=8；OMP / MKL threads=8，OpenBLAS=1。
- AdamW；lr0=0.0001；lrf=0.01；momentum=0.937；weight decay=0.0005；warmup=3 epoch，bias lr=0、momentum=0.8。
- 初始化：`yolo-runs/train/train10/weights/best.pt` + `run/detail_net_attn.pt`，不使用 r1 权重。
- Detect hook：`cv3[0][1]` / `cv3[1][1]` / `cv3[2][1]`，各 64ch。
- 原生 64×64 Detail crop；Detail 输出 3×4×4；按输入支持回填 P3=8×8、P4=4×4、P5=2×2。
- 三头 CAM resize 后平均聚合；重叠 Detail 回填为 sum。CAM mean 与 scatter sum 是两件事。
- 无 alpha 参数，也没有用 alpha=0 伪装删除；回归 logits 不替换。
- 参数量：训练 4,286,844；YOLO Conv–BN fused 后 4,281,644。

## 权重

训练机原文件：

`experiments/runs/best318_multihead_p3p4p5_detecthead_geometry_v5_input64_sum_e50_lr1e4_r2/weights/best.pt`

GitHub 可下载：[multi_r2_best.pt](weights/direct_input64_20260916/multi_r2_best.pt)，SHA256 `aec33ba87c3f534740a4501b6a545fcf9d7e2c27f080207b7613e7b4b06bb2c0`。只上传 best，不上传重复的 last。

## 可视化

[00601 / 00890 三头改善可视化](DIRECT_INPUT64_MULTI_IMPROVEMENT_VISUALS_20260916.md)包含完整 Test 严格筛选、真实 CAM、Detail 位置、全部原生截图和无外部白边的清晰红框。完整筛选如实报告 1 张改善、1,595 张不变、2 张变差，不能只展示正例后声称所有图片都改善。
