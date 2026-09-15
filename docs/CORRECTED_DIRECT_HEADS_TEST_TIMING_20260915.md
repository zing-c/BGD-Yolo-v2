# Corrected Direct 单头与三头：Test 结果及统一推理时间复测

更新日期：2026-09-15。状态：四个模型各完成三次完整 Test，共 12 次。

本页合并当前修正版 P3、P4、P5 单头与最新三头实验的 Test 指标、统一推理时间、计算量口径和权重位置。所有模型不使用 alpha，重叠 Detail 特征使用 `sum`，且使用 Val mAP50 选择的 `best.pt`。此次只复测，没有重新训练或修改权重。

> 当前速度对比请引用本页，不要混入不同日期旧日志中的单次时间。P4 的本次 AP 与旧记录有微小差异，见第 5 节；不能声称四个模型的历史指标全部完全复现。

## 1. 本次完整 Test 结果

指标为 0–1；时间为三次完整 Test 平均推理时间的“均值 ± 样本标准差”。每个模型自己的三次指标完全一致。

| 模型 | Precision | Recall | mAP50 | mAP75 | mAP50–95 | Inference（ms/张） | Detail crops/张 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3 correct sum | 0.941964286 | 0.901709402 | 0.932006081 | 0.905851622 | 0.877765368 | 53.819 ± 0.778 | 1.462 |
| P4 correct sum | 0.963122511 | 0.892883718 | 0.938120921 | 0.910521931 | 0.884211406 | 53.951 ± 0.091 | 1.407 |
| P5 correct sum | 0.950226244 | 0.897435897 | 0.934470698 | 0.916394337 | 0.884660911 | 53.806 ± 0.044 | 1.395 |
| P3+P4+P5 correct sum | 0.955156951 | 0.910256410 | 0.942651148 | 0.921439920 | 0.889608738 | 55.168 ± 0.063 | 1.318 |

三个单头均值最多相差 0.145 ms，且 P3 自身的重复波动更大，不宜宣称单头之间有显著速度排名。三头比 P4 多约 1.217 ms（2.26%），本次不支持“P4 明显最慢”的旧判断。

三头的 mAP50–95 为 88.9609%，比本次最好的单头 P5 高 0.4948 个百分点。这里是一次训练种子的结果，不代表统计显著性。

### 三次时间明细

| 模型 | Round 1 | Round 2 | Round 3 | 单位 |
|---|---:|---:|---:|---|
| P3 | 52.920791 | 54.291847 | 54.244424 | ms/张 |
| P4 | 53.891878 | 53.904194 | 54.055898 | ms/张 |
| P5 | 53.813640 | 53.758864 | 53.845672 | ms/张 |
| P3+P4+P5 | 55.096418 | 55.216603 | 55.189761 | ms/张 |

原始精度数值及各轮计时见 [公开结果 JSON](results/corrected_direct_heads_test_timing_20260915.json)。

## 2. Val 选权重与训练来源

本项目 fork 的 fitness 使用 **Val mAP50** 选择 `best.pt`，没有使用 Test 选择 epoch。本次也未根据复测结果重新选权重。下面为训练 CSV 中对应 best 的行，指标保留 CSV 的五位精度。

| 模型 | Best epoch（从 1 开始） | Val P | Val R | Val mAP50 | Val mAP75 | Val mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 7 | 0.97849 | 0.93365 | 0.97294 | 0.94732 | 0.91585 |
| P4 | 36 | 0.97535 | 0.93749 | 0.97393 | 0.94193 | 0.91317 |
| P5 | 32 | 0.97666 | 0.93839 | 0.97339 | 0.94048 | 0.91590 |
| P3+P4+P5 | 48 | 0.97546 | 0.94186 | 0.97308 | 0.94293 | 0.91644 |

CSV epoch 从 0 开始，例如三头 best 对应 CSV 的 epoch=47。

共同训练设置：50 epochs、imgsz=640、batch=32、AdamW、lr0=1e-4、lrf=0.01、momentum=0.937、weight decay=5e-4、warmup=3 epochs、warmup momentum=0.8、warmup bias LR=0、AMP enabled、seed=0、candidate conf=0.2。单头与三头初始化均来自 YOLO `yolo-runs/train/train10/weights/best.pt` 和 Detail `run/detail_net_attn.pt`，不是从 `best318.pt` 初始化。

四行是四个独立训练实验的联合 checkpoint。三头模型内部共享一个 YOLO 和一个 Detail encoder，具有三个独立的融合投影；并不是将三个独立单头模型串行运行。

## 3. 统一测试环境与计时边界

| 项目 | 设置 |
|---|---|
| 完成时间 | 2026-09-15 17:39:13–18:00:49，Asia/Tokyo |
| GPU | 同一块 NVIDIA RTX 6000 Ada Generation |
| Python / PyTorch | bgd 环境 / 2.7.0+cu128 |
| Dataset | `experiments/bg_local.yaml`，split=test |
| Test 数量 | 1598 images，234 instances |
| Input | imgsz=640，rectangular letterbox |
| Batch / precision | 1 / FP32，half=False |
| Workers / CPU threads | 8 / 8 |
| Candidate conf / final conf | 0.2 / 0.5 |
| Final NMS IoU | 0.5 |
| Augment / plots / save JSON / save TXT | False / False / False / False（验证器输出选项） |
| Warmup | 每次 20 张真实图，不计入时间 |
| cuDNN | benchmark=False，deterministic=True |
| Repeats | 每模型 3 次完整 Test，同样的图像顺序 |
| 时钟 | 没有锁定 GPU 时钟 |

三轮测试顺序分别为 P3→P4→P5→三头、P4→P5→三头→P3、P5→三头→P3→P4，以减轻顺序偏差。

`imgsz=640` 配合 rectangular letterbox 不等于每张图都强制输入 640×640；四个模型使用同一数据、相同形状处理。逐图形状记录在本地 `round*_images.json` 中。

时间使用验证器 CUDA 同步后的 inference 统计，包含模型内部 YOLO、Grad-CAM 反向、原图裁剪、Detail、scatter、融合和再次解码；不包含验证器外部的数据加载、预处理和最终 NMS。因此它不是完整请求的端到端延迟。标准差是在三次完整 Test 的平均时间之间计算，不是逐图标准差。

每次测试前、预热后、每 400 张及结束后检查 GPU 计算进程，12 次测试共记录 72 次检查。除本次测试外，没有检测到其他训练或推理进程；远程桌面服务保留运行。18:01:01 测试结束后 GPU 利用率为 0%，显存回落到 814 MiB，剩余为桌面服务。这是当时的记录，不是实时状态。

## 4. GFLOPs：静态神经网络 forward，不是完整方法总 FLOPs

以下为此前在固定 640×640 输入下使用 THOP 对实际模块 forward 的测量，按 1 MAC=2 FLOPs 计。它与第 3 节实际 rectangular Test 延迟不是同一个输入形状统计口径。

| 模型 | 固定 forward GFLOPs | 加上 N 个 Detail crops 后 |
|---|---:|---|
| P3 单头 | 8.5832704 | 8.5832704 + 0.043610496 × N |
| P4 单头 | 8.2097152 | 8.2097152 + 0.043610496 × N |
| P5 单头 | 8.1163264 | 8.1163264 + 0.043610496 × N |
| P3+P4+P5 三头 | 8.7389184 | 8.7389184 + 0.043610496 × N |

组成：YOLO 640×640 forward 为 8.0851968 GFLOPs；单个 64×64 Detail crop forward 为 0.043610496 GFLOPs；额外融合投影与所选最终分类器在 P3/P4/P5 分别为 0.4980736 / 0.1245184 / 0.0311296 GFLOPs。

N 是图像实际进入 Detail 的 crop 数，不是固定超参数。三头共享一次 Detail forward，不能把 N 乘 3。三头在三种尺度分别使用独立融合投影。

这些数值不包括 Grad-CAM backward、CAM window 搜索、scatter、第二次 decode 等自定义操作，**不能标为完整方法总 GFLOPs**。也不能将表中固定 640×640 数值直接作为实际 rectangular Test 的平均 GFLOPs。推理时间已实际包含这些模型内部操作，所以不能仅根据此静态 forward 表推断速度排名。

## 5. P4 旧记录与本次复测的差异

| P4 指标 | 旧 Test 日志 | 本次统一复测 |
|---|---:|---:|
| Precision | 0.963122511 | 0.963122511 |
| Recall | 0.892883718 | 0.892883718 |
| mAP50 | 0.938131377 | 0.938120921 |
| mAP75 | 0.910592132 | 0.910521931 |
| mAP50–95 | 0.884544732 | 0.884211406 |

mAP50–95 差 -0.033333 个百分点。P3、P5、三头与旧 Test 日志一致；P4 在本次三次复测中也完全一致，但不完全复现旧 AP。旧记录保留在 [原始单头结果页](CORRECTED_SINGLE_HEAD_SUM_RESULTS.md)，不直接覆盖。

P4 checkpoint 的 method 标记为 `best318_framework_corrected_single_head_direct_v2`。本次加载时使用当前仓库的统一 `patch_corrected_single_head_direct()`，中心使用 round、patch 大小按 stride 确定，并钳制完整 patch 的边界。历史 v2 与当前运行代码存在版本核对限制；尚未隔离差异来自历史 scatter 实现还是数值执行环境。本页代表当前代码和统一环境下的复测，不能据此认定训练权重改变，也不能声称差异来源已确定。

## 6. 具体权重位置

以下均为训练机器上的本地文件，路径相对于项目根目录 `/home/user/projects/czy/BGD-Yolo_v2_copy`。这些路径不代表本次已上传权重二进制到 GitHub；此次上传内容是 docs 文档、结果数据和计时脚本快照。

| 模型 | best.pt 本地路径 | 大小（bytes） | SHA-256 |
|---|---|---:|---|
| P3 | `experiments/runs/best318_high_p3_detecthead_corrected_xy_scale8_sum_e50_lr1e4_r1/weights/best.pt` | 685017411 | `75525a957aeef9d85c4abb6240633b2e931339a6643c444603ba0e1efb488604` |
| P4 | `experiments/runs/best318_mid_p4_detecthead_correctedscatter_direct_e50_lr1e4_r1/weights/best.pt` | 685018819 | `83ae89e54500be2e7ac8722c66ffa925563b4f53366cad44c9415cf391b5e130` |
| P5 | `experiments/runs/best318_low_p5_detecthead_corrected_xy_scale2_direct_e50_lr1e4_r2/weights/best.pt` | 685017411 | `7921171241048cf56cda758d3a995d79f0e653b4e91b3d79c8d7408901e527a8` |
| P3+P4+P5 | `experiments/runs/best318_multihead_p3p4p5_detecthead_corrected_xy_scaleaware_sum_e50_lr1e4_r3_w8/weights/best.pt` | 684152917 | `cc044dd4cbbb8aada51c3139cad2d29337d38708145ea0eeb07d9ec4e95cf10f` |

三头总参数 4,286,844；单头总参数 4,209,276。checkpoint 文件大小包含序列化运行时等历史对象，不能把约 685 MB 的文件直接等同于网络参数存储大小。

各模型完整训练记录位于对应 run 的 `results.csv`；三头最终 Test 日志为 `experiments/best318_multihead_p3p4p5_detecthead_corrected_xy_scaleaware_sum_e50_lr1e4_r3_w8.final_test.log`。

## 7. 数据、代码与复现

- [公开结果 JSON](results/corrected_direct_heads_test_timing_20260915.json)：包含精确指标、每轮各阶段速度、逐图 forward 分位数、实际 Detail crop 数与历史一致性标记。仅将路径改为仓库相对路径，并去除机器 GPU 进程身份；指标和计时值未改。
- [计时脚本原样快照](results/benchmark_corrected_direct_heads_20260915.py)：SHA-256 `d16dc74d757ece32cc3cbb0d6732521eb2ef807e14428f91892ee2a539e2d8dc`。原运行位置是 `experiments/benchmark_corrected_direct_heads.py`。
- 本地完整结果目录：`experiments/analysis/corrected_direct_timing_20260915_r1/`，其中 `results.json` 保留机器检查快照，12 个 `round*_images.json` 保留逐图路径、输入形状、延迟、crop 数。
- 本地复测日志：`experiments/corrected_direct_heads_controlled_timing_20260915_r1.log`。
- 模型运行代码对应 [实验分支提交 53c12b3](https://github.com/zing-c/BGD-Yolo-v2/tree/53c12b3)。main 的本次提交是文档与数据发布，不表示 main 的全部历史训练代码已同步。

脚本快照不能直接从 `docs/results` 运行：需要在上述实验代码版本中放回 `experiments/benchmark_corrected_direct_heads.py`，并准备相应数据、历史代码 ZIP 和权重。在同一 bgd 环境中运行：

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u experiments/benchmark_corrected_direct_heads.py \
  --rounds 3 --warmup-images 20 --workers 8 --threads 8 \
  --output experiments/analysis/corrected_direct_timing_repeat
```

output 必须是尚不存在的新目录。发现额外 GPU 训练/推理进程时脚本会报错停止，不会结束或杀掉其他用户进程。
