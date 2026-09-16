# Damage-Location Distribution

![Damage-location distribution](assets/dataset_statistics/damage_location_distribution.png)

该图是当前正式 Train、Val、Test 全部破损玻璃 GT bounding boxes 的 **normalized spatial occurrence map**。每张图片首先表示为统一的归一化 `1×1` 坐标空间，坐标原点位于左上角；每个 GT 框覆盖到的所有网格均累加 1：

\[
C(u,v)=\sum_{i=1}^{N}\mathbf 1[(u,v)\in \hat B_i].
\]

最终仅用全图最大覆盖计数进行归一化：

\[
H(u,v)=\frac{C(u,v)}{\max_{u,v}C(u,v)}.
\]

因此色条固定为 0–1，但 `1.0` 表示相对最高覆盖频率，并不表示 100% 的图片都在该位置包含破损。这里应称为 **normalized occurrence map**，而不是 probability map。

本数据的最高网格覆盖计数为 1,818，因此图中的 `1.0` 对应 1,818 个 GT 框覆盖该相对位置。最高值出现在图像中央附近；热图的连续变化来自大量不同大小 bbox 的自然叠加，而不是平滑操作。

## 计算设置

- 数据：Train 1,462、Val 422、Test 234，共 2,118 个 GT 框；
- 统一画布：`256×256` 归一化网格；
- 累计方式：每个 bbox 覆盖到的全部网格计数加 1；
- 平滑：无 Gaussian、无 KDE、无其他平滑；
- 显示插值：`nearest`，不使用 bilinear；
- 色阶：线性 `0–1`，浅蓝→蓝紫→紫红；
- 画布：6×5 inches、300 DPI，PNG 为 1800×1500。

| Split | 图片数 | GT 框 |
|---|---:|---:|
| Train | 1,832 | 1,462 |
| Val | 503 | 422 |
| Test | 1,598 | 234 |
| 合计 | 3,933 | 2,118 |

该图统计的是矩形 GT 框的空间覆盖频率，不是框中心 KDE、Grad-CAM 或像素级破损 mask。较大的 bbox 会覆盖更多网格，这是 occupancy map 所表达的预期含义。CSV 和 JSON 同时保存每个网格的原始覆盖次数与 0–1 归一化值。

下载：[PNG](assets/dataset_statistics/damage_location_distribution.png) · [SVG](assets/dataset_statistics/damage_location_distribution.svg) · [PDF](assets/dataset_statistics/damage_location_distribution.pdf) · [精确网格 CSV](assets/dataset_statistics/damage_location_distribution.csv) · [统计 JSON](assets/dataset_statistics/damage_location_distribution.json)。可复现脚本：[plot_damage_location_distribution.py](../experiments/plot_damage_location_distribution.py)。
