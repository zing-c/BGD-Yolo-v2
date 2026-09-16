# Damage-Location Distribution

![Damage-location distribution](assets/dataset_statistics/damage_location_distribution.png)

该图统计当前正式 Train、Val、Test 中每个破损玻璃 GT bounding box 的归一化中心点 `(x_center, y_center)`。坐标原点位于图片左上角：横轴 0→1 表示从左到右，纵轴 0→1 表示从上到下。

共统计 2,118 个 GT 框，使用 20×20 个等宽网格，每格覆盖归一化宽高 0.05×0.05。每个网格的热力值定义为 `该网格中心点数量 / 最高密度网格数量`，因此色条固定为 0–1；1.0 对应最高网格的 253 个框。图片采用浅蓝→蓝紫→紫红的冷色系渐变，并使用 bilinear 插值和 `PowerNorm(gamma=0.55)` 使低密度位置仍然可见。CSV 和 JSON 同时保存归一化热力值与变换前的精确网格计数。

| Split | 图片数 | GT 框 |
|---|---:|---:|
| Train | 1,832 | 1,462 |
| Val | 503 | 422 |
| Test | 1,598 | 234 |
| 合计 | 3,933 | 2,118 |

全部中心点的均值为 `(0.499635, 0.492942)`，median 为 `(0.500391, 0.500347)`。最高密度网格为 `x=0.50–0.55、y=0.50–0.55`，包含 253 个 GT 框，说明当前数据的框中心明显集中在图像中央附近。图中的水平和垂直高密度带来自大量中心坐标接近 0.5 的真实标签，并非绘图伪影。

这里展示的是**破损框中心位置分布**，不是框覆盖面积热力图，也不是像素级破损 mask。画布为 6×5 inches、300 DPI，与其他两张数据集分布图高度和 PNG 输出尺寸一致。

下载：[PNG](assets/dataset_statistics/damage_location_distribution.png) · [SVG](assets/dataset_statistics/damage_location_distribution.svg) · [PDF](assets/dataset_statistics/damage_location_distribution.pdf) · [精确分箱 CSV](assets/dataset_statistics/damage_location_distribution.csv) · [统计 JSON](assets/dataset_statistics/damage_location_distribution.json)。可复现脚本：[plot_damage_location_distribution.py](../experiments/plot_damage_location_distribution.py)。
