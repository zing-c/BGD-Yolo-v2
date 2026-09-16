"""Plot normalized GT broken-region center locations as a 2D heatmap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect(dataset_root: Path, splits: list[str]) -> tuple[np.ndarray, dict]:
    centers, summary = [], {}
    for split in splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = sorted(path for path in image_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        instances = 0
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing label file for {image}")
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                row = line.split()
                if len(row) != 5:
                    raise ValueError(f"Expected YOLO class/xywh row in {label}: {row}")
                class_id, x, y, width, height = map(float, row)
                if class_id != 0 or not all(0 <= value <= 1 for value in (x, y, width, height)):
                    raise ValueError(f"Invalid one-class normalized label in {label}: {row}")
                centers.append((x, y))
                instances += 1
        summary[split] = {"images": len(images), "instances": instances}
    array = np.asarray(centers, dtype=np.float64)
    if not len(array):
        raise ValueError("No broken-region annotations found")
    return array, summary


def heatmap(centers: np.ndarray, output: Path, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plt.rcParams.update({
        "axes.linewidth": 1.5, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    })
    # Rows are normalized y intervals and columns are normalized x intervals.
    counts, y_edges, x_edges = np.histogram2d(
        centers[:, 1], centers[:, 0], bins=bins, range=((0.0, 1.0), (0.0, 1.0)))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set_facecolor("#F8F8F8")
    image = ax.imshow(counts, origin="lower", extent=(0, 1, 0, 1),
                      interpolation="bilinear", cmap="YlOrRd", aspect="equal",
                      norm=PowerNorm(gamma=.5, vmin=0, vmax=float(counts.max())))
    # Image coordinates conventionally start at the top-left.
    ax.invert_yaxis()
    ticks = np.linspace(0.0, 1.0, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{value:.1f}" for value in ticks])
    ax.set_yticklabels([f"{value:.1f}" for value in ticks])
    ax.set_xlabel("Normalized Image Width")
    ax.set_ylabel("Normalized Image Height")
    ax.set_title("Damage-Location Distribution", fontsize=12,
                 pad=15, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(.8)
        spine.set_color("#4B5563")
    colorbar = fig.colorbar(image, ax=ax, fraction=.052, pad=.035)
    colorbar.set_label("Broken-Region Centers per Bin", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.outline.set_linewidth(.6)
    colorbar.outline.set_edgecolor("#4B5563")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    return counts, x_edges, y_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--bins", type=int, default=20,
                        help="equal-width bins per normalized image axis")
    parser.add_argument("--output", type=Path, required=True,
                        help="output stem; PNG/SVG/PDF/JSON/CSV are generated")
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")

    centers, splits = collect(args.dataset_root, args.splits)
    counts, x_edges, y_edges = heatmap(centers, args.output, args.bins)
    peak_y, peak_x = np.unravel_index(np.argmax(counts), counts.shape)
    report = {
        "definition": "normalized center (x_center, y_center) of every GT broken-region bbox",
        "coordinate_origin": "top-left", "range": [0.0, 1.0],
        "bins_per_axis": args.bins, "bin_width": 1.0 / args.bins,
        "visual_interpolation": "bilinear; CSV/JSON retain exact bin counts",
        "display_color_norm": "PowerNorm gamma=0.5 to expose low-density bins",
        "splits": splits, "instances": int(len(centers)),
        "mean_center_xy": centers.mean(0).tolist(),
        "median_center_xy": np.median(centers, axis=0).tolist(),
        "peak_bin": {
            "x_range": [float(x_edges[peak_x]), float(x_edges[peak_x + 1])],
            "y_range": [float(y_edges[peak_y]), float(y_edges[peak_y + 1])],
            "count": int(counts[peak_y, peak_x]),
        },
        "figure_inches": [6.0, 5.0], "png_dpi": 300,
        "exact_bin_counts_y_by_x": counts.astype(int).tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["x_start", "x_end", "y_start", "y_end", "count"])
        for y in range(args.bins):
            for x in range(args.bins):
                writer.writerow([f"{x_edges[x]:.2f}", f"{x_edges[x + 1]:.2f}",
                                 f"{y_edges[y]:.2f}", f"{y_edges[y + 1]:.2f}",
                                 int(counts[y, x])])
    print(json.dumps({key: value for key, value in report.items()
                      if key != "exact_bin_counts_y_by_x"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
