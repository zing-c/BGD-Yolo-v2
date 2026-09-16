"""Plot normalized GT broken-region bounding-box occupancy as a heatmap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect(dataset_root: Path, splits: list[str]) -> tuple[np.ndarray, dict]:
    boxes, summary = [], {}
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
                boxes.append((x, y, width, height))
                instances += 1
        summary[split] = {"images": len(images), "instances": instances}
    array = np.asarray(boxes, dtype=np.float64)
    if not len(array):
        raise ValueError("No broken-region annotations found")
    return array, summary


def occupancy_map(boxes: np.ndarray, grid_size: int) -> np.ndarray:
    """Increment every grid cell intersected by each normalized GT bbox."""
    delta = np.zeros((grid_size + 1, grid_size + 1), dtype=np.int32)
    for x, y, width, height in boxes:
        left, right = np.clip((x - width / 2, x + width / 2), 0.0, 1.0)
        top, bottom = np.clip((y - height / 2, y + height / 2), 0.0, 1.0)
        x0 = min(grid_size - 1, max(0, int(np.floor(left * grid_size))))
        y0 = min(grid_size - 1, max(0, int(np.floor(top * grid_size))))
        x1 = min(grid_size, max(x0 + 1, int(np.ceil(right * grid_size))))
        y1 = min(grid_size, max(y0 + 1, int(np.ceil(bottom * grid_size))))
        delta[y0, x0] += 1
        delta[y1, x0] -= 1
        delta[y0, x1] -= 1
        delta[y1, x1] += 1
    return delta.cumsum(0).cumsum(1)[:grid_size, :grid_size]


def plot(boxes: np.ndarray, output: Path, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    plt.rcParams.update({
        "axes.linewidth": 1.5, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    })
    counts = occupancy_map(boxes, grid_size)
    normalized = counts.astype(np.float64) / counts.max()
    cmap = LinearSegmentedColormap.from_list(
        "damage_occupancy", ["#FFF9E6", "#F6E6A9", "#E3EABD",
                             "#C7EFCF", "#9BD8B5", "#70B893", "#4D9470"])

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set_facecolor("#F8F8F8")
    image = ax.imshow(normalized, origin="lower", extent=(0, 1, 0, 1),
                      interpolation="nearest", cmap=cmap, aspect="equal",
                      vmin=0, vmax=1)
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
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.set_label("Normalized GT-Box Occupancy", fontsize=10)
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
    return counts, normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--grid-size", type=int, default=256,
                        help="square normalized occupancy raster size")
    parser.add_argument("--output", type=Path, required=True,
                        help="output stem; PNG/SVG/PDF/JSON/CSV are generated")
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")

    boxes, splits = collect(args.dataset_root, args.splits)
    counts, normalized = plot(boxes, args.output, args.grid_size)
    peak_locations = np.argwhere(counts == counts.max())
    peak_y, peak_x = peak_locations[0]
    cell = 1.0 / args.grid_size
    report = {
        "definition": "normalized spatial occurrence map from all GT bbox-covered grid cells",
        "formula": "H = bbox_coverage_count / maximum_bbox_coverage_count",
        "coordinate_origin": "top-left", "range": [0.0, 1.0],
        "grid_size_hw": [args.grid_size, args.grid_size], "cell_size": cell,
        "rasterization": "each bbox increments every grid cell it intersects",
        "smoothing": "none", "visual_interpolation": "nearest",
        "display_color_norm": "linear", "heat_value_range": [0.0, 1.0],
        "colormap": "pastel yellow-to-green (#FFF9E6 to #4D9470)",
        "one_is_not_probability": "1.0 is the maximum relative occurrence, not 100% of images",
        "splits": splits, "instances": int(len(boxes)),
        "maximum_coverage_count": int(counts.max()),
        "peak_cell_count": int(len(peak_locations)),
        "example_peak_cell": {
            "x_range": [float(peak_x * cell), float((peak_x + 1) * cell)],
            "y_range": [float(peak_y * cell), float((peak_y + 1) * cell)],
        },
        "figure_inches": [6.0, 5.0], "png_dpi": 300,
        "exact_coverage_counts_y_by_x": counts.tolist(),
        "normalized_occupancy_y_by_x": normalized.tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["x_start", "x_end", "y_start", "y_end",
                         "coverage_count", "normalized_occupancy"])
        for y in range(args.grid_size):
            for x in range(args.grid_size):
                writer.writerow([f"{x * cell:.6f}", f"{(x + 1) * cell:.6f}",
                                 f"{y * cell:.6f}", f"{(y + 1) * cell:.6f}",
                                 int(counts[y, x]), f"{normalized[y, x]:.8f}"])
    omitted = {"exact_coverage_counts_y_by_x", "normalized_occupancy_y_by_x"}
    print(json.dumps({key: value for key, value in report.items() if key not in omitted},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
