"""Plot the native image-resolution distribution as a bubble chart."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect(dataset_root: Path, splits: list[str]) -> tuple[Counter, dict]:
    resolutions: Counter = Counter()
    summary = {}
    for split in splits:
        image_dir = dataset_root / "images" / split
        images = sorted(path for path in image_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        for path in images:
            with Image.open(path) as image:
                resolutions[image.size] += 1
        summary[split] = {"images": len(images)}
    if not resolutions:
        raise ValueError("No images found")
    return resolutions, summary


def plot(resolutions: Counter, output: Path) -> None:
    plt.rcParams.update({
        "axes.linewidth": 1.5, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    })
    rows = sorted((width, height, count)
                  for (width, height), count in resolutions.items())
    widths = np.asarray([row[0] for row in rows], dtype=np.float64)
    heights = np.asarray([row[1] for row in rows], dtype=np.float64)
    counts = np.asarray([row[2] for row in rows], dtype=np.float64)

    # Matplotlib's `s` is marker area in pt^2. The variable part is linear in
    # image count; the small floor keeps singleton resolutions visible.
    areas = 18.0 + 1000.0 * counts / counts.max()

    # Match the 6 x 5 inch height and output dimensions of the ratio chart.
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set_facecolor("#F8F8F8")
    ax.grid(True, linestyle="-", linewidth=1.5, color="white", alpha=.9)
    ax.set_axisbelow(True)
    ax.scatter(widths, heights, s=areas, color="#4F68D9", alpha=.62,
               edgecolor="black", linewidth=.55, zorder=3)

    # Label only separated high-frequency modes; nearby 1080x1440 is visible
    # as an overlapping bubble but deliberately not given a colliding label.
    label_offsets = {
        (1920, 1080): (12, -24),
        (4000, 2250): (10, 10),
        (1080, 1439): (10, 12),
        (3024, 4032): (10, 10),
    }
    for resolution, offset in label_offsets.items():
        if resolution not in resolutions:
            continue
        width, height = resolution
        count = resolutions[resolution]
        ax.annotate(f"{width}×{height}\nn={count:,}", xy=resolution,
                    xytext=offset, textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.2,
                    bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                          "edgecolor": "#777777", "linewidth": .55, "alpha": .88},
                    arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": .55})

    ax.set_xlabel("Image Width (pixels)")
    ax.set_ylabel("Image Height (pixels)")
    ax.set_title("Image-Resolution Distribution", fontsize=12,
                 pad=15, fontweight="bold")
    ax.set_xlim(0, widths.max() * 1.06)
    ax.set_ylim(0, heights.max() * 1.06)
    for spine in ax.spines.values():
        spine.set_linewidth(.8)
        spine.set_color("black")
    # There is one dataset distribution, so no categorical legend is needed.
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output", type=Path, required=True,
                        help="output stem; PNG/SVG/PDF/JSON/CSV are generated")
    args = parser.parse_args()

    resolutions, splits = collect(args.dataset_root, args.splits)
    plot(resolutions, args.output)
    rows = sorted(((width, height, count)
                   for (width, height), count in resolutions.items()),
                  key=lambda row: (-row[2], row[0], row[1]))
    total = sum(resolutions.values())
    report = {
        "definition": "native stored image width and height in pixels",
        "splits": splits, "images": total,
        "unique_resolutions": len(resolutions),
        "width_range": [min(width for width, _ in resolutions),
                        max(width for width, _ in resolutions)],
        "height_range": [min(height for _, height in resolutions),
                         max(height for _, height in resolutions)],
        "most_common": [
            {"width": width, "height": height, "count": count,
             "percentage": 100 * count / total}
            for width, height, count in rows[:20]
        ],
        "bubble_area": "18 + 1000 * resolution_count / maximum_resolution_count (pt^2)",
        "figure_inches": [6.0, 5.0], "png_dpi": 300, "legend": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["width", "height", "count", "percentage"])
        for width, height, count in rows:
            writer.writerow([width, height, count, f"{100 * count / total:.6f}"])
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
