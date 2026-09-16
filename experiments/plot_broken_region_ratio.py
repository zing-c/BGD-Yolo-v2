"""Plot the GT broken-region bounding-box area ratio on a fixed 0--1 axis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect(dataset_root: Path, splits: list[str]) -> tuple[np.ndarray, dict]:
    values, summary = [], {}
    for split in splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = sorted(path for path in image_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        instances, positive = 0, 0
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing label file for {image}")
            rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
            positive += bool(rows)
            for row in rows:
                if len(row) != 5:
                    raise ValueError(f"Expected YOLO class/xywh row in {label}: {row}")
                class_id, _, _, width, height = map(float, row)
                if class_id != 0 or not (0 <= width <= 1 and 0 <= height <= 1):
                    raise ValueError(f"Invalid one-class normalized label in {label}: {row}")
                ratio = width * height
                if not 0 <= ratio <= 1:
                    raise ValueError(f"Area ratio outside [0,1] in {label}: {ratio}")
                values.append(ratio)
                instances += 1
        summary[split] = {
            "images": len(images), "positive_images": positive,
            "background_images": len(images) - positive, "instances": instances,
        }
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("No broken-region annotations found")
    return array, summary


def plot(values: np.ndarray, output: Path, bins: int) -> tuple[np.ndarray, np.ndarray]:
    plt.rcParams.update({
        "axes.linewidth": 1.5, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    })
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * .90

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_facecolor("#F8F8F8")
    ax.grid(True, linestyle="-", linewidth=1.5, color="white", alpha=.9, axis="y")
    ax.set_axisbelow(True)
    bars = ax.bar(centers, counts, width=width, color="#4F68D9",
                  edgecolor="black", linewidth=.6, zorder=3)
    for bar, count in zip(bars, counts):
        ax.annotate(str(int(count)),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Ratio of a Broken Region to Its Whole Image")
    ax.set_ylabel("Number of Broken Regions")
    ax.set_title("Distribution of Broken-Region Ratio", fontsize=12,
                 pad=15, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.set_xticklabels([f"{value:.1f}" for value in np.linspace(0.0, 1.0, 11)])
    ax.set_ylim(0, max(counts) * 1.16)
    for spine in ax.spines.values():
        spine.set_linewidth(.8)
        spine.set_color("black")
    # Deliberately no legend: there is one measured distribution.
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    # Matplotlib path data and the csv module otherwise emit whitespace that
    # makes `git diff --check` noisy; this does not alter SVG geometry.
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    return edges, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--bins", type=int, default=20,
                        help="equal-width bins over the fixed [0,1] range")
    parser.add_argument("--output", type=Path, required=True,
                        help="output stem; PNG/SVG/PDF/JSON/CSV are generated")
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    values, splits = collect(args.dataset_root, args.splits)
    edges, counts = plot(values, args.output, args.bins)
    report = {
        "definition": "one GT bbox area / its whole image area = normalized_width * normalized_height",
        "range": [0.0, 1.0], "bins": args.bins, "splits": splits,
        "instances": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": {str(q): float(np.quantile(values, q))
                      for q in (0, .25, .5, .75, .9, .95, .99, 1)},
        "bin_edges": edges.tolist(), "bin_counts": counts.tolist(),
        "legend": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["bin_start", "bin_end", "count", "percentage"])
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            writer.writerow([f"{left:.2f}", f"{right:.2f}", int(count),
                             f"{100 * count / len(values):.6f}"])
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
