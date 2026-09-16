"""Plot the reported six-class glass-scene distribution as a pie chart."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CATEGORIES = [
    ("Window", 47, "#EAD8C9"),
    ("Glass door", 15, "#C7EFCF"),
    ("Glass curtain wall", 15, "#FFC7C7"),
    ("Others", 10, "#D8CEF4"),
    ("Glass railing", 7, "#C6DDF2"),
    ("Glass ceiling", 5, "#F6E6A9"),
]


def plot(output: Path) -> None:
    plt.rcParams.update({
        "axes.linewidth": 1.5, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })
    values = [value for _, value, _ in CATEGORIES]
    colors = [color for _, _, color in CATEGORIES]
    angular_percentages = np.asarray(values, dtype=np.float64) / sum(values) * 100
    display_names = ["Window", "Glass\ndoor", "Glass curtain\nwall",
                     "Others", "Glass\nrailing", "Glass\nceiling"]
    fig, ax = plt.subplots(figsize=(7.2, 7.6), dpi=180)
    fig.patch.set_facecolor("white")
    wedges, _ = ax.pie(
        values, colors=colors, startangle=112, counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 2.2},
    )
    for wedge, name, reported, angular_pct in zip(
            wedges, display_names, values, angular_percentages):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        if angular_pct >= 30:
            radius, fontsize = .58, 25
        elif angular_pct >= 14:
            radius, fontsize = .64, 19
        elif angular_pct >= 9:
            radius, fontsize = .70, 19
        else:
            radius, fontsize = .76, 17
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        ax.text(x, y, f"{name}\n{reported}%", ha="center", va="center",
                fontsize=fontsize, fontfamily="DejaVu Sans", color="black",
                linespacing=1.12)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.25, 1.08)
    fig.tight_layout(pad=.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="output stem; PNG/SVG/PDF/JSON/CSV are generated")
    args = parser.parse_args()
    plot(args.output)

    supplied_total = sum(value for _, value, _ in CATEGORIES)
    report = {
        "source": "user-supplied rounded scene shares",
        "supplied_percent_total": supplied_total,
        "rounding_note": "reported integer percentages sum to 99; labels are preserved verbatim",
        "pie_geometry": "Matplotlib normalizes the 47:15:15:10:7:5 ratio to 360 degrees",
        "categories": [
            {"name": name, "reported_percentage": value, "color": color,
             "normalized_angular_share_percentage": 100 * value / supplied_total}
            for name, value, color in CATEGORIES
        ],
        "figure_inches": [7.2, 7.6], "png_dpi": 180, "legend": False,
        "labels": "inside wedges", "hatches": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["scene_type", "reported_percentage", "color"])
        writer.writerows((name, value, color) for name, value, color in CATEGORIES)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
