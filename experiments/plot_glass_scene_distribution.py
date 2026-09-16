"""Plot the reported six-class glass-scene distribution as a pie chart."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CATEGORIES = [
    ("Window", 47, "#4F68D9"),
    ("Glass door", 15, "#7399E6"),
    ("Glass curtain wall", 15, "#65B9B4"),
    ("Others", 10, "#F0B35F"),
    ("Glass railing", 7, "#DB7C78"),
    ("Glass ceiling", 5, "#9679C7"),
]


def plot(output: Path) -> None:
    plt.rcParams.update({
        "axes.linewidth": 1.5, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })
    values = [value for _, value, _ in CATEGORIES]
    colors = [color for _, _, color in CATEGORIES]
    labels = [f"{name}\n{value}%" for name, value, _ in CATEGORIES]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    fig.patch.set_facecolor("white")
    _, texts = ax.pie(
        values, labels=labels, colors=colors, startangle=90, counterclock=False,
        labeldistance=1.10, radius=.88,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9.5, "color": "#202530"},
    )
    for text in texts:
        text.set_ha("center")
    ax.set_title("Glass-Scene Distribution", fontsize=12,
                 pad=15, fontweight="bold")
    ax.axis("equal")
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
        "figure_inches": [6.0, 5.0], "png_dpi": 300, "legend": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["scene_type", "reported_percentage", "color"])
        writer.writerows((name, value, color) for name, value, color in CATEGORIES)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
