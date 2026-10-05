"""Post-audit display-only rendering for the moderate-dimensional stress test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from .moderate_dimension_rendering import COLORS, MARGINS, M_VALUES, panel_rows


def render_publication_figure(rows: list[dict[str, Any]], destination: Path) -> None:
    """Render the audited values with orthogonal margin and method legends."""

    data = panel_rows(rows)
    positions = np.arange(len(M_VALUES), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(7.35, 2.72))
    figure.subplots_adjust(
        left=0.075, right=0.99, top=0.78, bottom=0.25, wspace=0.35
    )
    markers = {-0.5: "o", 0.5: "s", 1.25: "^"}

    for margin in MARGINS:
        cells = [row for row in data if row["normalized_margin"] == margin]
        certificate = np.array(
            [row["certificate_acceptance"]["estimate"] for row in cells], dtype=float
        )
        cert_low = np.array(
            [row["certificate_acceptance"]["lower"] for row in cells], dtype=float
        )
        cert_high = np.array(
            [row["certificate_acceptance"]["upper"] for row in cells], dtype=float
        )
        axes[0].errorbar(
            positions,
            certificate,
            yerr=np.vstack((certificate - cert_low, cert_high - certificate)),
            color=COLORS[margin],
            marker=markers[margin],
            markerfacecolor="white",
            linewidth=1.05,
            markersize=4.2,
            capsize=2.0,
        )

        for key, linestyle in (
            ("cert_wby_mean_fdp", "-"),
            ("direct_wby_mean_fdp", "--"),
        ):
            estimates = np.array([row[key]["estimate"] for row in cells])
            lower = np.array([row[key]["lower"] for row in cells])
            upper = np.array([row[key]["upper"] for row in cells])
            axes[1].errorbar(
                positions,
                estimates,
                yerr=np.vstack((estimates - lower, upper - estimates)),
                color=COLORS[margin],
                linestyle=linestyle,
                marker=markers[margin],
                markerfacecolor="white",
                linewidth=0.95,
                markersize=3.8,
                capsize=1.8,
            )

        contrast = np.array(
            [row["paired_power_difference"]["estimate"] for row in cells]
        )
        contrast_low = np.array(
            [row["paired_power_difference"]["lower"] for row in cells]
        )
        contrast_high = np.array(
            [row["paired_power_difference"]["upper"] for row in cells]
        )
        axes[2].errorbar(
            positions,
            contrast,
            yerr=np.vstack((contrast - contrast_low, contrast_high - contrast)),
            color=COLORS[margin],
            marker=markers[margin],
            markerfacecolor="white",
            linewidth=1.05,
            markersize=4.2,
            capsize=2.0,
        )

    axes[1].axhline(0.10, color="0.35", linestyle=":", linewidth=0.8)
    axes[2].axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    axes[0].set_title("(A) Certificate acceptance", fontsize=8.6)
    axes[1].set_title("(B) Mean FDP", fontsize=8.6)
    axes[2].set_title("(C) Cert-wBY minus wBY power", fontsize=8.6)
    axes[0].set_ylabel("probability", fontsize=8.0)
    axes[1].set_ylabel("mean FDP", fontsize=8.0)
    axes[2].set_ylabel("paired mean-power difference", fontsize=8.0)
    axes[0].set_ylim(-0.035, 1.035)
    axes[1].set_ylim(-0.01, 0.115)

    for axis in axes:
        axis.set_xticks(positions, [str(value) for value in M_VALUES])
        axis.set_xlabel("number of hypotheses $m$", fontsize=8.0)
        axis.tick_params(labelsize=7.2)
        axis.grid(axis="y", color="0.90", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)

    margin_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[margin],
            marker=markers[margin],
            markerfacecolor="white",
            linewidth=1.05,
            markersize=4.2,
            label=rf"$x={margin:g}$",
        )
        for margin in MARGINS
    ]
    figure.legend(
        handles=margin_handles,
        title=r"normalized margin $x=s_J(\Gamma)/\Delta$",
        title_fontsize=7.2,
        frameon=False,
        fontsize=7.0,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.0,
        columnspacing=1.5,
    )
    method_handles = [
        Line2D(
            [0], [0], color="0.20", linestyle="-", linewidth=0.95,
            label="Cert-wBY"
        ),
        Line2D(
            [0], [0], color="0.20", linestyle="--", linewidth=0.95,
            label="direct wBY"
        ),
    ]
    axes[1].legend(
        handles=method_handles,
        frameon=False,
        fontsize=6.6,
        loc="upper right",
        handlelength=2.2,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
