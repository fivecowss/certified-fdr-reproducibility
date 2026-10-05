"""Conventional appendix rendering for prespecified routing summaries."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .supplemental_analysis import METHODS


FIGURE_METHODS = (
    "Cert_wBY",
    "old_both_q_internal",
    "wBY_q",
    "sign_oracle",
    "wBH_q_diagnostic",
)
STYLE = {
    "Cert_wBY": ("Cert-wBY", "#D55E00", "o", "-"),
    "old_both_q_internal": (r"old router ($q^\circ/q^\circ$)", "#7B3294", "D", "-"),
    "wBY_q": (r"wBY($q$)", "#6E6E6E", "s", "-"),
    "sign_oracle": ("sign oracle", "#0072B2", "^", "--"),
    "wBH_q_diagnostic": (r"wBH($q$), diagnostic", "#009E73", "x", ":"),
}


def load_summary(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 106 or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"expected 106 supplemental summary rows; found {len(rows)}")
    return rows


def summary_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        kind: sum(row["record_type"] == kind for row in rows)
        for kind in ("certificate", "method", "power_contrast")
    }
    if counts != {"certificate": 10, "method": 70, "power_contrast": 26}:
        raise RuntimeError(f"unexpected supplemental summary counts: {counts}")
    return counts


def render_supplemental_figure(rows: list[dict[str, Any]], destination: Path) -> None:
    summary_counts(rows)
    certificate = sorted(
        (row for row in rows if row["record_type"] == "certificate"),
        key=lambda row: float(row["normalized_margin"]),
    )
    method_rows = [row for row in rows if row["record_type"] == "method"]

    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.25, 3.10))
    figure.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.30, wspace=0.30)
    title_kw = {"fontsize": 9.2, "fontweight": "normal", "pad": 5}
    label_size = 8.2
    tick_size = 7.4

    x = np.asarray([float(row["normalized_margin"]) for row in certificate])
    y = np.asarray([float(row["certificate_acceptance"]["estimate"]) for row in certificate])
    lower = np.asarray([float(row["certificate_acceptance"]["lower"]) for row in certificate])
    upper = np.asarray([float(row["certificate_acceptance"]["upper"]) for row in certificate])
    ax_a.errorbar(
        x,
        y,
        yerr=np.vstack((y - lower, upper - y)),
        color="#0072B2",
        marker="o",
        markerfacecolor="white",
        markersize=4.2,
        linewidth=1.0,
        elinewidth=0.8,
        capsize=2.0,
    )
    ax_a.axvline(0.0, color="0.35", linewidth=0.75)
    ax_a.axvline(1.0, color="0.35", linestyle="--", linewidth=0.75)
    ax_a.axhline(0.01, color="0.55", linestyle=":", linewidth=0.7)
    ax_a.axhline(0.99, color="0.55", linestyle=":", linewidth=0.7)
    ax_a.set(
        xlabel=r"normalized margin $s_J/\Delta$",
        ylabel="certificate acceptance probability",
        ylim=(-0.035, 1.035),
    )
    ax_a.set_title("(a) Certificate transition", **title_kw)

    for method in FIGURE_METHODS:
        values = sorted(
            (
                row for row in method_rows
                if row["method"] == method
            ),
            key=lambda row: float(row["normalized_margin"]),
        )
        label, color, marker, linestyle = STYLE[method]
        method_x = np.asarray([float(row["normalized_margin"]) for row in values])
        estimates = np.asarray([float(row["mean_power"]["estimate"]) for row in values])
        lows = np.asarray([float(row["mean_power"]["lower"]) for row in values])
        highs = np.asarray([float(row["mean_power"]["upper"]) for row in values])
        if method == "sign_oracle":
            applicable = np.asarray([bool(row["oracle_applicable"]) for row in values])
            estimates = np.where(applicable, estimates, np.nan)
            lows = np.where(applicable, lows, np.nan)
            highs = np.where(applicable, highs, np.nan)
        ax_b.plot(
            method_x,
            estimates,
            color=color,
            marker=marker,
            markerfacecolor="white" if marker != "x" else color,
            markersize=3.8,
            linewidth=0.9,
            linestyle=linestyle,
            label=label,
        )
        ax_b.vlines(method_x, lows, highs, color=color, linewidth=0.65, alpha=0.8)
    ax_b.axvline(0.0, color="0.35", linewidth=0.75)
    ax_b.axvline(1.0, color="0.35", linestyle="--", linewidth=0.75)
    ax_b.set(
        xlabel=r"normalized margin $s_J/\Delta$",
        ylabel="mean power",
        ylim=(-0.02, 1.02),
    )
    ax_b.set_title("(b) Routing power", **title_kw)
    handles, labels = ax_b.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.8,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.745, 0.005),
        columnspacing=0.9,
        handletextpad=0.4,
    )

    for axis in (ax_a, ax_b):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.90", linewidth=0.5)
        axis.tick_params(labelsize=tick_size)
        axis.xaxis.label.set_size(label_size)
        axis.yaxis.label.set_size(label_size)

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def _format_interval(value: dict[str, Any]) -> str:
    return (
        f"{float(value['estimate']):.4f} "
        f"[{float(value['lower']):.4f},{float(value['upper']):.4f}]"
    )


def method_table_bytes(rows: list[dict[str, Any]]) -> bytes:
    summary_counts(rows)
    method_rows = sorted(
        (row for row in rows if row["record_type"] == "method"),
        key=lambda row: (float(row["normalized_margin"]), METHODS.index(row["method"])),
    )
    lines = [
        r"\begin{longtable}{rlccc}",
        r"\toprule",
        r"$s_J/\Delta$ & Method & Mean FDP [95\% HB] & Mean power [95\% HB] & Rej. fraction [95\% HB] \\",
        r"\midrule",
        r"\endhead",
    ]
    labels = {
        "BY_q": r"BY($q$)",
        "wBY_q": r"wBY($q$)",
        "wBH_q_diagnostic": r"wBH($q$), diagnostic",
        "Cert_wBY": "Cert-wBY",
        "Cert_BY": "Cert-BY",
        "sign_oracle": "sign oracle",
        "old_both_q_internal": r"old $q^\circ/q^\circ$ router",
    }
    for row in method_rows:
        lines.append(
            f"{float(row['normalized_margin']):.2f} & {labels[row['method']]} & "
            f"{_format_interval(row['mean_fdp'])} & "
            f"{_format_interval(row['mean_power'])} & "
            f"{_format_interval(row['mean_rejection_fraction'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    return "\n".join(lines).encode("utf-8")


def contrast_table_bytes(rows: list[dict[str, Any]]) -> bytes:
    summary_counts(rows)
    contrast_rows = sorted(
        (row for row in rows if row["record_type"] == "power_contrast"),
        key=lambda row: (float(row["normalized_margin"]), row["contrast"]),
    )
    lines = [
        r"\begin{longtable}{rllc}",
        r"\toprule",
        r"$s_J/\Delta$ & Left method & Right method & Mean power difference [95\% HB] \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in contrast_rows:
        lines.append(
            f"{float(row['normalized_margin']):.2f} & {row['left_method']} & "
            f"{row['right_method']} & {_format_interval(row['mean_power_difference'])} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    return "\n".join(lines).encode("utf-8")
