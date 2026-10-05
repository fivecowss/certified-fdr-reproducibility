"""Deterministic scientific figure and LaTeX-table rendering helpers."""

from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PANEL_COLUMNS = (
    "panel", "x", "facet", "series", "metric", "estimate", "lower", "upper",
    "interval_method", "diagnostic", "claim_label",
)

APPENDIX_COLUMNS = (
    "appendix", "module", "configuration_id", "regime", "d", "N_C",
    "gamma", "N_C_over_N_upper", "eta", "noncentrality", "method", "metric",
    "estimate", "lower", "upper", "interval_method", "diagnostic",
    "oracle_applicable",
)


def panel_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PANEL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in PANEL_COLUMNS})
    return output.getvalue().encode("utf-8")


def appendix_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=APPENDIX_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in APPENDIX_COLUMNS})
    return output.getvalue().encode("utf-8")


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def method_table_bytes(rows: list[dict[str, str]]) -> bytes:
    lines = [
        r"\begin{tabular}{lllll}",
        r"\toprule",
        (
            r"Method & Learned weights & Dependence input & "
            r"Arbitrary-dependence guarantee & Certificate \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                _latex_escape(row[key])
                for key in (
                    "method", "learned_weights", "dependence_input",
                    "arbitrary_dependence_guarantee", "uses_certificate",
                )
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines).encode("utf-8")


def hbn_table_bytes(row: dict[str, Any]) -> bytes:
    split = row["split_sizes"]
    counts = row["rejection_counts"]
    split_text = f"{split['D_W']}/{split['D_C']}/{split['D_T']}"
    count_text = "/".join(
        str(counts[key]) for key in ("raw_z0=0", "raw_z0=0.1", "raw_z0=0.2")
    )
    lines = [
        r"\begin{tabular}{lrrrrlll}",
        r"\toprule",
        (
            r"$N_W/N_C^{\rm raw}/N_T$ & $d$ & $N_C$ & $\tau$ & "
            r"$\min\widehat\Gamma$ & $C$ & Branch & Rejections ($z_0=0,.1,.2$) \\"
        ),
        r"\midrule",
        (
            f"{split_text} & {row['d']} & {row['N_C']} & {row['tau']:.4f} & "
            f"{row['minimum_cross_moment']:.4f} & {row['C']} & "
            f"{_latex_escape(row['selected_branch'])} & {count_text} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\par\smallskip\noindent\textit{Selected ROIs:} "
        + ", ".join(_latex_escape(value) for value in row["ROI_list"])
        + ".",
        "",
        r"\par\smallskip\noindent\textit{Limitation:} "
        + _latex_escape(row["family_information_limitation"]),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _errorbar(ax, x, rows, colors, labels=True):
    for index, row in enumerate(rows):
        estimate = float(row["estimate"])
        lower = float(row["lower"])
        upper = float(row["upper"])
        ax.errorbar(
            x[index], estimate,
            yerr=np.array([[estimate - lower], [upper - estimate]]),
            fmt="o", color=colors[index], capsize=2.5,
            label=row["series"] if labels else None,
        )


def render_main_figure(rows: list[dict[str, Any]], destination: Path) -> None:
    figure = plt.figure(figsize=(11.0, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])
    palette = plt.get_cmap("tab10")

    a_rows = [row for row in rows if row["panel"] == "A"]
    for color_index, regime in enumerate(("valid_interior", "boundary", "invalid")):
        values = sorted(
            (row for row in a_rows if row["series"] == regime),
            key=lambda row: float(row["x"]),
        )
        x = np.asarray([float(row["x"]) for row in values])
        y = np.asarray([float(row["estimate"]) for row in values])
        lo = np.asarray([float(row["lower"]) for row in values])
        hi = np.asarray([float(row["upper"]) for row in values])
        ax_a.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=2,
                      color=palette(color_index), label=regime.replace("_", " "))
    ax_a.axvline(1.0, color="0.45", linestyle="--", linewidth=1)
    ax_a.set(xlabel=r"$N_C/N_{\rm upper}$", ylabel="Certificate acceptance", ylim=(-0.03, 1.03))
    ax_a.legend(frameon=False, fontsize=8)
    ax_a.set_title("A  Certificate transition", loc="left", fontweight="bold")

    b_rows = [row for row in rows if row["panel"] == "B"]
    regimes = ("valid", "boundary", "invalid")
    methods = ("BY_q", "wBY_q", "wBH_q_diagnostic", "Cert_wBY", "Cert_BY", "sign_oracle")
    base = np.arange(len(regimes), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(methods))
    for method_index, method in enumerate(methods):
        values = [
            next(
                row
                for row in b_rows
                if row["x"] == regime and row["series"] == method
            )
            for regime in regimes
        ]
        y = np.asarray([float(row["estimate"]) for row in values])
        lo = np.asarray([float(row["lower"]) for row in values])
        hi = np.asarray([float(row["upper"]) for row in values])
        label = method.replace("_q_diagnostic", " (diagnostic)")
        ax_b.errorbar(base + offsets[method_index], y, yerr=np.vstack([y - lo, hi - y]),
                      fmt="o", capsize=2, color=palette(method_index), label=label)
    ax_b.axhline(0.1, color="0.35", linestyle="--", linewidth=1, label="q=0.10")
    ax_b.set_xticks(base, regimes)
    ax_b.set(ylabel="Mean FDP", ylim=(-0.01, None))
    ax_b.legend(frameon=False, fontsize=6.8, ncol=2)
    ax_b.set_title("B  End-to-end FDR", loc="left", fontweight="bold")

    ax_c.set_axis_off()
    ax_c.set_title("C  End-to-end power", loc="left", fontweight="bold")
    c_rows = [row for row in rows if row["panel"] == "C"]
    c_lower = max(0.0, min(float(row["lower"]) for row in c_rows) - 0.03)
    c_upper = min(1.0, max(float(row["upper"]) for row in c_rows) + 0.03)
    for regime_index, regime in enumerate(regimes):
        inset = ax_c.inset_axes([regime_index / 3 + 0.02, 0.08, 0.29, 0.80])
        for method_index, method in enumerate(("Cert_wBY", "wBY_q", "wBH_q_diagnostic")):
            values = sorted(
                (row for row in c_rows if row["facet"] == regime and row["series"] == method),
                key=lambda row: float(row["x"]),
            )
            x = np.asarray([float(row["x"]) for row in values])
            y = np.asarray([float(row["estimate"]) for row in values])
            lo = np.asarray([float(row["lower"]) for row in values])
            hi = np.asarray([float(row["upper"]) for row in values])
            inset.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                marker="o",
                capsize=2,
                color=palette(method_index + 3),
                label=method if regime_index == 0 else None,
            )
        inset.set_title(regime, fontsize=8)
        inset.set_xticks([0.0, 0.5, 1.0])
        inset.set_ylim(c_lower, c_upper)
        inset.set_xlabel(r"$\eta$", fontsize=8)
        inset.tick_params(labelsize=7)
        if regime_index == 0:
            inset.set_ylabel("Mean power", fontsize=8)
            inset.legend(frameon=False, fontsize=6)

    d_rows = [row for row in rows if row["panel"] == "D"]
    order = ("tau", "population_s_J", "Delta", "observed_minimum_cross_moment")
    values = [next(row for row in d_rows if row["x"] == name) for name in order]
    labels = [r"$\tau$", r"population $s_J$", r"$\Delta$", r"observed $\min\widehat\Gamma$"]
    colors = ["0.35", palette(0), "0.35", palette(3)]
    ax_d.scatter(np.arange(4), [row["estimate"] for row in values], c=colors, s=48)
    ax_d.set_xticks(np.arange(4), labels, rotation=18, ha="right")
    ax_d.set_ylabel("Certificate scale")
    ax_d.set_title("D  HBN-calibrated bridge and application", loc="left", fontweight="bold")
    ax_d.text(0.02, 0.98, r"$0\leq s_J<\Delta$: no oracle-completeness guarantee",
              transform=ax_d.transAxes, va="top", fontsize=8)

    for axis in (ax_a, ax_b, ax_d):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
