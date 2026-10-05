"""Render the prespecified moderate-dimensional stress-test outputs."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


M_VALUES = (20, 50, 100)
MARGINS = (-0.5, 0.5, 1.25)
COLORS = {-0.5: "#0072B2", 0.5: "#D55E00", 1.25: "#009E73"}


def _index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row["record_type"] == "certificate":
            key = ("certificate", int(row["m"]), float(row["normalized_margin"]))
        elif row["record_type"] == "method_metric":
            key = (
                "method_metric",
                int(row["m"]),
                float(row["normalized_margin"]),
                str(row["method"]),
                str(row["metric"]),
            )
        elif row["record_type"] == "power_contrast":
            key = ("power_contrast", int(row["m"]), float(row["normalized_margin"]))
        else:
            raise ValueError("unknown summary record type")
        if key in result:
            raise ValueError("duplicate summary key")
        result[key] = row
    if len(result) != 144:
        raise RuntimeError("expected 144 unique summary rows")
    return result


def panel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = _index(rows)
    result = []
    for m in M_VALUES:
        for margin in MARGINS:
            certificate = values[("certificate", m, margin)]
            cert_fdp = values[("method_metric", m, margin, "Cert_wBY", "mean_fdp")]
            cert_power = values[
                ("method_metric", m, margin, "Cert_wBY", "mean_power")
            ]
            direct_fdp = values[("method_metric", m, margin, "wBY_q", "mean_fdp")]
            direct_power = values[("method_metric", m, margin, "wBY_q", "mean_power")]
            contrast = values[("power_contrast", m, margin)]
            result.append(
                {
                    "m": m,
                    "d": int(certificate["d"]),
                    "N_C": int(certificate["N_C"]),
                    "normalized_margin": margin,
                    "record_count": int(certificate["record_count"]),
                    "certificate_acceptance": certificate["certificate_acceptance"],
                    "cert_wby_mean_fdp": cert_fdp["value"],
                    "cert_wby_mean_power": cert_power["value"],
                    "direct_wby_mean_fdp": direct_fdp["value"],
                    "direct_wby_mean_power": direct_power["value"],
                    "paired_power_difference": contrast["value"],
                }
            )
    return result


def render_figure(rows: list[dict[str, Any]], destination: Path) -> None:
    data = panel_rows(rows)
    positions = np.arange(len(M_VALUES), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(7.35, 2.55))
    figure.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.28, wspace=0.34)
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
            linewidth=1.0,
            markersize=4.0,
            capsize=2.0,
            label=rf"$x={margin:g}$",
        )
        for method, linestyle, marker in (
            ("cert_wby_mean_fdp", "-", markers[margin]),
            ("direct_wby_mean_fdp", "--", "x"),
        ):
            estimates = np.array([row[method]["estimate"] for row in cells])
            lower = np.array([row[method]["lower"] for row in cells])
            upper = np.array([row[method]["upper"] for row in cells])
            label = rf"$x={margin:g}$, " + ("Cert-wBY" if method.startswith("cert") else "wBY")
            axes[1].errorbar(
                positions,
                estimates,
                yerr=np.vstack((estimates - lower, upper - estimates)),
                color=COLORS[margin],
                linestyle=linestyle,
                marker=marker,
                markerfacecolor="white" if marker != "x" else COLORS[margin],
                linewidth=0.9,
                markersize=3.6,
                capsize=1.8,
                label=label,
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
            linewidth=1.0,
            markersize=4.0,
            capsize=2.0,
            label=rf"$x={margin:g}$",
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
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    axes[0].legend(handles_a, labels_a, frameon=False, fontsize=6.8, loc="best")
    axes[1].legend(frameon=False, fontsize=5.4, ncol=2, loc="best", columnspacing=0.6)
    axes[2].legend(frameon=False, fontsize=6.8, loc="best")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _format(value: dict[str, Any]) -> str:
    return (
        f"{float(value['estimate']):.4f} "
        f"[{float(value['lower']):.4f},{float(value['upper']):.4f}]"
    )


def latex_table_bytes(rows: list[dict[str, Any]]) -> bytes:
    data = panel_rows(rows)
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{@{}rrrrlllll@{}}",
        r"\toprule",
        r"$m$ & $d$ & $N_C$ & $x$ & Certificate [95\% CP] & Cert-wBY FDP [95\% HB] & Cert-wBY power [95\% HB] & wBY power [95\% HB] & Paired difference [95\% HB] \\",
        r"\midrule",
    ]
    for row in data:
        lines.append(
            f"{row['m']} & {row['d']} & {row['N_C']} & {row['normalized_margin']:.2f} & "
            f"{_format(row['certificate_acceptance'])} & "
            f"{_format(row['cert_wby_mean_fdp'])} & "
            f"{_format(row['cert_wby_mean_power'])} & "
            f"{_format(row['direct_wby_mean_power'])} & "
            f"{_format(row['paired_power_difference'])} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup", ""])
    return "\n".join(lines).encode("utf-8")


def panel_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    data = panel_rows(rows)
    output = StringIO()
    fields = [
        "m", "d", "N_C", "normalized_margin", "record_count",
        "certificate_acceptance", "certificate_lower", "certificate_upper",
        "cert_wby_mean_fdp", "cert_wby_fdp_lower", "cert_wby_fdp_upper",
        "cert_wby_mean_power", "cert_wby_power_lower", "cert_wby_power_upper",
        "direct_wby_mean_power", "direct_wby_power_lower", "direct_wby_power_upper",
        "paired_power_difference", "paired_power_lower", "paired_power_upper",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in data:
        writer.writerow(
            {
                "m": row["m"], "d": row["d"], "N_C": row["N_C"],
                "normalized_margin": row["normalized_margin"],
                "record_count": row["record_count"],
                "certificate_acceptance": row["certificate_acceptance"]["estimate"],
                "certificate_lower": row["certificate_acceptance"]["lower"],
                "certificate_upper": row["certificate_acceptance"]["upper"],
                "cert_wby_mean_fdp": row["cert_wby_mean_fdp"]["estimate"],
                "cert_wby_fdp_lower": row["cert_wby_mean_fdp"]["lower"],
                "cert_wby_fdp_upper": row["cert_wby_mean_fdp"]["upper"],
                "cert_wby_mean_power": row["cert_wby_mean_power"]["estimate"],
                "cert_wby_power_lower": row["cert_wby_mean_power"]["lower"],
                "cert_wby_power_upper": row["cert_wby_mean_power"]["upper"],
                "direct_wby_mean_power": row["direct_wby_mean_power"]["estimate"],
                "direct_wby_power_lower": row["direct_wby_mean_power"]["lower"],
                "direct_wby_power_upper": row["direct_wby_mean_power"]["upper"],
                "paired_power_difference": row["paired_power_difference"]["estimate"],
                "paired_power_lower": row["paired_power_difference"]["lower"],
                "paired_power_upper": row["paired_power_difference"]["upper"],
            }
        )
    return output.getvalue().encode("utf-8")
