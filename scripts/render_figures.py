#!/usr/bin/env python3
"""Render manuscript figures from immutable frozen summaries.

This is a display-only transformation: it performs no simulation, aggregation,
interval calculation, or scientific-value change.
"""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from certified_fdr.moderate_dimension_rendering import (  # noqa: E402
    COLORS as MD_COLORS,
    MARGINS as MD_MARGINS,
    M_VALUES,
    panel_rows as moderate_panel_rows,
)


COLUMN_WIDTH = 3.25
TEXT_WIDTH = 6.75
ROUTING_MARGINS = (-1.50, -1.00, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00, 1.25)
CERT_D = (5, 20, 100)
CERT_GAMMA = (0.05, 0.10, 0.20)
CERT_RATIOS = (0.125, 0.25, 0.5, 1.0)

ROUTING_STYLES = {
    "Cert_wBY": ("Cert-wBY", "#E66100", "o", "-", "white"),
    "wBY_q": (r"direct wBY($q$)", "#6B6B6B", "s", "-", "white"),
    "old_both_q_internal": (
        "two-discounted branch",
        "#7B3294",
        "D",
        "-",
        "#7B3294",
    ),
    "sign_oracle": (
        "population-sign branch selector",
        "#0571B0",
        "^",
        "--",
        "white",
    ),
    "wBH_q_diagnostic": (
        r"wBH($q$), diagnostic",
        "#009E73",
        "x",
        ":",
        "none",
    ),
}

CERT_STYLES = {
    5: ("#0072B2", "o", "-", "white"),
    20: ("#D55E00", "s", "--", "white"),
    100: ("#009E73", "^", ":", "#009E73"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_summary(
    summary: Path,
    manifest_path: Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("summary_compressed_sha256", manifest.get("summary_sha256"))
    require(expected_hash == digest(summary), "summary hash differs from its sealed manifest")
    require(
        int(manifest["summary_record_count"]) == expected_rows,
        "manifest summary-row count mismatch",
    )
    return manifest


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _save(figure: Any, pdf: Path) -> None:
    figure.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Author": "certified-fdr-reproducibility",
            "Creator": Path(__file__).name,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        pdf.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"Software": Path(__file__).name},
    )
    plt.close(figure)


def _routing_data(rows: list[dict[str, Any]]) -> tuple[dict[float, float], dict[tuple[float, str], dict[str, float]]]:
    require(len(rows) == 106, "routing summary must contain 106 rows")
    certificates = [row for row in rows if row.get("record_type") == "certificate"]
    methods = [row for row in rows if row.get("record_type") == "method"]
    require(len(certificates) == 10, "routing certificate summary count must be 10")
    require(len(methods) == 70, "routing method summary count must be 70")
    acceptance = {
        float(row["normalized_margin"]): float(row["certificate_acceptance"]["estimate"])
        for row in certificates
    }
    values = {
        (float(row["normalized_margin"]), str(row["method"])): {
            "fdp": float(row["mean_fdp"]["estimate"]),
            "power": float(row["mean_power"]["estimate"]),
        }
        for row in methods
    }
    require(set(acceptance) == set(ROUTING_MARGINS), "routing margin grid mismatch")
    expected = {(x, method) for x in ROUTING_MARGINS for method in ROUTING_STYLES}
    require(expected.issubset(values), "routing method grid is incomplete")
    return acceptance, values


def _routing_axis_style(axis: Any) -> None:
    axis.axvline(0.0, color="0.25", linewidth=0.75)
    axis.axvline(1.0, color="0.25", linewidth=0.75, linestyle="--")
    axis.grid(axis="y", color="0.88", linewidth=0.45)
    axis.spines[["top", "right"]].set_visible(False)


def render_routing(rows: list[dict[str, Any]], destination: Path, single_column: bool) -> None:
    acceptance, values = _routing_data(rows)
    x = np.asarray(ROUTING_MARGINS, dtype=float)
    if single_column:
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(COLUMN_WIDTH, 5.0),
            sharex=True,
            constrained_layout=True,
        )
    else:
        figure = plt.figure(figsize=(TEXT_WIDTH, 4.1))
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05), hspace=0.48, wspace=0.31)
        axes = np.asarray(
            [
                figure.add_subplot(grid[0, 0]),
                figure.add_subplot(grid[0, 1]),
                figure.add_subplot(grid[1, :]),
            ]
        )
        figure.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.22)

    axes[0].plot(x, [acceptance[v] for v in ROUTING_MARGINS], color="#0072B2", marker="o", markerfacecolor="white")
    axes[0].set_ylim(-0.035, 1.035)
    axes[0].set_title("(A) Certificate acceptance")
    axes[0].set_ylabel("acceptance probability")
    handles = []
    for method, (label, color, marker, linestyle, markerface) in ROUTING_STYLES.items():
        fdp = np.asarray([values[(v, method)]["fdp"] for v in ROUTING_MARGINS])
        power = np.asarray([values[(v, method)]["power"] for v in ROUTING_MARGINS])
        if method == "sign_oracle":
            domain = (x < 0.0) | (x >= 1.0)
            fdp = np.where(domain, fdp, np.nan)
            power = np.where(domain, power, np.nan)
        (line,) = axes[1].plot(
            x, fdp, color=color, marker=marker, linestyle=linestyle,
            markerfacecolor=markerface, markeredgewidth=0.9, label=label,
        )
        handles.append(line)
        axes[2].plot(
            x, power, color=color, marker=marker, linestyle=linestyle,
            markerfacecolor=markerface, markeredgewidth=0.9,
        )

    axes[1].axhline(0.10, color="0.20", linewidth=0.8, linestyle="--")
    axes[1].set_ylim(0.0, 0.11)
    axes[1].set_title("(B) Mean FDP")
    axes[1].set_ylabel("mean FDP")
    axes[2].set_ylim(0.18, 0.55)
    axes[2].set_title("(C) Mean power")
    axes[2].set_ylabel("mean power")
    for axis in axes:
        _routing_axis_style(axis)
    axes[-1].set_xlabel(r"normalized margin $s_J/\Delta$")

    if single_column:
        figure.legend(
            handles=handles,
            labels=[item[0] for item in ROUTING_STYLES.values()],
            ncol=2,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
        )
    else:
        axes[0].set_xlabel(r"normalized margin $s_J/\Delta$")
        axes[1].set_xlabel(r"normalized margin $s_J/\Delta$")
        figure.legend(
            handles=handles,
            labels=[item[0] for item in ROUTING_STYLES.values()],
            ncol=3,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
        )
    _save(figure, destination)


def _certificate_panel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(len(rows) == 360, "primary summary must contain 360 rows")
    module_a = [row for row in rows if row.get("module") == "A"]
    require(len(module_a) == 108, "Module A summary must contain 108 rows")
    panel = [row for row in module_a if row.get("regime") == "valid_interior"]
    require(len(panel) == 36, "positive-margin certificate panel must contain 36 rows")
    require(all(int(row["record_count"]) == 200000 for row in panel), "certificate record-count mismatch")
    expected = {(d, gamma, ratio) for d in CERT_D for gamma in CERT_GAMMA for ratio in CERT_RATIOS}
    observed = {
        (int(row["d"]), float(row["gamma"]), float(row["N_C_over_N_upper"]))
        for row in panel
    }
    require(observed == expected, "certificate panel grid mismatch")
    anchors = {
        0.125: (0.0, 0.00028),
        0.25: (0.018135, 0.232000),
        0.5: (0.986870, 0.999625),
        1.0: (1.0, 1.0),
    }
    for ratio, anchor in anchors.items():
        estimates = [
            float(row["certificate_acceptance_probability"])
            for row in panel
            if _same(row["N_C_over_N_upper"], ratio)
        ]
        require(
            _same(min(estimates), anchor[0]) and _same(max(estimates), anchor[1]),
            f"certificate acceptance anchor failed at r={ratio}",
        )
    return panel


def render_certificate(rows: list[dict[str, Any]], destination: Path) -> None:
    panel = _certificate_panel(rows)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(COLUMN_WIDTH, 4.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    positions = np.arange(4, dtype=float)
    handles = []
    for panel_index, (axis, gamma) in enumerate(zip(axes, CERT_GAMMA, strict=True)):
        for d in CERT_D:
            cells = [
                row for row in panel
                if int(row["d"]) == d and _same(row["gamma"], gamma)
            ]
            cells.sort(key=lambda row: float(row["N_C_over_N_upper"]))
            color, marker, linestyle, markerface = CERT_STYLES[d]
            (line,) = axis.plot(
                positions,
                [float(row["certificate_acceptance_probability"]) for row in cells],
                color=color,
                marker=marker,
                linestyle=linestyle,
                markerfacecolor=markerface,
                markeredgecolor=color,
                markeredgewidth=0.85,
                label=str(d),
            )
            if panel_index == 0:
                handles.append(line)
        axis.set_title(rf"$\gamma={gamma:.2f}$", loc="left", pad=2.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks((0.0, 0.5, 1.0))
        axis.grid(axis="y", color="0.88", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(positions, ("1/8", "1/4", "1/2", "1"))
    figure.legend(
        handles=handles,
        labels=[str(d) for d in CERT_D],
        title=r"$d$",
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    figure.supylabel("certificate acceptance probability")
    figure.supxlabel(r"nominal sample-size multiplier $r$")
    _save(figure, destination)


def render_moderate(rows: list[dict[str, Any]], destination: Path) -> None:
    require(len(rows) == 144, "moderate-dimensional summary must contain 144 rows")
    data = moderate_panel_rows(rows)
    positions = np.arange(len(M_VALUES), dtype=float)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(COLUMN_WIDTH, 4.6),
        sharex=True,
        constrained_layout=True,
    )
    markers = {-0.5: "o", 0.5: "s", 1.25: "^"}
    for margin in MD_MARGINS:
        cells = [row for row in data if row["normalized_margin"] == margin]
        certificate = np.asarray([row["certificate_acceptance"]["estimate"] for row in cells])
        cert_low = np.asarray([row["certificate_acceptance"]["lower"] for row in cells])
        cert_high = np.asarray([row["certificate_acceptance"]["upper"] for row in cells])
        axes[0].errorbar(
            positions,
            certificate,
            yerr=np.vstack((certificate - cert_low, cert_high - certificate)),
            color=MD_COLORS[margin],
            marker=markers[margin],
            markerfacecolor="white",
            capsize=2.0,
        )
        for key, linestyle in (("cert_wby_mean_fdp", "-"), ("direct_wby_mean_fdp", "--")):
            estimate = np.asarray([row[key]["estimate"] for row in cells])
            lower = np.asarray([row[key]["lower"] for row in cells])
            upper = np.asarray([row[key]["upper"] for row in cells])
            axes[1].errorbar(
                positions,
                estimate,
                yerr=np.vstack((estimate - lower, upper - estimate)),
                color=MD_COLORS[margin],
                linestyle=linestyle,
                marker=markers[margin],
                markerfacecolor="white",
                capsize=2.0,
            )
        contrast = np.asarray([row["paired_power_difference"]["estimate"] for row in cells])
        lower = np.asarray([row["paired_power_difference"]["lower"] for row in cells])
        upper = np.asarray([row["paired_power_difference"]["upper"] for row in cells])
        axes[2].errorbar(
            positions,
            contrast,
            yerr=np.vstack((contrast - lower, upper - contrast)),
            color=MD_COLORS[margin],
            marker=markers[margin],
            markerfacecolor="white",
            capsize=2.0,
        )

    axes[1].axhline(0.10, color="0.30", linestyle=":", linewidth=0.8)
    axes[2].axhline(0.0, color="0.30", linestyle=":", linewidth=0.8)
    axes[0].set_title("(A) Certificate acceptance", loc="left")
    axes[1].set_title("(B) Mean FDP", loc="left")
    axes[2].set_title("(C) Paired power difference", loc="left")
    axes[0].set_ylabel("probability")
    axes[1].set_ylabel("mean FDP")
    axes[2].set_ylabel("Cert-wBY - direct wBY")
    axes[0].set_ylim(-0.035, 1.035)
    axes[1].set_ylim(-0.01, 0.115)
    for axis in axes:
        axis.set_xticks(positions, [str(value) for value in M_VALUES])
        axis.grid(axis="y", color="0.90", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("number of hypotheses $m$")

    margin_handles = [
        Line2D(
            [0], [0], color=MD_COLORS[margin], marker=markers[margin],
            markerfacecolor="white", label=rf"$x={margin:g}$",
        )
        for margin in MD_MARGINS
    ]
    figure.legend(
        handles=margin_handles,
        title=r"normalized margin $x=s_J(\Gamma)/\Delta$",
        frameon=False,
        ncol=3,
        loc="outside upper center",
    )
    method_handles = [
        Line2D([0], [0], color="0.20", linestyle="-", label="Cert-wBY"),
        Line2D([0], [0], color="0.20", linestyle="--", label="direct wBY"),
    ]
    axes[1].legend(
        handles=method_handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        loc="upper right",
        handlelength=2.0,
    )
    _save(figure, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-summary", type=Path, required=True)
    parser.add_argument("--routing-manifest", type=Path, required=True)
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--moderate-summary", type=Path, required=True)
    parser.add_argument("--moderate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--figure1-layout",
        choices=("full", "single"),
        default="full",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "routing_summary": args.routing_summary.resolve(),
        "routing_manifest": args.routing_manifest.resolve(),
        "primary_summary": args.primary_summary.resolve(),
        "primary_manifest": args.primary_manifest.resolve(),
        "moderate_summary": args.moderate_summary.resolve(),
        "moderate_manifest": args.moderate_manifest.resolve(),
    }
    output = args.output_dir.resolve()
    require(not output.exists() or not any(output.iterdir()), "output directory must be absent or empty")
    verify_summary(inputs["routing_summary"], inputs["routing_manifest"], expected_rows=106)
    verify_summary(inputs["primary_summary"], inputs["primary_manifest"], expected_rows=360)
    verify_summary(inputs["moderate_summary"], inputs["moderate_manifest"], expected_rows=144)
    routing_rows = read_rows(inputs["routing_summary"])
    primary_rows = read_rows(inputs["primary_summary"])
    moderate_rows = read_rows(inputs["moderate_summary"])
    output.mkdir(parents=True, exist_ok=True)

    outputs = {
        "figure1": output / "fig_routing_transition_main_v1.pdf",
        "figure2": output / "fig_certificate_calibration_main_v1.pdf",
        "figure3": output / "fig_moderate_dimension_stress_singlecol_v2.pdf",
    }
    render_routing(routing_rows, outputs["figure1"], args.figure1_layout == "single")
    render_certificate(primary_rows, outputs["figure2"])
    render_moderate(moderate_rows, outputs["figure3"])

    manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "classification": "FROZEN_SUMMARY_RENDER",
        "scientific_metrics_changed": False,
        "source_assets_overwritten": False,
        "figure1_layout": args.figure1_layout,
        "inputs": {
            key: {"filename": path.name, "sha256": digest(path)}
            for key, path in inputs.items()
        },
        "outputs": {
            path.name: {"pdf_sha256": digest(path), "png_sha256": digest(path.with_suffix('.png'))}
            for path in outputs.values()
        },
    }
    manifest_path = output / "figure_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("FIGURE_RENDER_GATE=PASS")
    print("SCIENTIFIC_METRICS_CHANGED=FALSE")
    print("SOURCE_ASSETS_OVERWRITTEN=FALSE")
    print(f"FIGURE1_LAYOUT={args.figure1_layout}")
    for name, path in outputs.items():
        print(f"{name.upper()}={path}")
        print(f"{name.upper()}_SHA256={digest(path)}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
