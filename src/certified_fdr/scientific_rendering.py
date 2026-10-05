"""Post-result, theory-aligned rendering of immutable summaries.

This module does not alter or replace the prespecified v1 extraction.  It creates a
separately labelled descriptive view whose choices are recorded in
``theory_aligned_render_spec_v2.json``.
"""

from __future__ import annotations

import csv
import gzip
from io import StringIO
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .monte_carlo import bounded_mean_interval


EXPECTED_SUMMARY_ROWS = 360
REGIMES = ("valid", "boundary", "invalid")

METHOD_STYLE = {
    "Cert_wBY": {
        "label": "Cert-wBY (proposed)",
        "color": "#D55E00",
        "marker": "o",
        "fill": "white",
        "zorder": 5,
    },
    "old_both_q_internal": {
        "label": r"old router (both at $q^\circ$)",
        "color": "#7B3294",
        "marker": "D",
        "fill": "#7B3294",
        "zorder": 4,
    },
    "wBY_q": {
        "label": r"wBY($q$), safe direct",
        "color": "#6E6E6E",
        "marker": "s",
        "fill": "#6E6E6E",
        "zorder": 3,
    },
    "sign_oracle": {
        "label": "sign oracle",
        "color": "#0072B2",
        "marker": "^",
        "fill": "#0072B2",
        "zorder": 3,
    },
    "wBH_q_diagnostic": {
        "label": r"wBH($q$), diagnostic",
        "color": "#009E73",
        "marker": "x",
        "fill": "none",
        "zorder": 2,
    },
}


def load_summary(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != EXPECTED_SUMMARY_ROWS or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"expected {EXPECTED_SUMMARY_ROWS} summary rows; found {len(rows)}")
    return rows


def load_single_gzip_jsonl(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("application file must contain exactly one JSON object")
    return rows[0]


def _bounded_row(row: dict[str, Any], metric: str, family_size: int) -> dict[str, Any]:
    estimate = float(row[metric])
    interval = bounded_mean_interval(
        estimate,
        int(row["record_count"]),
        confidence=0.95,
        family_size=family_size,
    )
    return {
        "panel": "B" if metric == "mean_fdp" else "C",
        "module": row["module"],
        "configuration_id": row["configuration_id"],
        "regime": row["regime"],
        "method": row["method"],
        "metric": metric,
        "estimate": estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "record_count": int(row["record_count"]),
        "certificate_acceptance_count": int(row["certificate_acceptance_count"]),
        "certificate_acceptance_probability": float(
            row["certificate_acceptance_probability"]
        ),
        "interval_method": f"Hoeffding-Bonferroni; family_size={family_size}",
        "oracle_applicable": bool(row["oracle_applicable"]),
    }


def extract_theory_aligned_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the complete A grid and primary B comparisons in the v2 plan."""

    module_a = [row for row in summaries if row["module"] == "A"]
    if len(module_a) != 108:
        raise RuntimeError(f"expected 108 Module A cells; found {len(module_a)}")
    output: list[dict[str, Any]] = []
    for row in module_a:
        delta = float(row["Delta"])
        if not delta > 0.0:
            raise RuntimeError("Module A Delta must be positive")
        output.append(
            {
                "panel": "A",
                "module": "A",
                "configuration_id": row["configuration_id"],
                "regime": row["regime"],
                "method": row["method"],
                "metric": "certificate_acceptance_probability",
                "x": float(row["s_J"]) / delta,
                "d": int(row["d"]),
                "gamma": float(row["gamma"]),
                "N_C_over_N_upper": float(row["N_C_over_N_upper"]),
                "estimate": float(row["certificate_acceptance_probability"]),
                "lower": float(row["certificate_interval_lower"]),
                "upper": float(row["certificate_interval_upper"]),
                "interval_method": row["certificate_interval_method"],
                "oracle_applicable": bool(row["oracle_applicable"]),
            }
        )

    primary = [
        row
        for row in summaries
        if row["module"] == "B"
        and float(row["eta"]) == 1.0
        and float(row["noncentrality"]) == 2.0
    ]
    if len(primary) != 21:
        raise RuntimeError(f"expected 21 primary Module B method cells; found {len(primary)}")
    by_key = {(row["regime"], row["method"]): row for row in primary}
    if len(by_key) != 21:
        raise RuntimeError("primary Module B cells are not unique")

    fdr_methods = ("Cert_wBY", "old_both_q_internal", "wBY_q", "wBH_q_diagnostic")
    power_methods = (
        "Cert_wBY",
        "old_both_q_internal",
        "wBY_q",
        "sign_oracle",
        "wBH_q_diagnostic",
    )
    for regime in REGIMES:
        for method in fdr_methods:
            output.append(_bounded_row(by_key[(regime, method)], "mean_fdp", 21))
        for method in power_methods:
            output.append(_bounded_row(by_key[(regime, method)], "mean_power", 21))

    counts = {
        panel: sum(row["panel"] == panel for row in output)
        for panel in ("A", "B", "C")
    }
    if counts != {"A": 108, "B": 12, "C": 15}:
        raise RuntimeError(f"unexpected theory-aligned panel counts: {counts}")
    return sorted(
        output,
        key=lambda row: (
            row["panel"],
            str(row.get("regime")),
            str(row.get("method")),
            float(row.get("x", 0.0)),
        ),
    )


def panel_csv_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    values = list(rows)
    fields = (
        "panel",
        "module",
        "configuration_id",
        "regime",
        "method",
        "metric",
        "x",
        "d",
        "gamma",
        "N_C_over_N_upper",
        "estimate",
        "lower",
        "upper",
        "record_count",
        "certificate_acceptance_count",
        "certificate_acceptance_probability",
        "interval_method",
        "oracle_applicable",
    )
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in values:
        writer.writerow({field: row.get(field) for field in fields})
    return buffer.getvalue().encode("utf-8")


def _errorbar(ax: Any, x: float, row: dict[str, Any], method: str) -> None:
    style = METHOD_STYLE[method]
    estimate = float(row["estimate"])
    lower = float(row["lower"])
    upper = float(row["upper"])
    marker = style["marker"]
    markerfacecolor = style["fill"]
    if marker == "x":
        markerfacecolor = style["color"]
    ax.errorbar(
        x,
        estimate,
        yerr=np.asarray([[estimate - lower], [upper - estimate]]),
        color=style["color"],
        marker=marker,
        markerfacecolor=markerfacecolor,
        markeredgecolor=style["color"],
        markeredgewidth=1.0,
        linestyle="none",
        linewidth=0.9,
        elinewidth=0.9,
        capsize=2.0,
        markersize=4.5,
        zorder=style["zorder"],
    )


def render_theory_aligned_figure(rows: list[dict[str, Any]], destination: Path) -> None:
    counts = {panel: sum(row["panel"] == panel for row in rows) for panel in ("A", "B", "C")}
    if counts != {"A": 108, "B": 12, "C": 15}:
        raise RuntimeError(f"unexpected panel counts: {counts}")

    # Conventional journal figure: the graphic carries the estimates and
    # intervals; theorem-region prose, routing counts, and qualifications live
    # in the caption/table rather than being embedded as dashboard annotations.
    figure = plt.figure(figsize=(7.25, 4.80))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 1.05),
        width_ratios=(1.08, 0.92),
        left=0.105,
        right=0.985,
        top=0.955,
        bottom=0.175,
        hspace=0.48,
        wspace=0.31,
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, :])

    title_kw = {"loc": "center", "fontweight": "normal", "fontsize": 9.2, "pad": 5}
    label_size = 8.2
    tick_size = 7.4

    a_rows = [row for row in rows if row["panel"] == "A"]
    region_style = {
        "valid_interior": ("valid interior", "#0072B2", "o"),
        "boundary": ("boundary", "#E69F00", "s"),
        "invalid": ("invalid", "#009E73", "^")
    }
    for regime, (label, color, marker) in region_style.items():
        values = [row for row in a_rows if row["regime"] == regime]
        values.sort(key=lambda row: (float(row["x"]), int(row["d"]), float(row["gamma"])))
        x = np.asarray([float(row["x"]) for row in values])
        y = np.asarray([float(row["estimate"]) for row in values])
        lower = np.asarray([float(row["lower"]) for row in values])
        upper = np.asarray([float(row["upper"]) for row in values])
        ax_a.errorbar(
            x,
            y,
            yerr=np.vstack((y - lower, upper - y)),
            color=color,
            marker=marker,
            linestyle="none",
            alpha=0.78,
            linewidth=0.6,
            elinewidth=0.6,
            capsize=1.2,
            markersize=3.2,
            label=label,
            zorder=3,
        )
    ax_a.axvline(0.0, color="0.35", linewidth=0.75)
    ax_a.axvline(1.0, color="0.35", linestyle="--", linewidth=0.75)
    ax_a.axhline(0.01, color="0.55", linestyle=":", linewidth=0.7)
    ax_a.axhline(0.99, color="0.55", linestyle=":", linewidth=0.7)
    ax_a.set(
        xlabel=r"normalized margin $s_J/\Delta$",
        ylabel=r"certificate acceptance probability",
        ylim=(-0.035, 1.035),
    )
    ax_a.set_title("(a) Certificate acceptance", **title_kw)
    ax_a.legend(
        frameon=False,
        fontsize=7.0,
        loc="center left",
        bbox_to_anchor=(0.015, 0.53),
        borderaxespad=0.0,
        handletextpad=0.45,
    )

    b_rows = [row for row in rows if row["panel"] == "B"]
    fdr_methods = ("Cert_wBY", "old_both_q_internal", "wBY_q", "wBH_q_diagnostic")
    base = np.arange(3, dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(fdr_methods))
    for method_index, method in enumerate(fdr_methods):
        for regime_index, regime in enumerate(REGIMES):
            row = next(
                value for value in b_rows
                if value["regime"] == regime and value["method"] == method
            )
            _errorbar(ax_b, base[regime_index] + offsets[method_index], row, method)
    ax_b.axhline(0.10, color="0.2", linestyle="--", linewidth=0.9)
    ax_b.set_xticks(base, REGIMES)
    ax_b.set(ylabel="mean FDP", ylim=(-0.004, 0.108))
    ax_b.set_title(r"(b) Mean FDP", **title_kw)

    c_rows = [row for row in rows if row["panel"] == "C"]
    power_methods = (
        "Cert_wBY",
        "old_both_q_internal",
        "wBY_q",
        "sign_oracle",
        "wBH_q_diagnostic",
    )
    power_offsets = np.linspace(-0.28, 0.28, len(power_methods))
    for method_index, method in enumerate(power_methods):
        for regime_index, regime in enumerate(REGIMES):
            row = next(
                value for value in c_rows
                if value["regime"] == regime and value["method"] == method
            )
            descriptive_oracle = method == "sign_oracle" and not bool(
                row["oracle_applicable"]
            )
            if descriptive_oracle:
                style = METHOD_STYLE[method]
                estimate = float(row["estimate"])
                lower = float(row["lower"])
                upper = float(row["upper"])
                ax_c.errorbar(
                    base[regime_index] + power_offsets[method_index],
                    estimate,
                    yerr=np.asarray([[estimate - lower], [upper - estimate]]),
                    color=style["color"],
                    marker=style["marker"],
                    markerfacecolor="white",
                    markeredgecolor=style["color"],
                    markeredgewidth=1.0,
                    linestyle="none",
                    linewidth=0.9,
                    elinewidth=0.9,
                    capsize=2.0,
                    markersize=4.5,
                    alpha=0.48,
                    zorder=style["zorder"],
                )
            else:
                _errorbar(
                    ax_c,
                    base[regime_index] + power_offsets[method_index],
                    row,
                    method,
                )
    ax_c.set_xticks(base, REGIMES)
    ax_c.set(ylabel="mean power", ylim=(0.23, 0.57))
    ax_c.set_title(r"(c) Mean power", **title_kw)

    for axis in (ax_a, ax_b, ax_c):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.90", linewidth=0.5)
        axis.tick_params(labelsize=tick_size)
        axis.xaxis.label.set_size(label_size)
        axis.yaxis.label.set_size(label_size)

    handles = []
    labels = []
    short_labels = {
        "Cert_wBY": "Cert-wBY",
        "old_both_q_internal": r"old router ($q^\circ/q^\circ$)",
        "wBY_q": r"wBY($q$)",
        "sign_oracle": "sign oracle",
        "wBH_q_diagnostic": r"wBH($q$), diagnostic",
    }
    for method in power_methods:
        style = METHOD_STYLE[method]
        facecolor = style["fill"] if style["marker"] != "x" else style["color"]
        handle = plt.Line2D(
            [], [], color=style["color"], marker=style["marker"], linestyle="none",
            markerfacecolor=facecolor,
            markeredgecolor=style["color"], markersize=4.5,
        )
        handles.append(handle)
        labels.append(short_labels[method])
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.2,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.035),
        columnspacing=1.05,
        handletextpad=0.35,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _primary_c(summaries: list[dict[str, Any]], method: str) -> dict[str, Any]:
    matches = [
        row
        for row in summaries
        if row["module"] == "C"
        and float(row["eta"]) == 1.0
        and float(row["noncentrality"]) == 2.0
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"primary Module C row is not unique for {method}")
    return matches[0]


def module_c_table_bytes(summaries: list[dict[str, Any]]) -> bytes:
    proposed = _primary_c(summaries, "Cert_wBY")
    safe = _primary_c(summaries, "wBY_q")
    diagnostic = _primary_c(summaries, "wBH_q_diagnostic")
    old = _primary_c(summaries, "old_both_q_internal")
    method_rows = [
        ("Cert-wBY (proposed)", proposed),
        (r"old $q^\circ/q^\circ$ router", old),
        (r"direct wBY($q$)", safe),
        (r"diagnostic wBH($q$)", diagnostic),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        (r"Geometry & $(m,d)$ & $N_C$ & $\tau$ & $s_J$ & $\Delta$ & "
         r"$s_J/\Delta$ & $\widehat{\Pr}(C=1)$ [95\% CP] \\"),
        r"\midrule",
        ("HBN-frozen Gaussian & "
         f"{int(proposed['m'])}/{int(proposed['d'])} & {int(proposed['N_C'])} & "
         f"{float(proposed['tau']):.4f} & {float(proposed['s_J']):.4f} & "
         f"{float(proposed['Delta']):.4f} & "
         f"{float(proposed['s_J']) / float(proposed['Delta']):.4f} & "
         f"{float(proposed['certificate_acceptance_probability']):.4f} "
         f"[{float(proposed['certificate_interval_lower']):.4f},"
         f"{float(proposed['certificate_interval_upper']):.4f}] " + r"\\"),
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\par\smallskip",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Mean FDP [95\% HB] & Mean power [95\% HB] \\",
        r"\midrule",
    ]
    for label, row in method_rows:
        fdp = bounded_mean_interval(
            float(row["mean_fdp"]), int(row["record_count"]),
            confidence=0.95, family_size=63,
        )
        power_interval = bounded_mean_interval(
            float(row["mean_power"]), int(row["record_count"]),
            confidence=0.95, family_size=63,
        )
        lines.append(
            f"{label} & {float(row['mean_fdp']):.4f} "
            f"[{fdp.lower:.4f},{fdp.upper:.4f}] & "
            f"{float(row['mean_power']):.4f} "
            f"[{power_interval.lower:.4f},{power_interval.upper:.4f}] " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
        (r"\par\smallskip\noindent\textit{Interpretation:} "
         r"At the primary $(\mu,\eta)=(2,1)$, $0<s_J<\Delta$, so the "
         r"certified--oracle comparison guarantee is unavailable. HB intervals use "
         r"the prespecified 63-cell Module C family per metric. The wBH value is "
         r"diagnostic only."),
        "",
    ])
    return "\n".join(lines).encode("utf-8")


def hbn_table_bytes(hbn: dict[str, Any]) -> bytes:
    if hbn.get("claim_label") != "model-based exploratory application":
        raise RuntimeError("HBN claim label changed")
    if hbn.get("contains_participant_ids") is not False:
        raise RuntimeError("HBN application contains participant identifiers")
    if hbn.get("contains_individual_rows") is not False:
        raise RuntimeError("HBN application contains individual rows")
    expected_methods = {
        "BY_q", "Cert_BY", "Cert_wBY", "old_both_q_internal",
        "wBH_q_diagnostic", "wBY_q",
    }
    for result in hbn["threshold_results"]:
        methods = result["methods"]
        if set(methods) != expected_methods:
            raise RuntimeError("HBN method set differs from the reviewed six methods")
        if any(
            int(methods[name]["rejection_count"]) != int(hbn["m"])
            for name in expected_methods
        ):
            raise RuntimeError("all-method HBN saturation claim is false")
    primary_eta = float(hbn["primary_eta"])
    rows = [row for row in hbn["threshold_results"] if float(row["eta"]) == primary_eta]
    rows.sort(key=lambda row: float(row["raw_correlation_threshold"]))
    counts = [int(row["methods"]["Cert_wBY"]["rejection_count"]) for row in rows]
    primary = next(
        row for row in rows
        if float(row["raw_correlation_threshold"])
        == float(hbn["primary_raw_correlation_threshold"])
    )
    split = hbn["split_sizes"]
    lines = [
        r"\begin{tabular}{lrrrrrrlll}",
        r"\toprule",
        (r"Application & $|D_W|/|D_C|/|D_T|$ & $(m,d)$ & paired $N_C$ & $\tau$ & "
         r"$\min_J\widehat\Gamma$ & margin & $C$/branch & $W_{\min},W_{\max}$ & "
         r"Cert-wBY rej. ($r_0=0,.1,.2$) \\"),
        r"\midrule",
        ("Exploratory & "
         f"{split['D_W']}/{split['D_C']}/{split['D_T']} & "
         f"{int(hbn['m'])}/{int(hbn['d'])} & {int(hbn['N_C'])} & "
         f"{float(hbn['tau']):.4f} & "
         f"{float(hbn['minimum_certified_cross_moment']):.4f} & "
         f"{float(hbn['minimum_certified_cross_moment']) - float(hbn['tau']):.4f} & "
         f"{int(bool(hbn['certificate_accepted']))}/{hbn['selected_branch']} & "
         f"{float(primary['weight_minimum']):.1f},{float(primary['weight_maximum']):.1f} & "
         + "/".join(map(str, counts)) + r"\\"),
        r"\bottomrule",
        r"\end{tabular}",
        "",
        (r"\par\smallskip\noindent\textit{Model-based exploratory application.} "
         r"All primary weights were uniform and all methods were saturated at 10 rejections; "
         r"this is not evidence of a learned-weight gain. The primary setting is "
         r"$(r_0,\eta)=(.1,1)$, and execution transforms $r_0$ by "
         r"$\operatorname{atanh}$. Family identifiers were unavailable, so family-wise split "
         r"separation could not be verified."),
        "",
    ]
    return "\n".join(lines).encode("utf-8")
