"""Display-only rendering for scientifically reviewed outputs."""

from __future__ import annotations

import csv
import gzip
from io import StringIO
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PANEL_COUNTS = {"A": 12, "B": 18, "C": 27, "D": 4}

METHOD_STYLE = {
    "BY_q": {"label": "BY", "color": "#0072B2", "marker": "o", "linestyle": "-"},
    "wBY_q": {"label": "wBY", "color": "#E69F00", "marker": "s", "linestyle": ":"},
    "wBH_q_diagnostic": {
        "label": "wBH (diagnostic)", "color": "#009E73", "marker": "x",
        "linestyle": "--",
    },
    "Cert_wBY": {
        "label": "Cert-wBY", "color": "#D55E00", "marker": "o",
        "linestyle": "-", "markerfacecolor": "none", "markeredgewidth": 1.1,
    },
    "Cert_BY": {"label": "Cert-BY", "color": "#CC79A7", "marker": "D", "linestyle": "-."},
    "sign_oracle": {"label": "sign oracle", "color": "#6B4C3B", "marker": "^", "linestyle": "-"},
}


def load_panel_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = {panel: sum(row["panel"] == panel for row in rows) for panel in PANEL_COUNTS}
    if counts != PANEL_COUNTS or len(rows) != sum(PANEL_COUNTS.values()):
        raise RuntimeError(f"unexpected panel counts: {counts}")
    return rows


def load_single_gzip_jsonl(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("HBN application must contain exactly one JSON object.")
    return rows[0]


def _number(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _interval(ax, x, row, style, *, label=None, zorder=2) -> None:
    estimate = _number(row, "estimate")
    lower = _number(row, "lower")
    upper = _number(row, "upper")
    kwargs = dict(style)
    kwargs.pop("label", None)
    ax.errorbar(
        x, estimate,
        yerr=np.array([[estimate - lower], [upper - estimate]]),
        label=label, capsize=1.8, linewidth=0.9, markersize=4.0,
        zorder=zorder, **kwargs,
    )


def render_publication_figure(rows: list[dict[str, Any]], destination: Path) -> None:
    figure = plt.figure(figsize=(7.2, 5.35), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    title_kw = {"loc": "left", "fontweight": "bold", "fontsize": 9}
    label_size = 7.5
    tick_size = 6.8

    a_rows = [row for row in rows if row["panel"] == "A"]
    a_styles = {
        "valid_interior": ("valid interior", "#0072B2", "o", "-"),
        "boundary": ("boundary", "#E69F00", "s", "--"),
        "invalid": ("invalid", "#009E73", "^", ":"),
    }
    for regime, (label, color, marker, linestyle) in a_styles.items():
        values = sorted(
            (row for row in a_rows if row["series"] == regime),
            key=lambda row: _number(row, "x"),
        )
        x = np.asarray([_number(row, "x") for row in values])
        y = np.asarray([_number(row, "estimate") for row in values])
        lo = np.asarray([_number(row, "lower") for row in values])
        hi = np.asarray([_number(row, "upper") for row in values])
        ax_a.errorbar(
            x, y, yerr=np.vstack([y - lo, hi - y]), color=color,
            marker=marker, linestyle=linestyle, linewidth=1.0,
            markersize=3.8, capsize=1.7, label=label,
        )
    ax_a.axvline(1.0, color="0.45", linestyle="--", linewidth=0.8)
    ax_a.set(xlabel=r"$N_C/N_{\rm upper}$", ylabel="Certificate acceptance", ylim=(-0.03, 1.03))
    ax_a.legend(frameon=False, fontsize=6.2, loc="center right")
    ax_a.set_title("A  Certificate transition", **title_kw)

    b_rows = [row for row in rows if row["panel"] == "B"]
    regimes = ("valid", "boundary", "invalid")
    methods = ("BY_q", "wBY_q", "wBH_q_diagnostic", "Cert_wBY", "Cert_BY", "sign_oracle")
    base = np.arange(len(regimes), dtype=float)
    offsets = np.linspace(-0.25, 0.25, len(methods))
    for method_index, method in enumerate(methods):
        style = METHOD_STYLE[method]
        values = [next(row for row in b_rows if row["x"] == regime and row["series"] == method)
                  for regime in regimes]
        y = np.asarray([_number(row, "estimate") for row in values])
        lo = np.asarray([_number(row, "lower") for row in values])
        hi = np.asarray([_number(row, "upper") for row in values])
        kwargs = dict(style)
        label = kwargs.pop("label")
        kwargs["linestyle"] = "none"
        ax_b.errorbar(
            base + offsets[method_index], y, yerr=np.vstack([y - lo, hi - y]),
            label=label, capsize=1.6, linewidth=0, markersize=3.8,
            zorder=3, **kwargs,
        )
    ax_b.axhline(0.1, color="0.25", linestyle="--", linewidth=0.8)
    ax_b.text(0.02, 0.97, r"target $q=0.10$", transform=ax_b.transAxes,
              va="top", fontsize=6.4)
    ax_b.set_xticks(base, regimes)
    ax_b.set(ylabel="Mean FDP", ylim=(-0.005, 0.108))
    ax_b.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper center",
                bbox_to_anchor=(0.62, 0.94), columnspacing=0.8, handletextpad=0.35)
    ax_b.set_title("B  End-to-end FDR", **title_kw)

    ax_c.set_axis_off()
    ax_c.set_title("C  End-to-end power", **title_kw)
    c_rows = [row for row in rows if row["panel"] == "C"]
    c_lower = max(0.0, min(_number(row, "lower") for row in c_rows) - 0.025)
    c_upper = min(1.0, max(_number(row, "upper") for row in c_rows) + 0.025)
    c_methods = ("wBH_q_diagnostic", "wBY_q", "Cert_wBY")
    legend_handles = None
    legend_labels = None
    for regime_index, regime in enumerate(regimes):
        inset = ax_c.inset_axes([regime_index / 3 + 0.015, 0.25, 0.305, 0.61])
        for method in c_methods:
            values = sorted(
                (row for row in c_rows if row["facet"] == regime and row["series"] == method),
                key=lambda row: _number(row, "x"),
            )
            x = np.asarray([_number(row, "x") for row in values])
            y = np.asarray([_number(row, "estimate") for row in values])
            lo = np.asarray([_number(row, "lower") for row in values])
            hi = np.asarray([_number(row, "upper") for row in values])
            style = dict(METHOD_STYLE[method])
            label = style.pop("label")
            inset.errorbar(
                x, y, yerr=np.vstack([y - lo, hi - y]), label=label,
                capsize=1.5, linewidth=0.9, markersize=3.5,
                zorder=4 if method == "Cert_wBY" else 2, **style,
            )
        inset.set_title(regime, fontsize=6.8, pad=2)
        inset.set_xticks([0.0, 0.5, 1.0])
        inset.set_ylim(c_lower, c_upper)
        inset.set_xlabel(r"$\eta$", fontsize=label_size)
        inset.tick_params(labelsize=6.1, pad=1)
        if regime_index == 0:
            inset.set_ylabel("Mean power", fontsize=label_size)
            legend_handles, legend_labels = inset.get_legend_handles_labels()
        else:
            inset.tick_params(labelleft=False)
        if regime == "valid":
            inset.text(0.04, 0.96, "Cert-wBY = wBH", transform=inset.transAxes,
                       va="top", fontsize=5.4)
        inset.spines[["top", "right"]].set_visible(False)
        inset.grid(axis="y", color="0.91", linewidth=0.45)
    ax_c.legend(legend_handles, legend_labels, frameon=False, fontsize=5.6,
                ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                columnspacing=0.7, handletextpad=0.3)

    d_rows = [row for row in rows if row["panel"] == "D"]
    order = ("tau", "population_s_J", "Delta", "observed_minimum_cross_moment")
    values = [next(row for row in d_rows if row["x"] == name) for name in order]
    estimates = [_number(row, "estimate") for row in values]
    labels = [r"$\tau$", r"population $s_J$", r"$\Delta$", r"observed $\min\widehat\Gamma$"]
    colors = ["0.35", "#0072B2", "0.35", "#D55E00"]
    markers = ["o", "o", "D", "o"]
    for index, (estimate, color, marker) in enumerate(zip(estimates, colors, markers)):
        ax_d.scatter(index, estimate, color=color, marker=marker, s=28, zorder=3)
    tau = estimates[0]
    observed = estimates[3]
    ax_d.axhline(tau, color="0.65", linestyle=":", linewidth=0.7)
    ax_d.set_xticks(np.arange(4), labels, rotation=17, ha="right")
    ax_d.set(ylabel="Certificate scale", ylim=(0.165, 0.455))
    ax_d.set_title("D  HBN bridge and application", **title_kw)
    ax_d.text(0.02, 0.98, r"$0\leq s_J<\Delta$: no oracle-completeness guarantee",
              transform=ax_d.transAxes, va="top", fontsize=6.2)
    ax_d.text(0.98, 0.07, rf"HBN: $C=1$, margin={observed - tau:.4f}",
              transform=ax_d.transAxes, ha="right", fontsize=6.2)

    for axis in (ax_a, ax_b, ax_d):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.91", linewidth=0.45)
        axis.tick_params(labelsize=tick_size)
        axis.xaxis.label.set_size(label_size)
        axis.yaxis.label.set_size(label_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight",
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def _latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _primary_threshold_result(hbn: dict[str, Any]) -> dict[str, Any]:
    eta = float(hbn["primary_eta"])
    threshold = float(hbn["primary_raw_correlation_threshold"])
    matches = [row for row in hbn["threshold_results"]
               if float(row["eta"]) == eta and float(row["raw_correlation_threshold"]) == threshold]
    if len(matches) != 1:
        raise RuntimeError("primary HBN sensitivity row is not unique.")
    return matches[0]


def publication_hbn_table_bytes(hbn: dict[str, Any]) -> bytes:
    primary = _primary_threshold_result(hbn)
    split = hbn["split_sizes"]
    threshold_rows = {
        float(row["raw_correlation_threshold"]): row
        for row in hbn["threshold_results"] if float(row["eta"]) == float(hbn["primary_eta"])
    }
    counts = []
    for threshold in (0.0, 0.1, 0.2):
        methods = threshold_rows[threshold]["methods"]
        unique_counts = {value["rejection_count"] for value in methods.values()}
        if len(unique_counts) != 1:
            raise RuntimeError("HBN methods do not share a unique rejection count.")
        counts.append(str(unique_counts.pop()))
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrlll}",
        r"\toprule",
        (r"$N_W/N_C^{\rm raw}/N_T$ & $d$ & $N_C$ & $\tau$ & "
         r"$\min\widehat\Gamma$ & $C$ & Branch & Rejections ($z_0=0,.1,.2$) \\"),
        r"\midrule",
        (f"{split['D_W']}/{split['D_C']}/{split['D_T']} & {hbn['d']} & {hbn['N_C']} & "
         f"{hbn['tau']:.4f} & {hbn['minimum_certified_cross_moment']:.4f} & "
         f"{int(bool(hbn['certificate_accepted']))} & {_latex_escape(hbn['selected_branch'])} & "
         f"{'/'.join(counts)} " + r"\\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        "",
        r"\begin{minipage}{\textwidth}\small\raggedright",
        (r"\par\smallskip\noindent\textit{Primary-weight diagnostic:} "
         + rf"at $\eta={float(hbn['primary_eta']):g}$ and $z_0={float(hbn['primary_raw_correlation_threshold']):g}$, "
         + rf"$W_{{\min}}={float(primary['weight_minimum']):.3f}$ and "
         + rf"$W_{{\max}}={float(primary['weight_maximum']):.3f}$. "
         + "Every reported method rejected all 10 hypotheses; this application therefore illustrates "
           "certification and routing, not a learned-weight power gain."),
        "",
        r"\par\smallskip\noindent\textit{Selected ROIs:} "
        + ", ".join(_latex_escape(value) for value in hbn["ROI_list"]) + ".",
        "",
        r"\par\smallskip\noindent\textit{Limitation:} "
        + _latex_escape(hbn["family_information_limitation"]),
        r"\end{minipage}",
        "",
    ]
    return "\n".join(lines).encode("utf-8")
