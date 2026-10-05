"""Prespecified panel and table extraction from frozen summaries."""

from __future__ import annotations

from math import isclose
from typing import Any

from .monte_carlo import bounded_mean_interval


PANEL_B_METHODS = (
    "BY_q",
    "wBY_q",
    "wBH_q_diagnostic",
    "Cert_wBY",
    "Cert_BY",
    "sign_oracle",
)
PANEL_C_METHODS = ("Cert_wBY", "wBY_q", "wBH_q_diagnostic")


def _appendix_row(
    *,
    appendix: str,
    row: dict[str, Any],
    metric: str,
    estimate: float,
    lower: float | None,
    upper: float | None,
    interval_method: str,
) -> dict[str, Any]:
    return {
        "appendix": appendix,
        "module": row["module"],
        "configuration_id": row["configuration_id"],
        "regime": row["regime"],
        "d": row["d"],
        "N_C": row["N_C"],
        "gamma": row["gamma"],
        "N_C_over_N_upper": row["N_C_over_N_upper"],
        "eta": row["eta"],
        "noncentrality": row["noncentrality"],
        "method": row["method"],
        "metric": metric,
        "estimate": float(estimate),
        "lower": lower,
        "upper": upper,
        "interval_method": interval_method,
        "diagnostic": (
            row["method"] == "wBH_q_diagnostic"
            or metric == "mean_oracle_disagreement_fraction"
        ),
        "oracle_applicable": row["oracle_applicable"],
    }


def extract_appendix_data(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract every prespecified appendix cell without post-result selection."""

    output: list[dict[str, Any]] = []
    module_a = [row for row in summaries if row["module"] == "A"]
    if len(module_a) != 108:
        raise ValueError(f"Module A appendix requires 108 cells; found {len(module_a)}.")
    for row in module_a:
        output.append(
            _appendix_row(
                appendix="module_A_full_grid",
                row=row,
                metric="certificate_acceptance_probability",
                estimate=row["certificate_acceptance_probability"],
                lower=row["certificate_interval_lower"],
                upper=row["certificate_interval_upper"],
                interval_method=row["certificate_interval_method"],
            )
        )

    module_b = [
        row
        for row in summaries
        if row["module"] == "B" and row["noncentrality"] in (1.5, 2.5)
    ]
    if len(module_b) != 126:
        raise ValueError(
            f"Module B sensitivity appendix requires 126 method cells; found {len(module_b)}."
        )
    for row in module_b:
        for metric in ("mean_fdp", "mean_power"):
            panel = _bounded_panel_row(
                panel="appendix_module_B_sensitivity",
                row=row,
                metric=metric,
                family_size=126,
                x=float(row["eta"]),
                facet=row["regime"],
            )
            output.append(
                _appendix_row(
                    appendix="module_B_sensitivity",
                    row=row,
                    metric=metric,
                    estimate=panel["estimate"],
                    lower=panel["lower"],
                    upper=panel["upper"],
                    interval_method=panel["interval_method"],
                )
            )

    module_c = [row for row in summaries if row["module"] == "C"]
    if len(module_c) != 63:
        raise ValueError(f"Module C appendix requires 63 method cells; found {len(module_c)}.")
    for row in module_c:
        for metric in ("mean_fdp", "mean_power"):
            panel = _bounded_panel_row(
                panel="appendix_module_C",
                row=row,
                metric=metric,
                family_size=63,
                x=float(row["eta"]),
                facet=row["regime"],
            )
            output.append(
                _appendix_row(
                    appendix="module_C_full_grid",
                    row=row,
                    metric=metric,
                    estimate=panel["estimate"],
                    lower=panel["lower"],
                    upper=panel["upper"],
                    interval_method=panel["interval_method"],
                )
            )

    oracle = [
        row
        for row in summaries
        if row["module"] in ("B", "C")
        and row["oracle_applicable"] is True
        and row["mean_oracle_disagreement_fraction"] is not None
    ]
    for row in oracle:
        output.append(
            _appendix_row(
                appendix="oracle_disagreement_applicable_regions_only",
                row=row,
                metric="mean_oracle_disagreement_fraction",
                estimate=row["mean_oracle_disagreement_fraction"],
                lower=None,
                upper=None,
                interval_method="none; descriptive diagnostic",
            )
        )

    if not any(row["method"] == "old_both_q_internal" for row in output):
        raise ValueError("old both-at-q-internal router ablation is missing.")
    return sorted(
        output,
        key=lambda row: (
            row["appendix"],
            row["module"],
            row["configuration_id"],
            row["method"],
            row["metric"],
        ),
    )


def _bounded_panel_row(
    *,
    panel: str,
    row: dict[str, Any],
    metric: str,
    family_size: int,
    x: str | float,
    facet: str | None,
) -> dict[str, Any]:
    estimate = float(row[metric])
    interval = bounded_mean_interval(
        estimate,
        int(row["record_count"]),
        confidence=0.95,
        family_size=family_size,
    )
    return {
        "panel": panel,
        "x": x,
        "facet": facet,
        "series": row["method"],
        "metric": metric,
        "estimate": estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "interval_method": f"Hoeffding-Bonferroni; family_size={family_size}",
        "diagnostic": row["method"] == "wBH_q_diagnostic",
        "claim_label": "simulation",
    }


def extract_main_panel_data(
    summaries: list[dict[str, Any]],
    hbn_application: dict[str, Any],
) -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []

    panel_a = [
        row
        for row in summaries
        if row["module"] == "A"
        and row["d"] == 20
        and row["gamma"] == 0.1
        and row["method"] == "certificate_only"
    ]
    if len(panel_a) != 12:
        raise ValueError(f"Panel A requires exactly 12 cells; found {len(panel_a)}.")
    for row in panel_a:
        panels.append(
            {
                "panel": "A",
                "x": row["N_C_over_N_upper"],
                "facet": None,
                "series": row["regime"],
                "metric": "certificate_acceptance_probability",
                "estimate": row["certificate_acceptance_probability"],
                "lower": row["certificate_interval_lower"],
                "upper": row["certificate_interval_upper"],
                "interval_method": row["certificate_interval_method"],
                "diagnostic": False,
                "claim_label": "simulation",
            }
        )

    panel_b = [
        row
        for row in summaries
        if row["module"] == "B"
        and row["eta"] == 1.0
        and row["noncentrality"] == 2.0
        and row["method"] in PANEL_B_METHODS
    ]
    if len(panel_b) != 18:
        raise ValueError(f"Panel B requires exactly 18 cells; found {len(panel_b)}.")
    for row in panel_b:
        panels.append(
            _bounded_panel_row(
                panel="B",
                row=row,
                metric="mean_fdp",
                family_size=18,
                x=row["regime"],
                facet=None,
            )
        )

    panel_c = [
        row
        for row in summaries
        if row["module"] == "B"
        and row["noncentrality"] == 2.0
        and row["method"] in PANEL_C_METHODS
    ]
    if len(panel_c) != 27:
        raise ValueError(f"Panel C requires exactly 27 cells; found {len(panel_c)}.")
    for row in panel_c:
        panels.append(
            _bounded_panel_row(
                panel="C",
                row=row,
                metric="mean_power",
                family_size=27,
                x=float(row["eta"]),
                facet=row["regime"],
            )
        )

    candidates = [
        row
        for row in summaries
        if row["module"] == "C"
        and row["eta"] == 1.0
        and row["noncentrality"] == 2.0
        and row["method"] == "Cert_wBY"
    ]
    if len(candidates) != 1:
        raise ValueError("Panel D requires one primary Module C summary cell.")
    bridge = candidates[0]
    if bridge["oracle_applicable"] is not False:
        raise ValueError("Panel D HBN geometry must remain in the no-oracle region.")
    if hbn_application.get("claim_label") != "model-based exploratory application":
        raise ValueError("Panel D requires the frozen exploratory HBN claim label.")
    if hbn_application.get("contains_participant_ids") is not False:
        raise ValueError("HBN application output contains participant identifiers.")
    if hbn_application.get("contains_individual_rows") is not False:
        raise ValueError("HBN application output contains individual rows.")
    if not isclose(
        float(hbn_application["tau"]),
        float(bridge["tau"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Module C and HBN application certificate thresholds differ.")
    quantities = (
        ("tau", bridge["tau"], "theory"),
        ("Delta", bridge["Delta"], "theory"),
        ("population_s_J", bridge["s_J"], "frozen HBN Gaussian bridge"),
        (
            "observed_minimum_cross_moment",
            hbn_application["minimum_certified_cross_moment"],
            "model-based exploratory application",
        ),
    )
    for name, value, claim in quantities:
        panels.append(
            {
                "panel": "D",
                "x": name,
                "facet": None,
                "series": name,
                "metric": "hbn_calibrated_certificate_point",
                "estimate": float(value),
                "lower": None,
                "upper": None,
                "interval_method": "none; not Monte Carlo uncertainty",
                "diagnostic": False,
                "claim_label": claim,
            }
        )

    return sorted(
        panels,
        key=lambda row: (
            row["panel"],
            str(row["facet"]),
            str(row["series"]),
            str(row["x"]),
        ),
    )


def method_comparison_rows() -> list[dict[str, str]]:
    return [
        {
            "method": "wBH",
            "learned_weights": "yes",
            "dependence_input": "PRDS assumed",
            "arbitrary_dependence_guarantee": "no",
            "uses_certificate": "no",
        },
        {
            "method": "wBY",
            "learned_weights": "yes",
            "dependence_input": "none",
            "arbitrary_dependence_guarantee": "yes",
            "uses_certificate": "no",
        },
        {
            "method": "conditional calibration",
            "learned_weights": "optional",
            "dependence_input": "covariance/joint law",
            "arbitrary_dependence_guarantee": "method-specific",
            "uses_certificate": "no",
        },
        {
            "method": "proposed",
            "learned_weights": "yes",
            "dependence_input": "held-out sign certificate",
            "arbitrary_dependence_guarantee": "yes through routing",
            "uses_certificate": "yes",
        },
    ]


def hbn_application_table_row(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("contains_participant_ids") is not False:
        raise ValueError("HBN table cannot contain participant identifiers.")
    if record.get("contains_individual_rows") is not False:
        raise ValueError("HBN table cannot contain individual rows.")
    counts: dict[str, int] = {}
    for result in record["threshold_results"]:
        if result["eta"] == record["primary_eta"]:
            key = f"raw_z0={result['raw_correlation_threshold']:g}"
            counts[key] = result["methods"]["Cert_wBY"]["rejection_count"]
    if set(counts) != {"raw_z0=0", "raw_z0=0.1", "raw_z0=0.2"}:
        raise ValueError("HBN threshold grid is incomplete.")
    return {
        "claim_label": record["claim_label"],
        "split_sizes": record["split_sizes"],
        "d": record["d"],
        "N_C": record["N_C"],
        "tau": record["tau"],
        "minimum_cross_moment": record["minimum_certified_cross_moment"],
        "C": int(record["certificate_accepted"]),
        "selected_branch": record["selected_branch"],
        "rejection_counts": counts,
        "ROI_list": record["ROI_list"],
        "family_information_limitation": record["family_information_limitation"],
    }
