"""Prespecified streaming summaries for the routing transition."""

from __future__ import annotations

from collections import defaultdict
from math import log, sqrt
from typing import Any, Iterable

from .monte_carlo import bounded_mean_interval, exact_binomial_interval


METHODS = (
    "BY_q",
    "wBY_q",
    "wBH_q_diagnostic",
    "Cert_wBY",
    "Cert_BY",
    "sign_oracle",
    "old_both_q_internal",
)
CONTRASTS = (
    ("Cert_wBY-old_both_q_internal", "Cert_wBY", "old_both_q_internal"),
    ("Cert_wBY-wBY_q", "Cert_wBY", "wBY_q"),
    ("Cert_wBY-sign_oracle", "Cert_wBY", "sign_oracle"),
)
BOUNDED_MEAN_FAMILY_SIZE = 70
PAIRED_POWER_FAMILY_SIZE = 26


def _interval(
    estimate: float,
    lower: float,
    upper: float,
    method: str,
    family_size: int,
) -> dict[str, Any]:
    return {
        "estimate": float(estimate),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": 0.95,
        "method": method,
        "family_size": int(family_size),
    }


def paired_power_interval(
    sample_mean: float,
    sample_size: int,
    *,
    confidence: float = 0.95,
    family_size: int = PAIRED_POWER_FAMILY_SIZE,
) -> dict[str, Any]:
    """Hoeffding--Bonferroni interval for a paired difference in [-1,1]."""

    if not -1.0 <= float(sample_mean) <= 1.0:
        raise ValueError("paired power difference must lie in [-1,1]")
    if int(sample_size) < 1 or int(family_size) < 1:
        raise ValueError("sample size and family size must be positive")
    alpha_per_target = (1.0 - float(confidence)) / int(family_size)
    radius = sqrt(2.0 * log(2.0 / alpha_per_target) / int(sample_size))
    return _interval(
        float(sample_mean),
        max(-1.0, float(sample_mean) - radius),
        min(1.0, float(sample_mean) + radius),
        "paired bounded-difference Hoeffding-Bonferroni",
        family_size,
    )


class SupplementalSummaryBuilder:
    def __init__(self, margins: Iterable[float], expected_repetitions: int) -> None:
        self.margins = tuple(float(value) for value in margins)
        if len(self.margins) != 10 or len(set(self.margins)) != 10:
            raise ValueError("supplemental analysis requires ten unique margins")
        self.expected_repetitions = int(expected_repetitions)
        if self.expected_repetitions < 1:
            raise ValueError("expected repetitions must be positive")
        self.counts: dict[float, int] = defaultdict(int)
        self.certificate_counts: dict[float, int] = defaultdict(int)
        self.aggressive_counts: dict[float, int] = defaultdict(int)
        self.regimes: dict[float, str] = {}
        self.oracle_applicable: dict[float, bool] = {}
        self.metric_sums: dict[tuple[float, str, str], float] = defaultdict(float)
        self.contrast_sums: dict[tuple[float, str], float] = defaultdict(float)

    def update(self, record: dict[str, Any]) -> None:
        if record.get("module") != "E" or record.get("run_kind") not in {
            "dry_run", "supplemental"
        }:
            raise ValueError("non-Module-E record passed to supplemental analysis")
        margin = float(record["normalized_margin"])
        if margin not in self.margins:
            raise ValueError(f"record has an unfrozen normalized margin: {margin}")
        if abs(float(record["s_J"]) / float(record["Delta"]) - margin) > 5e-14:
            raise ValueError("record normalized margin does not match s_J/Delta")
        methods = record["method_results"]
        if set(methods) != set(METHODS):
            raise ValueError("record method set differs from the frozen method set")

        regime = str(record["regime"])
        applicable = bool(record["oracle_applicable"])
        if margin in self.regimes and self.regimes[margin] != regime:
            raise ValueError("regime changed within a frozen configuration")
        if margin in self.oracle_applicable and self.oracle_applicable[margin] != applicable:
            raise ValueError("oracle applicability changed within a configuration")
        expected_applicable = margin < 0.0 or margin >= 1.0
        if applicable != expected_applicable:
            raise ValueError("oracle applicability differs from the frozen margin rule")
        self.regimes[margin] = regime
        self.oracle_applicable[margin] = applicable
        self.counts[margin] += 1
        accepted = int(bool(record["certificate_accepted"]))
        self.certificate_counts[margin] += accepted
        aggressive = int(methods["Cert_wBY"]["branch"] == "aggressive")
        if aggressive != accepted:
            raise ValueError("Cert-wBY branch does not equal the certificate decision")
        self.aggressive_counts[margin] += aggressive

        for method in METHODS:
            result = methods[method]
            values = {
                "mean_fdp": float(result["fdp"]),
                "mean_power": float(result["power"]),
                "mean_rejection_fraction": float(result["rejection_count"]) / float(record["m"]),
            }
            if any(not 0.0 <= value <= 1.0 for value in values.values()):
                raise ValueError("bounded method metric is outside [0,1]")
            for metric, value in values.items():
                self.metric_sums[(margin, method, metric)] += value

        for name, left, right in CONTRASTS:
            if right == "sign_oracle" and not applicable:
                continue
            difference = float(methods[left]["power"]) - float(methods[right]["power"])
            self.contrast_sums[(margin, name)] += difference

    def finalize(self) -> list[dict[str, Any]]:
        if any(self.counts[margin] != self.expected_repetitions for margin in self.margins):
            raise RuntimeError("supplemental configuration coverage is incomplete")
        rows: list[dict[str, Any]] = []
        for margin in self.margins:
            n = self.counts[margin]
            accepted = self.certificate_counts[margin]
            if self.aggressive_counts[margin] != accepted:
                raise RuntimeError("aggressive-branch count differs from acceptance count")
            interval = exact_binomial_interval(accepted, n, confidence=0.95)
            rows.append(
                {
                    "schema_version": "1.0.0",
                    "record_type": "certificate",
                    "module": "E",
                    "normalized_margin": margin,
                    "regime": self.regimes[margin],
                    "record_count": n,
                    "certificate_acceptance_count": accepted,
                    "aggressive_branch_count": self.aggressive_counts[margin],
                    "certificate_acceptance": _interval(
                        accepted / n,
                        interval.lower,
                        interval.upper,
                        interval.method + "; pointwise",
                        1,
                    ),
                }
            )
            for method in METHODS:
                metric_intervals: dict[str, dict[str, Any]] = {}
                for metric in ("mean_fdp", "mean_power", "mean_rejection_fraction"):
                    estimate = self.metric_sums[(margin, method, metric)] / n
                    bounded = bounded_mean_interval(
                        estimate,
                        n,
                        confidence=0.95,
                        family_size=BOUNDED_MEAN_FAMILY_SIZE,
                    )
                    metric_intervals[metric] = _interval(
                        estimate,
                        bounded.lower,
                        bounded.upper,
                        bounded.method,
                        BOUNDED_MEAN_FAMILY_SIZE,
                    )
                rows.append(
                    {
                        "schema_version": "1.0.0",
                        "record_type": "method",
                        "module": "E",
                        "normalized_margin": margin,
                        "regime": self.regimes[margin],
                        "record_count": n,
                        "method": method,
                        "oracle_applicable": self.oracle_applicable[margin],
                        **metric_intervals,
                    }
                )
            for name, left, right in CONTRASTS:
                applicable = right != "sign_oracle" or self.oracle_applicable[margin]
                if not applicable:
                    continue
                estimate = self.contrast_sums[(margin, name)] / n
                rows.append(
                    {
                        "schema_version": "1.0.0",
                        "record_type": "power_contrast",
                        "module": "E",
                        "normalized_margin": margin,
                        "regime": self.regimes[margin],
                        "record_count": n,
                        "contrast": name,
                        "left_method": left,
                        "right_method": right,
                        "oracle_applicable": self.oracle_applicable[margin],
                        "mean_power_difference": paired_power_interval(estimate, n),
                    }
                )
        counts = {
            kind: sum(row["record_type"] == kind for row in rows)
            for kind in ("certificate", "method", "power_contrast")
        }
        if counts != {"certificate": 10, "method": 70, "power_contrast": 26}:
            raise RuntimeError(f"unexpected supplemental summary counts: {counts}")
        return rows
