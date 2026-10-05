"""Prespecified summaries for the moderate-dimensional stress test."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log, sqrt
from typing import Any, Iterable

from .moderate_dimension_stress import METHODS, NORMALIZED_MARGINS
from .monte_carlo import bounded_mean_interval, exact_binomial_interval


METHOD_FAMILY_SIZE = 63
CONTRAST_FAMILY_SIZE = 9
METRICS = ("mean_fdp", "mean_power")


@dataclass
class CompensatedSum:
    total: float = 0.0
    correction: float = 0.0

    def add(self, value: float) -> None:
        adjusted = float(value) - self.correction
        updated = self.total + adjusted
        self.correction = (updated - self.total) - adjusted
        self.total = updated


def interval_dict(
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
        "method": str(method),
        "family_size": int(family_size),
    }


def paired_interval(estimate: float, n: int) -> dict[str, Any]:
    if not -1.0 <= float(estimate) <= 1.0:
        raise ValueError("paired power difference is outside [-1,1]")
    alpha = 0.05 / CONTRAST_FAMILY_SIZE
    radius = sqrt(2.0 * log(2.0 / alpha) / int(n))
    return interval_dict(
        estimate,
        max(-1.0, estimate - radius),
        min(1.0, estimate + radius),
        "paired bounded-difference Hoeffding-Bonferroni",
        CONTRAST_FAMILY_SIZE,
    )


class ModerateDimensionSummaryBuilder:
    def __init__(
        self,
        dimensions: Iterable[int],
        expected_repetitions: int,
    ) -> None:
        self.dimensions = tuple(map(int, dimensions))
        self.cells = tuple(
            (m, margin) for m in self.dimensions for margin in NORMALIZED_MARGINS
        )
        if self.dimensions != (20, 50, 100):
            raise ValueError("dimension grid differs from the frozen analysis grid")
        self.expected_repetitions = int(expected_repetitions)
        self.counts: dict[tuple[int, float], int] = defaultdict(int)
        self.accepted: dict[tuple[int, float], int] = defaultdict(int)
        self.d: dict[tuple[int, float], int] = {}
        self.n_certificate: dict[tuple[int, float], int] = {}
        self.regime: dict[tuple[int, float], str] = {}
        self.oracle_applicable: dict[tuple[int, float], bool] = {}
        self.metric_sums: dict[tuple[int, float, str, str], CompensatedSum] = defaultdict(
            CompensatedSum
        )
        self.contrast_sums: dict[tuple[int, float], CompensatedSum] = defaultdict(
            CompensatedSum
        )

    def update(self, record: dict[str, Any]) -> None:
        if record.get("module") != "MD_STRESS" or record.get("run_kind") not in {
            "dry_run",
            "moderate_dimension_stress",
        }:
            raise ValueError("record does not belong to the moderate-dimensional stress test")
        m = int(record["m"])
        margin = float(record["normalized_margin"])
        cell = (m, margin)
        if cell not in self.cells:
            raise ValueError("record is outside the frozen analysis grid")
        if abs(float(record["s_J"]) / float(record["Delta"]) - margin) > 5e-14:
            raise ValueError("record normalized margin does not equal s_J/Delta")
        methods = record["method_results"]
        if set(methods) != set(METHODS):
            raise ValueError("record method set differs from the seven frozen methods")
        expected_applicable = margin < 0.0 or margin >= 1.0
        if bool(record["oracle_applicable"]) != expected_applicable:
            raise ValueError("oracle applicability differs from the frozen comparison domain")
        if (methods["Cert_wBY"]["branch"] == "aggressive") != bool(
            record["certificate_accepted"]
        ):
            raise ValueError("Cert-wBY branch differs from the certificate decision")

        self.counts[cell] += 1
        self.accepted[cell] += int(bool(record["certificate_accepted"]))
        self.d[cell] = int(record["d"])
        self.n_certificate[cell] = int(record["N_C"])
        self.regime[cell] = str(record["regime"])
        self.oracle_applicable[cell] = bool(record["oracle_applicable"])
        for method in METHODS:
            values = {
                "mean_fdp": float(methods[method]["fdp"]),
                "mean_power": float(methods[method]["power"]),
            }
            if any(not 0.0 <= value <= 1.0 for value in values.values()):
                raise ValueError("method metric is outside [0,1]")
            for metric, value in values.items():
                self.metric_sums[(m, margin, method, metric)].add(value)
        self.contrast_sums[cell].add(
            float(methods["Cert_wBY"]["power"])
            - float(methods["wBY_q"]["power"])
        )

    def finalize(self) -> list[dict[str, Any]]:
        if any(
            self.counts[cell] != self.expected_repetitions for cell in self.cells
        ):
            raise RuntimeError("stress-test configuration coverage is incomplete")
        rows: list[dict[str, Any]] = []
        for m, margin in self.cells:
            cell = (m, margin)
            n = self.counts[cell]
            accepted = self.accepted[cell]
            cp = exact_binomial_interval(accepted, n, 0.95)
            common = {
                "schema_version": "1.0.0",
                "module": "MD_STRESS",
                "m": m,
                "d": self.d[cell],
                "N_C": self.n_certificate[cell],
                "normalized_margin": margin,
                "regime": self.regime[cell],
                "record_count": n,
            }
            rows.append(
                {
                    **common,
                    "record_type": "certificate",
                    "certificate_acceptance_count": accepted,
                    "certificate_acceptance": interval_dict(
                        accepted / n,
                        cp.lower,
                        cp.upper,
                        cp.method + "; pointwise",
                        1,
                    ),
                }
            )
            for method in METHODS:
                for metric in METRICS:
                    estimate = self.metric_sums[(m, margin, method, metric)].total / n
                    hb = bounded_mean_interval(
                        estimate, n, confidence=0.95, family_size=METHOD_FAMILY_SIZE
                    )
                    rows.append(
                        {
                            **common,
                            "record_type": "method_metric",
                            "method": method,
                            "metric": metric,
                            "oracle_applicable": self.oracle_applicable[cell],
                            "value": interval_dict(
                                estimate,
                                hb.lower,
                                hb.upper,
                                hb.method,
                                METHOD_FAMILY_SIZE,
                            ),
                        }
                    )
            estimate = self.contrast_sums[cell].total / n
            rows.append(
                {
                    **common,
                    "record_type": "power_contrast",
                    "contrast": "Cert_wBY-wBY_q",
                    "left_method": "Cert_wBY",
                    "right_method": "wBY_q",
                    "oracle_applicable": self.oracle_applicable[cell],
                    "value": paired_interval(estimate, n),
                }
            )
        counts = {
            name: sum(row["record_type"] == name for row in rows)
            for name in ("certificate", "method_metric", "power_contrast")
        }
        if counts != {
            "certificate": 9,
            "method_metric": 126,
            "power_contrast": 9,
        }:
            raise RuntimeError(f"unexpected summary counts: {counts}")
        return rows
