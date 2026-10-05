"""Deterministic streaming summaries for frozen replication ledgers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .monte_carlo import exact_binomial_interval


@dataclass
class CompensatedSum:
    total: float = 0.0
    correction: float = 0.0

    def add(self, value: float) -> None:
        adjusted = float(value) - self.correction
        updated = self.total + adjusted
        self.correction = (updated - self.total) - adjusted
        self.total = updated


@dataclass
class Cell:
    metadata: dict[str, Any]
    method: str
    record_count: int = 0
    certificate_acceptance_count: int = 0
    fdp: CompensatedSum = field(default_factory=CompensatedSum)
    power: CompensatedSum = field(default_factory=CompensatedSum)
    rejection_fraction: CompensatedSum = field(default_factory=CompensatedSum)
    oracle_disagreement_fraction: CompensatedSum = field(default_factory=CompensatedSum)
    oracle_disagreement_count: int = 0


def parse_module_a_configuration(configuration_id: str) -> tuple[float, float]:
    fields: dict[str, str] = {}
    for component in str(configuration_id).split("|")[1:]:
        key, separator, value = component.partition("=")
        if not separator or key in fields:
            raise ValueError("invalid Module A configuration identifier.")
        fields[key] = value
    if set(fields) != {"d", "gamma", "ratio", "law"}:
        raise ValueError("Module A configuration identifier has unexpected fields.")
    return float(fields["gamma"]), float(fields["ratio"])


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    gamma: float | None = None
    ratio: float | None = None
    if record["module"] == "A":
        gamma, ratio = parse_module_a_configuration(record["configuration_id"])
    return {
        "module": record["module"],
        "configuration_id": record["configuration_id"],
        "regime": record["regime"],
        "m": record["m"],
        "d": record["d"],
        "N_C": record["N_C"],
        "q": record["q"],
        "beta": record["beta"],
        "eta": record["eta"],
        "noncentrality": record["noncentrality"],
        "gamma": gamma,
        "N_C_over_N_upper": ratio,
        "s_J": record["s_J"],
        "Delta": record["Delta"],
        "tau": record["tau"],
        "oracle_region": record["oracle_region"],
        "oracle_applicable": record["oracle_applicable"],
    }


class SummaryBuilder:
    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = float(confidence)
        self.cells: dict[tuple[str, str, str], Cell] = {}

    def update(self, record: dict[str, Any]) -> None:
        if record.get("run_kind") != "primary" or record.get("noninferential") is not False:
            raise ValueError("summary accepts only inferential primary records.")
        metadata = _metadata(record)
        methods = record["method_results"]
        for method, result in methods.items():
            key = (record["module"], record["configuration_id"], method)
            cell = self.cells.setdefault(key, Cell(metadata=dict(metadata), method=method))
            if cell.metadata != metadata:
                raise ValueError("metadata changed within a summary cell.")
            cell.record_count += 1
            cell.certificate_acceptance_count += int(record["certificate_accepted"])
            if result["fdp"] is not None:
                cell.fdp.add(result["fdp"])
            if result["power"] is not None:
                cell.power.add(result["power"])
            if result["rejection_count"] is not None:
                cell.rejection_fraction.add(result["rejection_count"] / record["m"])
            if result["oracle_disagreement_count"] is not None:
                cell.oracle_disagreement_fraction.add(
                    result["oracle_disagreement_count"] / record["m"]
                )
                cell.oracle_disagreement_count += 1

    def finalize(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(self.cells):
            cell = self.cells[key]
            n = cell.record_count
            interval = exact_binomial_interval(
                cell.certificate_acceptance_count, n, self.confidence
            )
            has_testing_metrics = cell.method != "certificate_only"
            rows.append(
                {
                    "schema_version": "1.0.0",
                    **cell.metadata,
                    "method": cell.method,
                    "record_count": n,
                    "certificate_acceptance_count": cell.certificate_acceptance_count,
                    "certificate_acceptance_probability": cell.certificate_acceptance_count / n,
                    "certificate_interval_lower": interval.lower,
                    "certificate_interval_upper": interval.upper,
                    "certificate_interval_method": (
                        "Clopper-Pearson exact equal-tailed; marginal cell coverage"
                    ),
                    "mean_fdp": cell.fdp.total / n if has_testing_metrics else None,
                    "mean_power": cell.power.total / n if has_testing_metrics else None,
                    "mean_rejection_fraction": (
                        cell.rejection_fraction.total / n if has_testing_metrics else None
                    ),
                    "mean_oracle_disagreement_fraction": (
                        cell.oracle_disagreement_fraction.total
                        / cell.oracle_disagreement_count
                        if cell.oracle_disagreement_count
                        else None
                    ),
                    "oracle_disagreement_record_count": cell.oracle_disagreement_count,
                }
            )
        return rows


def summarize_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    builder = SummaryBuilder()
    for record in records:
        builder.update(record)
    return builder.finalize()
