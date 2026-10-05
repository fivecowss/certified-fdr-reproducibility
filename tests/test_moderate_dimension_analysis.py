from __future__ import annotations

from pathlib import Path

from certified_fdr.moderate_dimension_analysis import ModerateDimensionSummaryBuilder
from certified_fdr.moderate_dimension_rendering import latex_table_bytes, panel_csv_bytes
from certified_fdr.moderate_dimension_stress import METHODS


def record(m: int, margin: float, replication: int) -> dict:
    methods = {
        method: {
            "branch": "aggressive" if method == "Cert_wBY" and margin > 1 else "direct",
            "fdp": 0.01,
            "power": 0.5,
        }
        for method in METHODS
    }
    if method := methods.get("Cert_wBY"):
        method["branch"] = "aggressive" if margin > 1 else "fallback"
    return {
        "module": "MD_STRESS",
        "run_kind": "dry_run",
        "m": m,
        "d": m * (m - 1) // 2,
        "N_C": {20: 5000, 50: 5559, 100: 5967}[m],
        "normalized_margin": margin,
        "s_J": margin,
        "Delta": 1.0,
        "regime": "test",
        "replication": replication,
        "certificate_accepted": margin > 1,
        "oracle_applicable": margin < 0 or margin >= 1,
        "method_results": methods,
    }


def test_summary_contract_and_no_rejection_fraction(tmp_path: Path) -> None:
    builder = ModerateDimensionSummaryBuilder((20, 50, 100), 2)
    for m in (20, 50, 100):
        for margin in (-0.5, 0.5, 1.25):
            builder.update(record(m, margin, 0))
            builder.update(record(m, margin, 1))
    rows = builder.finalize()
    assert len(rows) == 144
    assert sum(row["record_type"] == "certificate" for row in rows) == 9
    assert sum(row["record_type"] == "method_metric" for row in rows) == 126
    assert sum(row["record_type"] == "power_contrast" for row in rows) == 9
    table = latex_table_bytes(rows).decode()
    panel = panel_csv_bytes(rows).decode()
    assert "Rej" not in table
    assert "rejection" not in panel.lower()
    assert len(panel.splitlines()) == 10
