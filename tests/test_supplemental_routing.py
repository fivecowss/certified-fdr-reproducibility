from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from certified_fdr.supplemental_routing import (
    records_sha256,
    simulate_transition_record,
    transition_configurations,
    transition_geometry,
)


ROOT = Path(__file__).resolve().parents[1]


def plan() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/routing_transition.json").read_text()
    )


def test_transition_path_hits_theory_landmarks_and_is_psd() -> None:
    values = transition_configurations(plan())
    assert len(values) == 10
    assert [value["normalized_margin"] for value in values] == [
        -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
    ]
    for value in values:
        geometry = value["geometry"]
        assert np.isclose(
            geometry["actual_s_J"] / geometry["Delta"],
            value["normalized_margin"],
            rtol=0.0,
            atol=5e-14,
        )
        assert geometry["minimum_eigenvalue"] > 0.0
    assert any(value["normalized_margin"] == 0.0 for value in values)
    assert any(value["normalized_margin"] == 1.0 for value in values)


def test_transition_record_is_replication_addressed_and_reproducible() -> None:
    candidate = plan()
    first = simulate_transition_record(candidate, 0.5, 7, "0" * 64)
    second = simulate_transition_record(candidate, 0.5, 7, "0" * 64)
    assert records_sha256([first]) == records_sha256([second])
    assert first["normalized_margin"] == 0.5
    assert np.isclose(first["s_J"] / first["Delta"], 0.5)
    assert first["run_kind"] == "dry_run"
    assert first["noninferential"] is True
    assert first["schema_version"] == "2.0.0"
    assert set(first["method_results"]) == {
        "BY_q", "wBY_q", "wBH_q_diagnostic", "Cert_wBY", "Cert_BY",
        "sign_oracle", "old_both_q_internal",
    }

    changed = dict(first)
    changed["normalized_margin"] = 0.75
    from certified_fdr.ledger import finish_record
    assert finish_record(changed)["record_id"] != first["record_id"]


def test_frozen_execution_metadata() -> None:
    candidate = plan()
    assert candidate["status"] == "frozen"
    assert candidate["design"]["repetitions_per_configuration"] == 200000
