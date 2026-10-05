from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from certified_fdr.certification import completeness_margin


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_core_design_anchors() -> None:
    plan = load("core_experiments.json")
    assert plan["execution"]["master_seed"] == 20260915
    assert plan["module_a"]["full_repetitions_per_configuration"] == 200000
    assert plan["module_b"]["nonnull_indices_zero_based"] == [0, 1, 10, 11]
    assert plan["module_b"]["boundary_loading_index_zero_based"] == 19
    assert plan["module_c"]["N_C"] == 682


def test_routing_design_anchors() -> None:
    plan = load("routing_transition.json")
    assert plan["design"]["varied_loading_index_zero_based"] == 19
    assert plan["design"]["repetitions_per_configuration"] == 200000
    assert plan["rng"]["master_seed"] == 2026091601


def test_moderate_dimension_anchors() -> None:
    plan = load("moderate_dimension.json")
    assert plan["design"]["m"] == [20, 50, 100]
    assert plan["design"]["expected_N_C"] == {"20": 5000, "50": 5559, "100": 5967}
    assert plan["rng"]["master_seed"] == 2026092401
    assert completeness_margin(5000, 190, 0.01) == 0.15525928194403735


def test_hbn_geometry_is_aggregate_and_psd() -> None:
    value = json.loads((ROOT / "data/hbn/frozen_geometry.json").read_text())
    matrix = np.asarray(value["correlation"]["matrix"], dtype=float)
    assert matrix.shape == (10, 10)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.linalg.eigvalsh(matrix).min() > 0.0
    assert value["privacy"]["contains_individual_rows"] is False
    assert value["privacy"]["contains_participant_ids"] is False

