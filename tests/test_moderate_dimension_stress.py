from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from certified_fdr.certification import completeness_margin
from certified_fdr.covariance import global_pairs, selected_correlations
from certified_fdr.moderate_dimension_stress import (
    EXPECTED_N_C,
    geometry_for_configuration,
    stress_configurations,
)
from certified_fdr.supplemental_routing import transition_geometry


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/moderate_dimension.json"


def load_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_grid_and_n_certificate_anchors() -> None:
    plan = load_plan()
    configurations = stress_configurations(plan)
    assert len(configurations) == 9
    delta_0 = completeness_margin(5000, 190, 0.01)
    assert delta_0 == 0.15525928194403735
    for configuration in configurations:
        m = configuration["m"]
        d = configuration["d"]
        n = configuration["N_C"]
        assert d == m * (m - 1) // 2
        assert n == EXPECTED_N_C[m]
        assert completeness_margin(n, d, 0.01) <= delta_0
        assert completeness_margin(n - 1, d, 0.01) > delta_0


def test_exact_margin_and_m20_routing_equivalence() -> None:
    plan = load_plan()
    baseline_plan = {
        "design": {
            "m": 20,
            "d": 190,
            "N_C": 5000,
            "beta": 0.01,
            "base_loading": 0.5,
            "varied_loading_index_zero_based": 19,
        }
    }
    for configuration in stress_configurations(plan):
        geometry = geometry_for_configuration(plan, configuration)
        correlation = np.asarray(geometry["correlation"])
        actual = np.min(selected_correlations(correlation, global_pairs(configuration["m"])))
        assert abs(actual - configuration["normalized_margin"] * geometry["Delta"]) < 5e-15
        if configuration["m"] == 20:
            original = transition_geometry(
                baseline_plan, configuration["normalized_margin"]
            )["correlation"]
            assert np.array_equal(correlation, original)


def test_dimension_scaled_nonnull_pattern() -> None:
    expected = {
        20: [0, 1, 10, 11],
        50: [*range(5), *range(25, 30)],
        100: [*range(10), *range(50, 60)],
    }
    for configuration in stress_configurations(load_plan()):
        assert configuration["nonnull_indices_zero_based"] == expected[configuration["m"]]
