from __future__ import annotations

import numpy as np
import pytest

from certified_fdr.metrics import false_discovery_proportion, power, rejection_count, symmetric_difference_count
from certified_fdr.monte_carlo import bounded_mean_interval, exact_binomial_interval


def test_replication_metrics() -> None:
    rejected = np.array([True, True, False, False])
    nonnull = np.array([False, True, True, False])
    assert rejection_count(rejected) == 2
    assert false_discovery_proportion(rejected, nonnull) == pytest.approx(0.5)
    assert power(rejected, nonnull) == pytest.approx(0.5)
    assert symmetric_difference_count(rejected, nonnull) == 2


def test_no_rejections_has_zero_fdp() -> None:
    assert false_discovery_proportion([False, False], [False, True]) == 0.0


def test_clopper_pearson_endpoints_and_coverage_label() -> None:
    none = exact_binomial_interval(0, 100)
    all_success = exact_binomial_interval(100, 100)
    assert none.lower == 0.0 and none.upper < 0.05
    assert all_success.upper == 1.0 and all_success.lower > 0.95
    assert none.method == "Clopper-Pearson exact"


def test_bounded_mean_interval_is_clipped_and_family_adjusted() -> None:
    single = bounded_mean_interval(0.5, 1000, family_size=1)
    family = bounded_mean_interval(0.5, 1000, family_size=10)
    edge = bounded_mean_interval(0.0, 10)
    assert family.upper - family.lower > single.upper - single.lower
    assert edge.lower == 0.0
