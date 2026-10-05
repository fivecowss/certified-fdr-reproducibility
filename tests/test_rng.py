from __future__ import annotations

import numpy as np

from certified_fdr.monte_carlo import rng_for, seed_words


def _draw(replication: int, stream: str = "testing") -> np.ndarray:
    return rng_for(8128, "module-b/config-001", replication, stream).normal(size=8)


def test_same_semantic_key_reproduces_exactly() -> None:
    assert np.array_equal(_draw(7), _draw(7))
    assert seed_words(8128, "module-b/config-001", 7, "testing") == seed_words(8128, "module-b/config-001", 7, "testing")


def test_replication_order_does_not_change_streams() -> None:
    forward = {rep: _draw(rep) for rep in range(12)}
    reverse = {rep: _draw(rep) for rep in reversed(range(12))}
    assert all(np.array_equal(forward[rep], reverse[rep]) for rep in forward)


def test_learning_certification_testing_streams_are_distinct() -> None:
    draws = [_draw(3, name) for name in ("learning", "certification", "testing")]
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[0], draws[2])
    assert not np.array_equal(draws[1], draws[2])
