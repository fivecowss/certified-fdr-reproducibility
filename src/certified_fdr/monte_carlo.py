"""Order-invariant random streams and prespecified Monte Carlo intervals."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import log, sqrt

import numpy as np
from scipy.stats import beta as beta_distribution


STREAM_NAMES = (
    "learning",
    "certification",
    "testing",
    "certification_pairing",
)


def _hash_words(value: str) -> tuple[int, int]:
    digest = sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little"), int.from_bytes(digest[4:8], "little")


def seed_words(master_seed: int, configuration_id: str, replication: int, stream: str) -> tuple[int, ...]:
    if isinstance(master_seed, bool) or int(master_seed) < 0:
        raise ValueError("master_seed must be a nonnegative integer.")
    if isinstance(replication, bool) or int(replication) < 0:
        raise ValueError("replication must be a nonnegative integer.")
    if stream not in STREAM_NAMES:
        raise ValueError(f"unknown stream: {stream}")
    config_words = _hash_words(str(configuration_id))
    stream_words = _hash_words(stream)
    master = int(master_seed)
    replicate = int(replication)
    return (
        master & 0xFFFFFFFF,
        (master >> 32) & 0xFFFFFFFF,
        *config_words,
        replicate & 0xFFFFFFFF,
        (replicate >> 32) & 0xFFFFFFFF,
        *stream_words,
    )


def rng_for(master_seed: int, configuration_id: str, replication: int, stream: str) -> np.random.Generator:
    """Create a stream determined only by its immutable semantic key."""
    return np.random.default_rng(np.random.SeedSequence(seed_words(master_seed, configuration_id, replication, stream)))


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    confidence: float
    method: str


def exact_binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Two-sided equal-tailed Clopper--Pearson interval."""
    if isinstance(trials, bool) or int(trials) < 1:
        raise ValueError("trials must be positive.")
    if isinstance(successes, bool) or not 0 <= int(successes) <= int(trials):
        raise ValueError("successes must lie between zero and trials.")
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    s, n = int(successes), int(trials)
    tail = (1.0 - level) / 2.0
    lower = 0.0 if s == 0 else float(beta_distribution.ppf(tail, s, n - s + 1))
    upper = 1.0 if s == n else float(beta_distribution.ppf(1.0 - tail, s + 1, n - s))
    return Interval(lower, upper, level, "Clopper-Pearson exact")


def bounded_mean_interval(
    sample_mean: float,
    sample_size: int,
    confidence: float = 0.95,
    family_size: int = 1,
) -> Interval:
    """Hoeffding interval for [0,1] outcomes with Bonferroni family coverage."""
    mean = float(sample_mean)
    if not 0.0 <= mean <= 1.0:
        raise ValueError("sample_mean must lie in [0, 1].")
    if isinstance(sample_size, bool) or int(sample_size) < 1:
        raise ValueError("sample_size must be positive.")
    if isinstance(family_size, bool) or int(family_size) < 1:
        raise ValueError("family_size must be positive.")
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    alpha_per_target = (1.0 - level) / int(family_size)
    radius = sqrt(log(2.0 / alpha_per_target) / (2.0 * int(sample_size)))
    return Interval(max(0.0, mean - radius), min(1.0, mean + radius), level, "Hoeffding-Bonferroni")
