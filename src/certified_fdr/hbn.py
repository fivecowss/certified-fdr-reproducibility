"""Leakage-resistant deterministic helpers for the exploratory HBN module."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import pandas as pd

from .monte_carlo import rng_for


def select_one_scan_per_subject(
    manifest: pd.DataFrame,
    subject_column: str,
    site_column: str,
    scan_column: str,
) -> pd.DataFrame:
    """Select the lexicographically first scan after validating subject/site."""
    required = {subject_column, site_column, scan_column}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"missing manifest columns: {sorted(missing)}")
    frame = manifest.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("subject, site, and scan identifiers must be nonmissing.")
    site_counts = frame.groupby(subject_column, dropna=False)[site_column].nunique()
    if bool((site_counts > 1).any()):
        raise ValueError("a subject appears at multiple sites.")
    frame[scan_column] = frame[scan_column].astype(str)
    frame = frame.sort_values([subject_column, scan_column], kind="mergesort")
    selected = frame.drop_duplicates(subject_column, keep="first").copy()
    return selected.reset_index(drop=True)


def _stable_rank(seed: int, stratum: str, group: str) -> str:
    return sha256(f"{int(seed)}|{stratum}|{group}".encode("utf-8")).hexdigest()


def assign_site_family_splits(
    one_scan_manifest: pd.DataFrame,
    subject_column: str,
    site_column: str,
    family_column: str | None,
    master_seed: int,
    proportions: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> pd.DataFrame:
    """Assign intact families deterministically within site-composition strata."""
    frame = one_scan_manifest.copy()
    required = {subject_column, site_column}
    if not required.issubset(frame.columns):
        raise ValueError("manifest lacks subject or site column.")
    if frame[subject_column].duplicated().any():
        raise ValueError("one scan per subject is required before splitting.")
    ratios = np.asarray(proportions, dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0.0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("proportions must be three positive values summing to one.")
    split_names = np.array(["D_W", "D_C", "D_T"], dtype=object)

    if family_column is not None and family_column in frame.columns:
        family = frame[family_column].astype("string")
        usable = family.notna() & (family.str.strip() != "")
        frame["_group"] = np.where(usable, "family:" + family.fillna(""), "subject:" + frame[subject_column].astype(str))
    else:
        frame["_group"] = "subject:" + frame[subject_column].astype(str)

    group_sites = frame.groupby("_group")[site_column].agg(lambda x: "|".join(sorted(set(map(str, x)))))
    group_sizes = frame.groupby("_group").size()
    assignment: dict[str, str] = {}
    for stratum in sorted(group_sites.unique()):
        groups = [name for name, value in group_sites.items() if value == stratum]
        groups.sort(key=lambda group: _stable_rank(master_seed, stratum, str(group)))
        target = ratios * float(sum(int(group_sizes[group]) for group in groups))
        allocated = np.zeros(3, dtype=float)
        for group in groups:
            remaining = target - allocated
            chosen = int(np.argmax(remaining))
            assignment[str(group)] = str(split_names[chosen])
            allocated[chosen] += int(group_sizes[group])

    frame["split"] = frame["_group"].map(assignment)
    if frame["split"].isna().any():
        raise RuntimeError("not every subject received a split.")
    return frame.drop(columns="_group").reset_index(drop=True)


def deterministic_disjoint_pairs(
    subject_count: int,
    master_seed: int,
    configuration_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return floor(n/2) disjoint index pairs and at most one unused index."""
    if isinstance(subject_count, bool) or int(subject_count) < 0:
        raise ValueError("subject_count must be nonnegative.")
    permutation = rng_for(master_seed, configuration_id, 0, "certification_pairing").permutation(int(subject_count))
    paired_count = 2 * (int(subject_count) // 2)
    return permutation[:paired_count].reshape(-1, 2), permutation[paired_count:]
