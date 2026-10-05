# Certified FDR Reproducibility

This repository contains code, frozen aggregate outputs, and rendering tools
for experiments on dependence-certified learned weighted false discovery rate
control.

The repository separates three tasks:

1. **Frozen reproduction** verifies immutable aggregate summaries and regenerates
   the manuscript figures without changing scientific values.
2. **Smoke validation** runs a small deterministic subset to check configuration,
   covariance, random-stream, and serial/parallel reproducibility gates.
3. **Full execution** regenerates replication ledgers and aggregate summaries.
   It is computationally expensive and is documented separately from the fast
   artifact checks.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/reproduce.py --mode frozen --output-dir outputs/frozen
python scripts/reproduce.py --mode smoke --output-dir outputs/smoke
python -m pytest
```

Frozen reproduction reads only files under `data/frozen/`. It does not rerun a
simulation or recompute an interval. Output directories must be absent or empty.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/` | Immutable scientific and rendering configurations |
| `data/frozen/` | Canonical compressed aggregate summaries and manifests |
| `data/hbn/` | Aggregate HBN-derived geometry; no participant-level rows |
| `src/certified_fdr/` | Simulation, testing, aggregation, and rendering kernels |
| `scripts/` | Portable verification and reproduction entry points |
| `tests/` | Unit, configuration, privacy, and provenance checks |
| `provenance/` | Source-snapshot and frozen-input hashes |

The observed HBN output is an exploratory, noncomparative application. The
included HBN-derived correlation geometry contains no participant identifiers
or participant-level feature rows.

## Platform policy

Python entry points are the primary interface. Shell-specific utilities,
absolute user paths, and local package-manager assumptions are not required.
Exact PDF bytes can vary across Matplotlib and font backends; scientific CSV
values, table values, input hashes, row counts, and panel structure are the
portable verification targets.

