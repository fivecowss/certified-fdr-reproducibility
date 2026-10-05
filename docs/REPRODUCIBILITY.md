# Reproducibility modes

## Frozen reproduction

`python scripts/reproduce.py --mode frozen --output-dir outputs/frozen`

This verifies SHA-256 hashes, compressed-summary row counts, structural cell
counts, and HBN privacy flags before rendering figures. It performs no
simulation, aggregation, interval calculation, or parameter fitting.

## Smoke validation

`python scripts/reproduce.py --mode smoke --output-dir outputs/smoke`

This runs one replication per core configuration and two replications per
routing stress configuration. It verifies deterministic semantic random streams,
positive semidefiniteness, exact margin construction, and serial/parallel record
identity. Smoke outputs are noninferential.

## Full execution

Full regeneration retains one record per replication and can require substantial
CPU time and storage. Portable chunked execution entry points are added in a
separate audited layer; they must not overwrite `data/frozen/`.

Monte Carlo confidence intervals describe simulation uncertainty. They are not
theoretical FDR guarantees.

