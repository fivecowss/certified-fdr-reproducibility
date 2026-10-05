# Full synthetic execution

The repository separates three reproducibility levels.

1. `frozen` verifies and renders the immutable sufficient summaries.
2. `smoke` checks configuration, geometry, random-stream, and serial/parallel invariants.
3. `full` regenerates all synthetic replication ledgers before aggregation.

The observed HBN application is not rerun because participant-level inputs are not
distributed. The HBN-derived Gaussian bridge is synthetic and is included as `core-C`.

## Preflight

Run a preflight before creating any result files:

```bash
python scripts/run_full.py \
  --group all \
  --result-root results/full \
  --workers 8 \
  --preflight-only
```

The complete grid contains 27,400 chunks and 29,900,000 replication records. It is
intentionally expensive. The groups `core`, `routing`, and `moderate` may be executed
separately without changing their frozen configurations.

## Execute and resume

```bash
python scripts/run_full.py \
  --group core \
  --result-root results/full \
  --workers 8
```

If a process stops after one or more ledger-manifest pairs have been sealed, inspect the
result directory and resume explicitly:

```bash
python scripts/run_full.py \
  --group core \
  --result-root results/full \
  --workers 8 \
  --resume
```

An existing ledger is reused only when its sidecar metadata and SHA-256 agree. Orphaned or
mismatched files stop execution rather than being overwritten.

## Aggregate and compare

After the corresponding group is complete:

```bash
python scripts/aggregate_full.py \
  --kind core \
  --result-root results/full \
  --output-dir results/full/aggregation/core

python scripts/compare_frozen.py \
  --kind core \
  --rerun-summary results/full/aggregation/core/core_experiments_summary.jsonl.gz
```

Replace `core` with `routing` or `moderate` for the other result families. Comparison is
performed on every decoded summary row and field. Compressed SHA-256 values are also
reported, but semantic equality is the required gate.

Execution logs and manifests do not print scientific estimates. Monte Carlo intervals in
the summaries are not theoretical guarantees.
