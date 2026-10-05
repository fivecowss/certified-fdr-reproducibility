# Environment and artifact packaging

## Exact environment

The reference environment uses CPython 3.11.15. The complete installed dependency closure
is pinned in `environment/requirements-lock.txt` and checked against
`environment/environment.json`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/ci_check.py
```

The container runs the same bounded checks. It does not execute the 29.9-million-record
full simulation.

```bash
docker build -t certified-fdr-reproducibility .
docker run --rm certified-fdr-reproducibility
```

## Anonymous archive

Build the archive only from a clean release commit:

```bash
python scripts/build_review_archive.py \
  --require-clean \
  --output ../certified-fdr-reproducibility.zip
```

The builder uses an allowlisted inventory, omits Git metadata and generated result roots,
checks identity-bearing strings, and refuses unexpected files under `data/`. The archive
contains frozen sufficient summaries and aggregate HBN geometry, but no participant-level
features or identifiers.

Some immutable manifests preserve historical execution labels inside byte-exact scientific
provenance. Those files are excluded from the public-surface terminology check, but they
remain subject to the identity and absolute-path audit.
