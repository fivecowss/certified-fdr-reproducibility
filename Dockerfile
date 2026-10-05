FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /artifact

COPY environment/requirements-lock.txt /tmp/requirements-lock.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: \
    --requirement /tmp/requirements-lock.txt

COPY . /artifact
RUN python -m pip install --no-cache-dir --no-deps /artifact

RUN useradd --create-home --uid 10001 artifact \
    && chown -R artifact:artifact /artifact
USER artifact

CMD ["python", "scripts/ci_check.py"]
