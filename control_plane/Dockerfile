# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml ./
COPY src ./src

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip wheel setuptools && \
    /opt/venv/bin/pip install .

# ---- runtime ----
FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends tini ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --system --no-create-home --uid 10001 nanotron

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NANOTRON_CP_BIND_HOST=0.0.0.0 \
    NANOTRON_CP_BIND_PORT=8090

USER nanotron
EXPOSE 8090

ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "nanotron_control"]
