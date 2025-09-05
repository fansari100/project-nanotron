# syntax=docker/dockerfile:1.7

# ---- builder ----
# Bumped from 1.78 because some transitive deps ship Cargo.toml manifests
# that require Cargo parser features beyond what older toolchains support.
# 1.88 is the floor that resolves cleanly against the current dep graph.
FROM rust:1.88-slim-bookworm AS builder

WORKDIR /build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends pkg-config libssl-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy the whole source up-front. Cargo.toml declares [[bench]] entries
# (benches/serde.rs, benches/ring_buffer.rs) and integration tests, all of
# which Cargo wants to resolve at manifest-parse time, so the
# "stub src/main.rs then cargo fetch" trick was failing the build.
COPY . .

ENV CARGO_TERM_COLOR=never \
    RUSTFLAGS="-C target-cpu=x86-64-v3"

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin nanotron-server && \
    cp target/release/nanotron-server /tmp/nanotron-server

# ---- runtime ----
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --system --no-create-home --uid 10001 nanotron

COPY --from=builder /tmp/nanotron-server /usr/local/bin/nanotron-server

USER nanotron
EXPOSE 8080

ENV NANOTRON_BIND=0.0.0.0:8080 \
    RUST_LOG=info

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/nanotron-server"]
