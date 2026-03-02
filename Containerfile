# ── Stage 1: Build ────────────────────────────────────────────────────────────
#
# Builds both switchboard-server and switchboard-local release binaries inside
# the official Rust image so the host doesn't need a Rust toolchain.
FROM rust:1.85-bookworm AS builder

WORKDIR /build

# Install protobuf compiler (required by tonic-build for guardrail protos).
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy the full workspace so cargo can resolve the workspace Cargo.toml.
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY proto/ ./proto/
COPY admin-ui/ ./admin-ui/

RUN cargo build --release \
    -p switchboard-server \
    -p switchboard-local \
    && strip target/release/switchboard-server \
    && strip target/release/switchboard-local

# ── Stage 2: switchboard-server runtime ───────────────────────────────────────
FROM debian:bookworm-slim AS server

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/switchboard-server /usr/local/bin/switchboard-server

# Config is mounted at runtime via a volume / bind mount.
ENV SWITCHBOARD_CONFIG=/etc/switchboard/server.toml

EXPOSE 8080 9090

ENTRYPOINT ["/usr/local/bin/switchboard-server"]

# ── Stage 3: switchboard-local runtime ────────────────────────────────────────
FROM debian:bookworm-slim AS local

# gettext-base provides envsubst for config template substitution.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/switchboard-local /usr/local/bin/switchboard-local
COPY deploy/entrypoint-local.sh /usr/local/bin/entrypoint-local.sh
RUN chmod +x /usr/local/bin/entrypoint-local.sh

# Config is read from $HOME/.switchboard/config.toml.
# Override HOME so the mount point is predictable.
ENV HOME=/home/switchboard
RUN mkdir -p /home/switchboard/.switchboard

EXPOSE 8877

ENTRYPOINT ["/usr/local/bin/entrypoint-local.sh"]
CMD ["start"]
