#!/bin/sh
# Entrypoint for switchboard-local container.
#
# Performs env var substitution on the config template before starting the
# proxy, so compose env vars (e.g. SWITCHBOARD_API_KEY) flow through without
# needing to bake secrets into the mounted TOML.
set -eu

CONFIG_DIR="/home/switchboard/.switchboard"
TEMPLATE="${CONFIG_DIR}/config.toml.tmpl"
OUTPUT="${CONFIG_DIR}/config.toml"

if [ -f "${TEMPLATE}" ]; then
    envsubst < "${TEMPLATE}" > "${OUTPUT}"
fi

exec /usr/local/bin/switchboard-local "$@"
