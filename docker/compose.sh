#!/usr/bin/env bash
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${DOCKER_DIR}/generate_compose_env.py"
exec docker compose -f "${DOCKER_DIR}/docker-compose.yml" --env-file "${DOCKER_DIR}/.env.compose" "$@"
