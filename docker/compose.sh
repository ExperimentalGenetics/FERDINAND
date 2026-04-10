#!/usr/bin/env bash
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USE_TRAEFIK=false

if [[ "${1:-}" == "--traefik" ]]; then
  USE_TRAEFIK=true
  shift
fi

python3 "${DOCKER_DIR}/generate_compose_env.py"

COMPOSE_FILES=(-f "${DOCKER_DIR}/compose.yml")

if [[ "${USE_TRAEFIK}" == "true" ]]; then
  COMPOSE_FILES+=(-f "${DOCKER_DIR}/compose.traefik.yml")
fi

exec docker compose \
  "${COMPOSE_FILES[@]}" \
  --env-file "${DOCKER_DIR}/.env.compose" \
  "$@"
