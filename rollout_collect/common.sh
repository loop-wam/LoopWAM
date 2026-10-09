#!/usr/bin/env bash
# Shared checks for the rollout collection scripts.
# Source this file; do not execute it.

require_env() {
  local name="$1"
  local kind="${2:-nonempty}"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "${name} is required." >&2
    exit 1
  fi
  if [[ "$kind" == "dir" && ! -d "$value" ]]; then
    echo "${name} is not a directory: ${value}" >&2
    exit 1
  fi
}
