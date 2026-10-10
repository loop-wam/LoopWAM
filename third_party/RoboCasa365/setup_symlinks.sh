#!/usr/bin/env bash
# Link large RoboCasa / robosuite assets into this tree.
#
#   ASSET_SRC_ROOT=/path/to/robocasa_assets \
#   ROBOSUITE_ASSET_SRC=/path/to/robosuite/models/assets \
#   bash setup_symlinks.sh
#
# Expected under ASSET_SRC_ROOT:
#   fixtures/  objects/  textures/  generative_textures/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${ASSET_SRC_ROOT:-}" ]]; then
  echo "[setup_symlinks] set ASSET_SRC_ROOT to the RoboCasa asset root." >&2
  exit 1
fi
if [[ -z "${ROBOSUITE_ASSET_SRC:-}" ]]; then
  echo "[setup_symlinks] set ROBOSUITE_ASSET_SRC to robosuite/models/assets." >&2
  exit 1
fi

ASSET_DST="${ROOT}/robocasa/robocasa/models/assets"
mkdir -p "${ASSET_DST}"

for name in fixtures objects textures generative_textures; do
  src="${ASSET_SRC_ROOT}/${name}"
  dst="${ASSET_DST}/${name}"
  if [[ ! -d "${src}" ]]; then
    echo "[setup_symlinks] missing: ${src}" >&2
    exit 1
  fi
  rm -rf "${dst}"
  ln -sfn "$(realpath "${src}")" "${dst}"
  echo "[setup_symlinks] ${dst} -> $(realpath "${src}")"
done

if [[ ! -d "${ROBOSUITE_ASSET_SRC}" ]]; then
  echo "[setup_symlinks] missing: ${ROBOSUITE_ASSET_SRC}" >&2
  exit 1
fi
mkdir -p "${ROOT}/robosuite/robosuite/models"
rm -rf "${ROOT}/robosuite/robosuite/models/assets"
ln -sfn "$(realpath "${ROBOSUITE_ASSET_SRC}")" "${ROOT}/robosuite/robosuite/models/assets"
echo "[setup_symlinks] ${ROOT}/robosuite/robosuite/models/assets -> $(realpath "${ROBOSUITE_ASSET_SRC}")"
echo "[setup_symlinks] done"
