#!/usr/bin/env bash
# Prepare a LoopWAM checkout for local RoboTwin evaluation.
# Safe to re-run: already-finished steps are skipped.
#
#   bash scripts/prepare_eval.sh
#
# Override the prebuilt RoboTwin tree (must contain assets/ and envs/curobo/):
#   ROBOTWIN_PREBUILT=/path/to/RoboTwin bash scripts/prepare_eval.sh
#
# If assets/curobo are already installed under this repo's third_party/RoboTwin,
# you can omit ROBOTWIN_PREBUILT / ASSETS_SRC / CUROBO_SRC.
#
# Override shared pretrained checkpoints (ActionDiT / DiffSynth / Wan):
#   CHECKPOINTS_SRC=/path/to/checkpoints bash scripts/prepare_eval.sh
#
# Links only (no hf download of robotwin ckpt / no download_pretrained):
#   SKIP_DOWNLOAD=1 bash scripts/prepare_eval.sh
#
# Force re-download of robotwin ckpt even if present:
#   FORCE_CKPT_DOWNLOAD=1 bash scripts/prepare_eval.sh
#
# Skip ensuring warp-lang==1.12.0:
#   SKIP_WARP_INSTALL=1 bash scripts/prepare_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

WARP_LANG_VERSION="${WARP_LANG_VERSION:-1.12.0}"

ROBOTWIN_PREBUILT="${ROBOTWIN_PREBUILT:-}"
ASSETS_SRC="${ASSETS_SRC:-${ROBOTWIN_PREBUILT:+${ROBOTWIN_PREBUILT}/assets}}"
CUROBO_SRC="${CUROBO_SRC:-${ROBOTWIN_PREBUILT:+${ROBOTWIN_PREBUILT}/envs/curobo}}"
CHECKPOINTS_SRC="${CHECKPOINTS_SRC:-}"

POLICY_SRC="${ROOT}/experiments/robotwin/fastwam_policy"
POLICY_DST="${ROOT}/third_party/RoboTwin/policy/fastwam_policy"
ASSETS_DST="${ROOT}/third_party/RoboTwin/assets"
CUROBO_DST="${ROOT}/third_party/RoboTwin/envs/curobo"
CKPT_DST="${ROOT}/checkpoints"

SHARED_CKPT_NAMES=(
  ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
  DiffSynth-Studio
  Wan2.2-TI2V-5B
  Wan-AI
)

ensure_warp_lang() {
  if [[ "${SKIP_WARP_INSTALL:-0}" == "1" ]]; then
    echo "[prepare_eval] skip warp-lang (SKIP_WARP_INSTALL=1)"
    return
  fi
  local have
  have="$(python -c "import importlib.metadata as m
try:
    print(m.version('warp-lang'))
except Exception:
    print('')
" 2>/dev/null || true)"
  if [[ "${have}" == "${WARP_LANG_VERSION}" ]]; then
    echo "[prepare_eval] skip warp-lang (already ${WARP_LANG_VERSION})"
    return
  fi
  echo "[prepare_eval] installing warp-lang==${WARP_LANG_VERSION} (have: ${have:-none}) ..."
  local local_whl
  local_whl="$(ls -1 "${ROOT}/third_party/wheels/warp_lang-${WARP_LANG_VERSION}-py3-none-manylinux"*.whl 2>/dev/null | head -1 || true)"
  if [[ -n "${local_whl}" ]]; then
    echo "[prepare_eval] installing from local wheel: ${local_whl}"
    python -m pip install "${local_whl}"
  else
    local indexes=(
      "https://mirrors.aliyun.com/pypi/simple/|mirrors.aliyun.com"
      "https://mirrors.cloud.tencent.com/pypi/simple/|mirrors.cloud.tencent.com"
      "https://mirrors.huaweicloud.com/repository/pypi/simple/|mirrors.huaweicloud.com"
      "https://pypi.org/simple|"
    )
    local ok=0 entry url host
    for entry in "${indexes[@]}"; do
      url="${entry%%|*}"
      host="${entry#*|}"
      echo "[prepare_eval] trying ${url}"
      if [[ -n "${host}" ]]; then
        if python -m pip install "warp-lang==${WARP_LANG_VERSION}" -i "${url}" --trusted-host "${host}"; then
          ok=1
          break
        fi
      else
        if python -m pip install "warp-lang==${WARP_LANG_VERSION}" -i "${url}"; then
          ok=1
          break
        fi
      fi
    done
    if [[ "${ok}" != "1" ]]; then
      echo "[prepare_eval] mirrors failed, retrying default pip index ..."
      python -m pip install "warp-lang==${WARP_LANG_VERSION}"
    fi
  fi
  python -c "import warp; import importlib.metadata as m; v=m.version('warp-lang'); assert v=='${WARP_LANG_VERSION}', v; print('[prepare_eval] warp-lang', v, 'torch_attr', hasattr(warp,'torch'))"
}

# Link src -> dst, or skip if already correct / already a usable local tree.
# Optional 3rd arg: relative marker that must exist under the resolved dst
# (e.g. "objects" for assets).
ensure_path() {
  local src="${1:-}"
  local dst="$2"
  local marker="${3:-}"
  local dst_ok=0
  local src_real="" dst_real=""

  if [[ -e "${dst}" || -L "${dst}" ]]; then
    if [[ -z "${marker}" || -e "${dst}/${marker}" ]]; then
      dst_ok=1
    fi
  fi

  if [[ -n "${src}" && -e "${src}" ]]; then
    src_real="$(realpath "${src}")"
  fi
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    dst_real="$(realpath "${dst}")"
  fi

  # Same real path (in-tree install): nothing to do.
  if [[ -n "${src_real}" && -n "${dst_real}" && "${src_real}" == "${dst_real}" ]]; then
    echo "[prepare_eval] skip link ${dst} (already at ${dst_real})"
    return
  fi

  # Symlink already points at the requested src.
  if [[ -L "${dst}" && -n "${src_real}" && "${dst_real}" == "${src_real}" ]]; then
    echo "[prepare_eval] skip link ${dst} (already -> ${dst_real})"
    return
  fi

  # Destination already usable and no src provided (or src missing): keep it.
  if [[ "${dst_ok}" == "1" && ( -z "${src}" || -z "${src_real}" ) ]]; then
    echo "[prepare_eval] skip link ${dst} (already present)"
    return
  fi

  # Destination is a real populated tree: do not replace with a symlink.
  if [[ "${dst_ok}" == "1" && ! -L "${dst}" ]]; then
    echo "[prepare_eval] skip link ${dst} (local tree already present)"
    return
  fi

  if [[ -z "${src_real}" ]]; then
    echo "[prepare_eval] missing ${dst}; set ROBOTWIN_PREBUILT or ${1:+ASSETS_SRC/CUROBO_SRC} / CHECKPOINTS_SRC" >&2
    echo "[prepare_eval] looked for src: ${src:-<empty>}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${dst}")"
  ln -sfn "${src_real}" "${dst}"
  echo "[prepare_eval] ${dst} -> ${src_real}"
}

shared_pretrained_ready() {
  [[ -f "${CKPT_DST}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" ]] \
    && [[ -d "${CKPT_DST}/DiffSynth-Studio/Wan-Series-Converted-Safetensors" ]] \
    && [[ -d "${CKPT_DST}/Wan2.2-TI2V-5B/google/umt5-xxl" || -d "${CKPT_DST}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" ]]
}

ensure_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD=(hf download)
    return
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_DOWNLOAD=(huggingface-cli download)
    return
  fi
  echo "[prepare_eval] installing huggingface_hub (provides hf / huggingface-cli) ..."
  python -m pip install -U 'huggingface_hub[cli]'
  if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD=(hf download)
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_DOWNLOAD=(huggingface-cli download)
  else
    echo "[prepare_eval] huggingface_hub installed but CLI not on PATH" >&2
    exit 1
  fi
}

echo "[prepare_eval] root: ${ROOT}"

ensure_warp_lang

ensure_path "${POLICY_SRC}" "${POLICY_DST}"
ensure_path "${ASSETS_SRC}" "${ASSETS_DST}" "objects"
ensure_path "${CUROBO_SRC}" "${CUROBO_DST}"

mkdir -p "${CKPT_DST}"

if [[ -n "${CHECKPOINTS_SRC}" ]]; then
  echo "[prepare_eval] ensuring shared pretrained weights from ${CHECKPOINTS_SRC}"
  for name in "${SHARED_CKPT_NAMES[@]}"; do
    if [[ -e "${CHECKPOINTS_SRC}/${name}" ]]; then
      ensure_path "${CHECKPOINTS_SRC}/${name}" "${CKPT_DST}/${name}"
    else
      echo "[prepare_eval] CHECKPOINTS_SRC missing ${name} (skip)"
    fi
  done
fi

echo "[prepare_eval] verifying RoboTwin paths ..."
ls "${ASSETS_DST}/objects" "${CUROBO_DST}" "${POLICY_DST}" >/dev/null
echo "[prepare_eval] paths ok"

if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  echo "[prepare_eval] SKIP_DOWNLOAD=1, skipping checkpoint downloads"
  exit 0
fi

if shared_pretrained_ready; then
  echo "[prepare_eval] skip download_pretrained (ActionDiT / DiffSynth / tokenizer already present)"
else
  echo "[prepare_eval] downloading text encoder / ActionDiT backbone ..."
  bash "${ROOT}/scripts/download_pretrained.sh"
fi

CKPT_PT="${CKPT_DST}/robotwin/step_037645.pt"
CKPT_STATS="${CKPT_DST}/robotwin/dataset_stats.json"
if [[ -f "${CKPT_PT}" && -f "${CKPT_STATS}" && "${FORCE_CKPT_DOWNLOAD:-0}" != "1" ]]; then
  echo "[prepare_eval] skip robotwin ckpt download (already present)"
else
  ensure_hf_cli
  echo "[prepare_eval] downloading LoopWAM RoboTwin checkpoint (~12GB) ..."
  "${HF_DOWNLOAD[@]}" loop-wam/loopwam \
    robotwin/step_037645.pt \
    robotwin/dataset_stats.json \
    --repo-type dataset \
    --local-dir "${CKPT_DST}"
fi

for f in "${CKPT_PT}" "${CKPT_STATS}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[prepare_eval] expected file missing: ${f}" >&2
    exit 1
  fi
done

echo "[prepare_eval] done"
echo "[prepare_eval] robotwin ckpt under checkpoints/robotwin/"
