#!/usr/bin/env bash
# Bake one Python reference scenario into tests/fixtures/expected/<scenario>/.
#
# Usage:  scripts/run_python_reference.sh <fixture-dir-win> <out-dir-win> \
#                                           <algorithm> <temporal-phase> [<use_phase>]
#
# Run from inside WSL. The first two args are Windows-style paths (we convert
# with `wslpath`). The Python module is invoked through micromamba so the
# nordicenv environment is used.
set -euo pipefail

MM="${MICROMAMBA:-/home/tsalo/.local/bin/micromamba}"
ENV_PREFIX="${NORDICENV:-/home/tsalo/micromamba/envs/nordicenv}"

if [[ $# -lt 4 ]]; then
    echo "usage: $0 <fixture-dir-win> <out-dir-win> <algorithm> <temporal-phase> [use_phase]" >&2
    exit 1
fi

FIX="$(wslpath "$1")"
OUT="$(wslpath "$2")"
ALGO="$3"
TPHASE="$4"
USE_PHASE="${5:-true}"

mkdir -p "$OUT"

ARGS=(
    -m "$FIX/synthetic_mag.nii.gz"
    --mag-norf "$FIX/synthetic_mag_noRF.nii.gz"
    --out-dir "$OUT"
    --algorithm "$ALGO"
    --temporal-phase "$TPHASE"
    --save-gfactor-map
)
if [[ "$USE_PHASE" == "true" ]]; then
    ARGS+=( -p "$FIX/synthetic_pha.nii.gz" --phase-norf "$FIX/synthetic_pha_noRF.nii.gz" )
fi

"$MM" run -p "$ENV_PREFIX" python -m nordic.cli.nordic_filewise "${ARGS[@]}"
