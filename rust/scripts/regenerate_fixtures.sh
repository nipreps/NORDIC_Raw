#!/usr/bin/env bash
# Rebuild the parity-test scenario matrix under tests/fixtures/expected/.
#
# Run from the crate root, inside WSL.
set -euo pipefail

CRATE="$(cd "$(dirname "$0")/.." && pwd)"
FIX_WIN="$(wslpath -w "$CRATE/tests/fixtures")"

# (Re)generate the synthetic NIfTIs first.
MM="${MICROMAMBA:-/home/tsalo/.local/bin/micromamba}"
ENV_PREFIX="${NORDICENV:-/home/tsalo/micromamba/envs/nordicenv}"
"$MM" run -p "$ENV_PREFIX" python "$CRATE/tests/fixtures/synthetic_make.py"

scenarios=(
    "nordic 1 true"
    "nordic 3 true"
    "nordic 1 false"
    "mppca 1 true"
    "mppca 3 true"
    "mppca 1 false"
    "gfactor+mppca 1 true"
    "gfactor+mppca 3 true"
)

for scen in "${scenarios[@]}"; do
    read -r algo tphase use_phase <<<"$scen"
    safe_algo="${algo//+/_}"
    out_rel="tests/fixtures/expected/${safe_algo}_tp${tphase}_pha${use_phase}"
    out_win="$(wslpath -w "$CRATE/$out_rel")"
    rm -rf "$CRATE/$out_rel"
    mkdir -p "$CRATE/$out_rel"
    echo "==> $algo / temporal_phase=$tphase / use_phase=$use_phase"
    "$CRATE/scripts/run_python_reference.sh" "$FIX_WIN" "$out_win" "$algo" "$tphase" "$use_phase"
done

echo "All reference scenarios rebuilt."
