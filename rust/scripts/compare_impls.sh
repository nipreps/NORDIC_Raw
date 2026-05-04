#!/usr/bin/env bash
# Run Python and Rust NORDIC implementations on matched inputs and capture
# wall time + peak RSS for each invocation. Designed to run inside WSL so both
# implementations are timed under the same kernel/scheduler/glibc.
#
# Reads:
#   tests/fixtures/synthetic_*.nii.gz        (auto-generated if missing)
#   /mnt/c/.../ds006131/sub-20188/.../*.nii.gz   (real fMRI data, 110x110x72x363)
#
# Writes:
#   comparison_results/timings.csv            (raw per-iter measurements)
#   comparison_results/<dataset>/<scenario>/python/   (canonical iter-1 outputs)
#   comparison_results/<dataset>/<scenario>/rust/
#
# Run scripts/compare_impls.py afterwards (or let this script invoke it at the
# end) to produce parity.csv and summary.md.
#
# Knobs (env vars):
#   ITERS=3              Iterations per (dataset, scenario, impl).
#   DATASETS="synthetic real"   Subset of datasets to run.
#   SCENARIOS_LIST=...   Override the scenario matrix (see SCENARIOS array below).
#   SKIP_REAL=1          Skip the real-data dataset entirely.
#   MICROMAMBA=...       Path to micromamba binary.
#   NORDICENV=...        Prefix of the conda env with the Python `nordic` package.
set -euo pipefail

CRATE="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$CRATE/comparison_results"
TIMINGS_CSV="$RESULTS/timings.csv"

ITERS="${ITERS:-3}"
MM="${MICROMAMBA:-/home/tsalo/.local/bin/micromamba}"
ENV_PREFIX="${NORDICENV:-/home/tsalo/micromamba/envs/nordicenv}"

# Real data (BIDS run from ds006131). Symlinked through git-annex; the symlink
# resolves to the real ~515MB content.
REAL_DIR=/mnt/c/Users/tsalo/Documents/datasets/ds006131/sub-20188/ses-1/func
REAL_BASE=sub-20188_ses-1_task-rat_dir-PA_run-01_echo-1_part
REAL_MAG="$REAL_DIR/${REAL_BASE}-mag_bold.nii.gz"
REAL_PHA="$REAL_DIR/${REAL_BASE}-phase_bold.nii.gz"
REAL_MAG_NORF="$REAL_DIR/${REAL_BASE}-mag_noRF.nii.gz"
REAL_PHA_NORF="$REAL_DIR/${REAL_BASE}-phase_noRF.nii.gz"

FIX_DIR="$CRATE/tests/fixtures"
FIX_MAG="$FIX_DIR/synthetic_mag.nii.gz"
FIX_PHA="$FIX_DIR/synthetic_pha.nii.gz"
FIX_MAG_NORF="$FIX_DIR/synthetic_mag_noRF.nii.gz"
FIX_PHA_NORF="$FIX_DIR/synthetic_pha_noRF.nii.gz"

DATASETS_DEFAULT="synthetic real"
DATASETS="${DATASETS:-$DATASETS_DEFAULT}"

# (algorithm, temporal_phase, use_phase) tuples to time. Defaults match the
# Rust parity-test matrix.
SCENARIOS=(
    "nordic 1 true"
    "nordic 3 true"
    "nordic 1 false"
    "mppca 1 true"
    "gfactor+mppca 1 true"
)

# Make sure /usr/bin/time exists — `time` builtin doesn't have -v.
if ! [[ -x /usr/bin/time ]]; then
    echo "ERROR: /usr/bin/time not installed (apt install time)." >&2
    exit 1
fi

# Build Rust release binary if missing.
RUST_BIN="$CRATE/target/release/nordic-rs"
if [[ ! -x "$RUST_BIN" ]]; then
    echo "==> Building Rust release binary"
    (cd "$CRATE" && cargo build --release)
fi

# Auto-generate synthetic fixtures if missing.
if [[ ! -f "$FIX_MAG" ]]; then
    echo "==> Generating synthetic fixtures"
    "$MM" run -p "$ENV_PREFIX" python "$FIX_DIR/synthetic_make.py"
fi

# Sanity-check real data (only matters if it's in the dataset list).
if [[ " $DATASETS " == *" real "* ]] && [[ "${SKIP_REAL:-0}" != "1" ]]; then
    for f in "$REAL_MAG" "$REAL_PHA" "$REAL_MAG_NORF" "$REAL_PHA_NORF"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: missing real-data input $f" >&2
            exit 1
        fi
    done
fi

mkdir -p "$RESULTS"
# Reset the timings CSV — we want a clean record per harness invocation.
echo "dataset,impl,algorithm,temporal_phase,use_phase,iter,wall_seconds,user_seconds,sys_seconds,peak_rss_kb,exit_code" \
    > "$TIMINGS_CSV"

# --- helper: run one impl once, append a row to TIMINGS_CSV --------------------
# Args: dataset impl algo tphase use_phase iter mag pha mag_norf pha_norf out_dir
run_once() {
    local dataset="$1" impl="$2" algo="$3" tphase="$4" use_phase="$5" iter="$6"
    local mag="$7" pha="$8" mag_norf="$9" pha_norf="${10}" out_dir="${11}"

    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    local time_log
    time_log=$(mktemp)

    local pha_args=()
    if [[ "$use_phase" == "true" ]]; then
        pha_args=(-p "$pha" --phase-norf "$pha_norf")
    fi

    local rc=0
    case "$impl" in
        python)
            /usr/bin/time -v -o "$time_log" \
                "$MM" run -p "$ENV_PREFIX" python -m nordic.cli.nordic_filewise \
                    -m "$mag" \
                    "${pha_args[@]}" \
                    --mag-norf "$mag_norf" \
                    --out-dir "$out_dir" \
                    --algorithm "$algo" \
                    --temporal-phase "$tphase" \
                    --save-gfactor-map \
                    --debug \
                    > "$out_dir/_stdout.log" 2> "$out_dir/_stderr.log" \
                || rc=$?
            ;;
        rust)
            /usr/bin/time -v -o "$time_log" \
                "$RUST_BIN" \
                    -m "$mag" \
                    "${pha_args[@]}" \
                    --mag-norf "$mag_norf" \
                    --out-dir "$out_dir" \
                    --algorithm "$algo" \
                    --temporal-phase "$tphase" \
                    --save-gfactor-map \
                    --debug \
                    > "$out_dir/_stdout.log" 2> "$out_dir/_stderr.log" \
                || rc=$?
            ;;
        *)
            echo "unknown impl $impl" >&2
            return 2
            ;;
    esac

    # Parse /usr/bin/time -v output. Format examples:
    #   Elapsed (wall clock) time (h:mm:ss or m:ss): 1:23.45
    #   User time (seconds): 12.34
    #   System time (seconds): 0.56
    #   Maximum resident set size (kbytes): 1234567
    local wall_raw user_s sys_s rss_kb wall_s
    wall_raw=$(awk -F': ' '/Elapsed \(wall clock\)/ {print $NF}' "$time_log")
    user_s=$(awk -F': ' '/User time \(seconds\)/ {print $NF}' "$time_log")
    sys_s=$(awk -F': ' '/System time \(seconds\)/ {print $NF}' "$time_log")
    rss_kb=$(awk -F': ' '/Maximum resident set size/ {print $NF}' "$time_log")

    # Convert hh:mm:ss / mm:ss to seconds.
    wall_s=$(python3 -c '
import sys
parts = sys.argv[1].split(":")
if len(parts) == 3:
    print(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
elif len(parts) == 2:
    print(int(parts[0]) * 60 + float(parts[1]))
else:
    print(float(parts[0]))
' "$wall_raw")

    echo "$dataset,$impl,$algo,$tphase,$use_phase,$iter,$wall_s,$user_s,$sys_s,$rss_kb,$rc" \
        >> "$TIMINGS_CSV"
    rm -f "$time_log"
    return "$rc"
}

# --- main loop ----------------------------------------------------------------
for dataset in $DATASETS; do
    if [[ "$dataset" == "synthetic" ]]; then
        mag="$FIX_MAG" pha="$FIX_PHA" mag_norf="$FIX_MAG_NORF" pha_norf="$FIX_PHA_NORF"
    elif [[ "$dataset" == "real" ]]; then
        if [[ "${SKIP_REAL:-0}" == "1" ]]; then
            continue
        fi
        mag="$REAL_MAG" pha="$REAL_PHA" mag_norf="$REAL_MAG_NORF" pha_norf="$REAL_PHA_NORF"
    else
        echo "unknown dataset $dataset" >&2
        continue
    fi

    for scen in "${SCENARIOS[@]}"; do
        read -r algo tphase use_phase <<<"$scen"
        safe="${algo//+/_}_tp${tphase}_pha${use_phase}"
        scen_dir="$RESULTS/$dataset/$safe"
        py_canonical="$scen_dir/python"
        rs_canonical="$scen_dir/rust"

        echo "=== $dataset / $algo / tp=$tphase / use_phase=$use_phase ==="

        for iter in $(seq 1 "$ITERS"); do
            # Iteration 1's output is kept on disk for parity comparison.
            # Later iterations write into a tmp dir so we don't waste storage.
            if [[ "$iter" == "1" ]]; then
                py_out="$py_canonical"
                rs_out="$rs_canonical"
            else
                tmp=$(mktemp -d)
                py_out="$tmp/python"
                rs_out="$tmp/rust"
            fi

            echo "  iter $iter/$ITERS python -> $py_out"
            run_once "$dataset" python "$algo" "$tphase" "$use_phase" "$iter" \
                "$mag" "$pha" "$mag_norf" "$pha_norf" "$py_out" || true

            echo "  iter $iter/$ITERS rust   -> $rs_out"
            run_once "$dataset" rust   "$algo" "$tphase" "$use_phase" "$iter" \
                "$mag" "$pha" "$mag_norf" "$pha_norf" "$rs_out" || true

            if [[ "$iter" != "1" ]]; then
                rm -rf "$tmp"
            fi
        done
    done
done

echo
echo "==> Wrote $TIMINGS_CSV"
echo "==> Running parity comparator"
"$MM" run -p "$ENV_PREFIX" python "$CRATE/scripts/compare_impls.py" "$RESULTS"
echo "==> Done. See $RESULTS/summary.md"
