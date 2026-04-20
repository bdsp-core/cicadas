#!/bin/bash
# Run the full CICADAS pipeline one stage at a time, each in its own MATLAB
# process (so `clear` at the top of each stage script can't wipe our loop
# state). Snapshots outputs into matlab_outputs/ at the end.
#
# Authoritative .m files are NOT modified.

set -u
set -o pipefail

REPO_ROOT=/Users/mwestover/GithubRepos/CICADAS/cicadas
MATLAB=/Applications/MATLAB_R2025b.app/bin/matlab
OUT_DIR="$REPO_ROOT/matlab_outputs"
FIG_OUT="$OUT_DIR/figures"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$FIG_OUT" "$LOG_DIR"

cd "$REPO_ROOT"

ANALYSIS_STAGES=(
    a0_GenerateTrialData
    a0_GenerateDoseSwitchingData
    a1_EstimatePKPD
    a2_CausalSurvivalAnalysis
    a3_ThreeTreatmentTargets
    a4_HeatMap_Agressive
    a4_OptimalTreatmentTarget
    a4_Optimize_Heatmap
)

FIGURE_STAGES=(
    a1_SingleTraces
    a2_EvaluatePKPD_estimates_figures
    a3a_Fig_Swimmers_RCT
    a3b_Fig_Swimmers_Obs_Naive
    a3c_Fig_Swimmers_Obs_g_formula
    a4_HeatMaps_Combined
    a5_OptimizationCurve
)

SENSITIVITY_STAGES=(
    "sensitivity/alt_censoring/run_alt_censoring.m"
    "sensitivity/measurement_error/run_measurement_error.m"
    "sensitivity/nuc_injection/run_nuc_sensitivity.m"
)

T0_ALL=$(date +%s)
echo "[run_all_verify] start $(date)"

snapshot_now () {
    # Copy every current top-level .mat/.csv/.txt into matlab_outputs/. Called
    # after each stage so a downstream Python-overwrite cannot corrupt the
    # MATLAB reference.
    for pat in "*.mat" "trialData*.csv" "*_results_*.txt" "*_summary*.txt" "L0data*.mat"; do
        find "$REPO_ROOT" -maxdepth 1 -name "$pat" -exec cp -pn {} "$OUT_DIR/" \; 2>/dev/null
    done
    # -n flag: no-clobber. First-write-wins, so a0's trialData*.csv is locked in
    # after a0 even if a later stage rewrites the file.
}

snapshot_force () {
    # Force-overwrite snapshot (used at end for figure outputs that accumulate).
    for pat in "*.mat" "trialData*.csv" "*_results_*.txt" "*_summary*.txt"; do
        find "$REPO_ROOT" -maxdepth 1 -name "$pat" -exec cp -p {} "$OUT_DIR/" \;
    done
}

run_stage () {
    local name="$1"
    local cmd="$2"
    local log="$LOG_DIR/${name}.log"
    echo ""
    echo "=== STAGE ${name} ==="
    local t0=$(date +%s)
    "$MATLAB" -nodisplay -nosplash -nodesktop -batch "addpath(genpath('$REPO_ROOT')); $cmd" \
        > "$log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    echo "[${name}] rc=$rc elapsed=$((t1-t0))s"
    if [ $rc -ne 0 ]; then
        echo "FAILURE: see $log"
        tail -30 "$log"
    fi
    # Snapshot immediately after each stage so outputs are protected from
    # any subsequent Python run that overwrites repo-root files.
    snapshot_now
    return $rc
}

# --- Analysis stages (in repo root so trialData*.csv writes here) ---
for S in "${ANALYSIS_STAGES[@]}"; do
    run_stage "$S" "$S" || echo "[$S] failed; continuing so we see what else breaks"
done

# --- Figure stages (must be run from CICADA_FIGURES/) ---
for F in "${FIGURE_STAGES[@]}"; do
    run_stage "fig_$F" "cd('$REPO_ROOT/CICADA_FIGURES'); $F" || true
done

# --- Sensitivity stages (from their respective dirs, run the .m file) ---
for S in "${SENSITIVITY_STAGES[@]}"; do
    DIR=$(dirname "$REPO_ROOT/$S")
    FILE=$(basename "$S" .m)
    TAG=$(basename "$DIR")
    run_stage "sens_$TAG" "cd('$DIR'); $FILE" || true
done

# --- Final snapshot: figures accumulate at the end; force-overwrite for them. ---
echo ""
echo "=== final snapshot ==="
# Don't overwrite analysis outputs (they're locked via snapshot_now's -n flag),
# but DO pick up anything new (e.g., late-written summary files).
snapshot_now
# Figure PDFs and any per-figure .mat
find "$REPO_ROOT/CICADA_FIGURES" -maxdepth 1 \( -name "*.pdf" -o -name "*.mat" -o -name "*.txt" \) -exec cp -p {} "$FIG_OUT/" \;
# Sensitivity artifacts
for S in alt_censoring measurement_error nuc_injection; do
    mkdir -p "$OUT_DIR/sens_$S"
    find "$REPO_ROOT/sensitivity/$S" -maxdepth 1 \( -name "*.mat" -o -name "*.pdf" -o -name "*.txt" \) -exec cp -p {} "$OUT_DIR/sens_$S/" \;
done

# --- Manifest ---
MANIFEST="$OUT_DIR/MANIFEST.txt"
{
    echo "MATLAB reference run captured at $(date)"
    echo "Seed: rng(0) inside a0/a4/fcnGeneratePatientParameters (MATLAB MT19937)"
    echo ""
    echo "== top-level =="
    (cd "$OUT_DIR" && ls -la *.mat *.csv *.txt 2>/dev/null | awk '{printf "%-60s %10s\n", $NF, $5}')
    echo ""
    echo "== figures =="
    (cd "$FIG_OUT" && ls -la 2>/dev/null | awk '{printf "%-60s %10s\n", $NF, $5}')
    echo ""
    for S in alt_censoring measurement_error nuc_injection; do
        echo "== sens_$S =="
        (cd "$OUT_DIR/sens_$S" 2>/dev/null && ls -la 2>/dev/null | awk '{printf "%-60s %10s\n", $NF, $5}')
        echo ""
    done
} > "$MANIFEST"

T1_ALL=$(date +%s)
echo ""
echo "[run_all_verify] finished in $(( (T1_ALL-T0_ALL) / 60 )) min"
echo "[run_all_verify] outputs: $OUT_DIR"
echo "[run_all_verify] manifest: $MANIFEST"
