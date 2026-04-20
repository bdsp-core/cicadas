"""Full CICADAS Python pipeline orchestrator — Python port of matlab/run_all.m.

Runs stages a0 -> a1 -> a2 -> a3 -> a4_* -> figure scripts -> sensitivity in
order. Each stage is invoked as its own Python subprocess so that per-stage
module-level state (np.random seeds, global imports) does not leak between
stages — matches how MATLAB re-enters each script.

Usage (from cicadas/python/):
    python run_all.py

Or:
    python run_all.py --skip-figures --skip-sensitivity   # analysis only
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # cicadas/python
REPO_ROOT = HERE.parent                          # cicadas/
PY = sys.executable

ANALYSIS_STAGES = [
    ("generatetrialdata.py",                    "a0_GenerateTrialData",           25),
    ("generate_trial_data_dose_changing.py",    "a0_GenerateDoseSwitchingData",   90),
    ("EstimatePKPD.py",                         "a1_EstimatePKPD",                85),
    ("CausalSurvivalAnalysis.py",               "a2_CausalSurvivalAnalysis",    2400),
    ("ThreeTreatmentTargets.py",                "a3_ThreeTreatmentTargets",       15),
    ("HeatMap_Agressive.py",                    "a4_HeatMap_Agressive",          180),
    ("OptimalTreatmentTarget.py",               "a4_OptimalTreatmentTarget",    3600),
    ("Optimize_Heatmap.py",                     "a4_Optimize_Heatmap",          2700),
]

# Figure scripts live in cicadas/CICADA_FIGURES_PY (Python port)
FIGURE_STAGES = [
    "a1_SingleTraces.py",
    "a2_EvaluatePKPD_estimates_figures.py",
    "a3a_Fig_Swimmers_RCT.py",
    "a3b_Fig_Swimmers_Obs_Naive.py",
    "a3c_Fig_Swimmers_Obs_g_formula.py",
    "a4_HeatMaps_Combined.py",
    "a5_OptimizationCurve.py",
]

SENSITIVITY_STAGES = [
    "sensitivity/alt_censoring/run_alt_censoring.py",
    "sensitivity/measurement_error/run_measurement_error.py",
    "sensitivity/nuc_injection/run_nuc_sensitivity.py",
]


def run_stage(label: str, script: Path, cwd: Path, log_dir: Path, env: dict) -> int:
    log = log_dir / f"{label}.log"
    print(f"\n=== STAGE {label} ===  ({script.relative_to(REPO_ROOT)})")
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(
            [PY, str(script)],
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    dt = time.time() - t0
    status = "ok" if rc == 0 else f"FAILED rc={rc}"
    print(f"[{label}] {status} elapsed={dt:.0f}s log={log.relative_to(REPO_ROOT)}")
    return rc


def snapshot_outputs(output_dir: Path) -> None:
    """Copy Python-produced outputs at the repo root into python_outputs/."""
    fig_out = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    patterns = ["*.mat", "trialData*.csv", "*_results_*.txt", "*_summary*.txt"]
    for pat in patterns:
        for f in REPO_ROOT.glob(pat):
            shutil.copy2(f, output_dir / f.name)

    # Figures from the Python figure dir (once ported)
    fig_dir = REPO_ROOT / "CICADA_FIGURES_PY"
    if fig_dir.exists():
        for f in list(fig_dir.glob("*.pdf")) + list(fig_dir.glob("*.mat")) + list(fig_dir.glob("*.txt")):
            shutil.copy2(f, fig_out / f.name)

    # Sensitivity artifacts
    for sub in ["alt_censoring", "measurement_error", "nuc_injection"]:
        sdir = REPO_ROOT / "sensitivity" / sub
        if sdir.exists():
            target = output_dir / f"sens_{sub}"
            target.mkdir(exist_ok=True)
            for f in list(sdir.glob("*.mat")) + list(sdir.glob("*.pdf")) + list(sdir.glob("*.txt")):
                shutil.copy2(f, target / f.name)


def write_manifest(output_dir: Path) -> None:
    manifest = output_dir / "MANIFEST.txt"
    with open(manifest, "w") as f:
        f.write(f"Python reference run captured at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Seed: np.random.seed(0) in a0/a4 scripts (numpy PCG64 upgraded from MT19937)\n")
        f.write("Note: Python PCG64 stream != MATLAB MT19937 stream; statistical parity only.\n\n")
        for item in sorted(output_dir.rglob("*")):
            if item.is_file():
                rel = item.relative_to(output_dir)
                f.write(f"{str(rel):<60}  {item.stat().st_size:>10} bytes\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-sensitivity", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--only", nargs="+", help="only run stages whose label starts with any of these", default=[])
    args = p.parse_args()

    output_dir = REPO_ROOT / "python_outputs"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Put python/ on PYTHONPATH so helper modules can be imported from any cwd.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(HERE) + (os.pathsep + existing_pp if existing_pp else "")

    t0_all = time.time()
    failures = []

    if not args.skip_analysis:
        for fname, label, _expected_s in ANALYSIS_STAGES:
            if args.only and not any(label.startswith(x) for x in args.only):
                continue
            script = HERE / fname
            if not script.exists():
                print(f"[{label}] SKIP — script not found: {script}")
                continue
            # Analysis stages run from REPO_ROOT because they read/write trialData*.csv there.
            rc = run_stage(label, script, REPO_ROOT, log_dir, env)
            if rc != 0:
                failures.append(label)

    if not args.skip_figures:
        fig_dir_py = REPO_ROOT / "CICADA_FIGURES_PY"
        for fname in FIGURE_STAGES:
            if args.only and not any(fname.startswith(x) for x in args.only):
                continue
            script = fig_dir_py / fname
            if not script.exists():
                print(f"[fig {fname}] SKIP — not yet ported ({script} missing)")
                continue
            rc = run_stage(f"fig_{fname}", script, fig_dir_py, log_dir, env)
            if rc != 0:
                failures.append(f"fig_{fname}")

    if not args.skip_sensitivity:
        for rel in SENSITIVITY_STAGES:
            if args.only and not any(Path(rel).name.startswith(x) for x in args.only):
                continue
            script = REPO_ROOT / rel
            if not script.exists():
                print(f"[sens {rel}] SKIP — not yet ported ({script} missing)")
                continue
            rc = run_stage(f"sens_{Path(rel).parent.name}", script, script.parent, log_dir, env)
            if rc != 0:
                failures.append(f"sens_{Path(rel).parent.name}")

    snapshot_outputs(output_dir)
    write_manifest(output_dir)

    dt = (time.time() - t0_all) / 60
    print(f"\n[run_all.py] finished in {dt:.1f} min")
    print(f"[run_all.py] outputs: {output_dir}")
    if failures:
        print(f"[run_all.py] failures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
