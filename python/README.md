# CICADAS — Python port

A full Python port of the CICADAS MATLAB implementation. Both ports are authoritative;
the Python port has been parity-verified against the MATLAB reference run (see
`../PARITY_REPORT.md`).

Bit-identical parity is not achievable because MATLAB (Mersenne Twister) and
NumPy (PCG64) produce different random streams from the same seed — but every
paper-headline quantity matches MATLAB within the `ok` stochastic-parity tier
(mean-z < 0.5, sign match, quantile rel < 20%).

## Running

Orchestrator (full pipeline, ~3–4 hours; parallel analysis + figures + sensitivity):

```bash
python3 run_all.py
```

Options:
- `--only <stage-prefix>` — run only stages whose label matches the prefix
- `--skip-figures` — skip the `CICADA_FIGURES_PY/` figure scripts
- `--skip-sensitivity` — skip the `sensitivity/*/run_*.py` drivers
- `--skip-analysis` — skip the main a0–a4 pipeline

Individual stages can be invoked directly; they read and write from the repo
root (`cicadas/`) the same way MATLAB does.

Quick sanity check (imports + tiny-N sims, ~5 s):

```bash
python3 test_smoke.py
```

## Scripts

Parallel to the MATLAB `a0`–`a4` naming:

| Python | MATLAB | Stage |
|---|---|---|
| `generatetrialdata.py` | `matlab/a0_GenerateTrialData.m` | RCT + observational simulation |
| `generate_trial_data_dose_changing.py` | `matlab/a0_GenerateDoseSwitchingData.m` | Dose-switching cohort for ke identifiability |
| `EstimatePKPD.py` | `matlab/a1_EstimatePKPD.m` | Four-method PK/PD parameter estimation |
| `CausalSurvivalAnalysis.py` | `matlab/a2_CausalSurvivalAnalysis.m` | g-formula + 1000-rep bootstrap |
| `ThreeTreatmentTargets.py` | `matlab/a3_ThreeTreatmentTargets.m` | Survival comparison across 3 thresholds |
| `HeatMap_Agressive.py` | `matlab/a4_HeatMap_Agressive.m` | Aggressive-treatment sensitivity heatmap |
| `OptimalTreatmentTarget.py` | `matlab/a4_OptimalTreatmentTarget.m` | Optimal-θ* search with parallel bootstrap |
| `Optimize_Heatmap.py` | `matlab/a4_Optimize_Heatmap.m` | (C̄, ḡ) stratified ATE grid |
| `fcn*.py` | `matlab/fcn*.m` | Simulation + estimation helpers |

Figure scripts are in `../CICADA_FIGURES_PY/` and mirror `../CICADA_FIGURES/`.
Sensitivity analyses are in `../sensitivity/{alt_censoring,measurement_error,nuc_injection}/`.

## Orchestration

`run_all.py` runs each stage in its own Python subprocess to avoid module-level
state bleed (seeds, globals, imports). Each stage's stdout/stderr is captured in
`../python_outputs/logs/<stage>.log`. At the end, produced artifacts are
snapshotted into `../python_outputs/` for comparison against `../matlab_outputs/`
via `../verify_parity.py`.

## Seeds and reproducibility

Every stochastic script begins with `np.random.seed(0)`, matching MATLAB's
`rng(0)` in the authoritative sources. Within the Python world, runs are
byte-for-byte reproducible. Cross-language bit-parity with MATLAB is not
achievable (PCG64 vs MT19937). The `verify_parity.py` harness compares
statistical summaries (mean, std, quantiles) rather than bit equality.

## Requirements

Python ≥ 3.10.

```bash
pip install -r requirements.txt
# or
pip install numpy scipy pandas matplotlib statsmodels lifelines h5py
```

`h5py` is optional; it is used to read MATLAB v7.3 `.mat` files when
`scipy.io.loadmat` cannot.

## Known trade-offs vs MATLAB

- **Optimizers.** `fmincon` (MATLAB) ↔ `scipy.optimize.minimize` with SLSQP or L-BFGS-B. Tolerances chosen to match; parameter estimates agree within rel 1%.
- **Kalman/RTS.** EKF forward pass + RTS backward smoother implemented verbatim from the MATLAB source.
- **Parallelism.** `parfor` ↔ `concurrent.futures.ProcessPoolExecutor` with 10 workers and per-worker seeds.
- **Figures.** Matplotlib instead of MATLAB handles. Layout and content match; pixel positions do not.
- **`sensitivity/nuc_injection`.** Defaults to "quick" mode (N=400, 1 seed/cell, ~6 min). Set `CICADAS_NUC_FULL=1` to run the full 5×5×3-seed N=1500 grid (~3–5 hours).

## Status (2026-04-20)

Full parity verification completed against a fresh MATLAB reference run. The
paper's headline result matches:

| Quantity | MATLAB | Python | Paper |
|---|---|---|---|
| g-formula ATE at 168 h (`a2_CausalSurvivalAnalysis`) | +14.4% | +14.1% | +14.5% |

See `../PARITY_REPORT.md` for per-artifact detail of every `.mat`/`.csv` produced.

Known residual: `a4_OptimalTreatmentTarget` reports ATE magnitudes across a
threshold sweep; at extreme thresholds (θ<0.1) Python over-reports ~2× versus
MATLAB even with matched `L0`. Both runs agree on the optimal θ* = 0.02 and the
qualitative shape of the ATE-vs-θ curve; absolute magnitudes differ due to
hazard-stream sensitivity amplified by the nonlinear mortality model near θ=0.
Not a showstopper — this is a sensitivity-sweep script, not the paper's primary
endpoint — but worth noting to reviewers.

Every non-residual `drift` case has a documented explanation (intentional N
differences, RNG-stream divergence, or large-N mean-z artifacts where
quantile-relative differences are < 5%).

The `python.backup-2026-04-19/` directory alongside this folder is a snapshot of
the previous "rudimentary" port before this parity work; it can be discarded
once the new port is accepted.
