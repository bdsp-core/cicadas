# CICADAS Python-port audit

Date: 2026-04-19.  Scope: `matlab/` + `CICADA_FIGURES/` + `sensitivity/` vs `python/` + `benchmarks/`.

Key verdict up front: the four `a*` estimator/simulation cores (`a0`, `a1`, `a2`, `a3`, `a4_*`) have Python counterparts that are **structurally close** to MATLAB but carry several concrete numerical and functional drifts (detailed §3).  All 10 figure scripts and 4 swimmer helpers in `CICADA_FIGURES/` and all 4 sensitivity scripts are **missing**.  `a3_ThreeTreatmentTargets` and `a2_CausalSurvivalAnalysis` drop large text-file summary blocks.  `generate_trial_data_dose_changing.py` is **functionally wrong** (calls the RCT simulator instead of `fcnSimulate_DoseChanging`).

---

## 1. File-by-file parity table

### 1a. `matlab/` — analysis core

| MATLAB file | Python file | M LOC | Py LOC | Status | Note |
|---|---|---:|---:|---|---|
| a0_GenerateTrialData.m | generatetrialdata.py | 476 | 412 | drift | g-formula block (M:218-272) is commented out (P:315-349); text-summary block (M:353-476) not ported. Plot formatting simplified. |
| a0_GenerateDoseSwitchingData.m | generate_trial_data_dose_changing.py | 96 | 133 | drift | **Wrong simulator**: calls `fcnSimulate_N_Patients(RCT=1, treatProb=0.5)` (P:73-82) instead of `fcnSimulate_DoseChanging` (M:68). Produces RCT data, not dose-switching data. |
| a1_EstimatePKPD.m | EstimatePKPD.py | 547 | 453 | partial | 4 EM methods faithfully called; full text-summary (M:342-521) dropped. |
| a2_CausalSurvivalAnalysis.m | CausalSurvivalAnalysis.py | 341 | 323 | partial | Bootstrap loop parity OK; **entire `causal_survival_results_*.txt` export (M:229-341) absent**. `fcnDiseaseModelDiagnostics` call commented out (P:186-188). Uses `fcn_bootstrapBySID_py` with unseeded `default_rng` (P:73) — not reproducible. |
| a3_ThreeTreatmentTargets.m | ThreeTreatmentTargets.py | 258 | 185 | partial | Core loop OK; **full `treatment_targets_results_*.txt` export (M:46-242) absent**. Subplot legend placement changed. |
| a4_HeatMap_Agressive.m | HeatMap_Agressive.py | 372 | 239 | partial | Single-trial grid sweep; text-file export appears abbreviated. Has `rng(0)` / `np.random.seed(0)` parity (M:4, P:10). |
| a4_OptimalTreatmentTarget.m | OptimalTreatmentTarget.py | 337 | 235 | partial | Uses `ProcessPoolExecutor` vs MATLAB `parfor` (P:9). Deterministic across workers only if per-worker seed is passed — worker body at P:98+ does seed. Text-summary block mostly missing. |
| a4_Optimize_Heatmap.m | Optimize_Heatmap.py | 413 | 242 | partial | Main sweep present; text-file export abbreviated. |
| fcnBiasedAssignmentProb.m | fcnBiasedAssignmentProb.py | 82 | 105 | appears-complete | Logic matches MATLAB line-for-line; uses `np.random.randn()` for the 0.3*randn noise — inherits global seed. |
| fcnDiseaseModelDiagnostics.m | fcnDiseaseModelDiagnostic.py | 232 | 355 | partial | Python covers most diagnostics but is not invoked by a2 (see above). Singular spelling mismatch ("Diagnostic" vs "Diagnostics") — a2 imports commented out anyway. |
| fcnEstimateDeathParms.m | fcnEstimateDeathParms.py | 29 | 97 | appears-complete | Uses `statsmodels.GLM(Binomial, logit)` which matches MATLAB `glmfit` closely; has scikit-learn fallback (P:76-97). |
| fcnEstimateKe_Standalone.m | fcnEstimateKe_Standalone.py | 337 | 377 | appears-complete | Dose-change window approach mirrors MATLAB; MAD-based robust std and weighted median faithful. Bootstrap CI uses `default_rng()` (P:141) — unseeded. |
| fcnEstimateKe_WithBiasCorrection.m | fcn_EstimateKe_WithBiasCorrection.py | 113 | 157 | appears-complete | Filename mismatch (leading underscore); function name inside file is correct. Importers use `from fcn_EstimateKe_WithBiasCorrection import fcnEstimateKe_WithBiasCorrection`. |
| fcnEstimatePKPD_FixedKe_Optimized.m | fcnEstimatePKPD_FixedKe_Optimized.py | 402 | 432 | drift | Uses `scipy.optimize.minimize` (SLSQP/L-BFGS-B) vs MATLAB `fmincon` sqp; unseeded `default_rng` for initial C/g noise (P reads similarly to v2). |
| fcnEstimatePKPD_StateSpaceMixedEffects_v2.m | fcnEstimatePKPD_StateSpaceMixedEffects_v2.py | 523 | 511 | drift | (1) Individual-params init uses unseeded `rng = np.random.default_rng()` at P:90 — breaks reproducibility set in the caller. (2) `runEKF_improved` replaces the MATLAB RTS backward smoother (M:402-415) with a 5-point moving average (P:374). (3) Inner optimizer is SLSQP not fmincon-SQP. |
| fcnEstimateParmsL.m | fcnEstimateParmsL.py | 49 | 83 | appears-complete | Row-wise extraction and >10 treated-patient guard match. |
| fcnEstimateParmsPKPD.m | fcnEstimateParmsPKPD.py | 134 | 214 | drift | Uses L-BFGS-B instead of `fmincon` interior-point (P:82). MATLAB calls stochastic `fcn_generateTrajectory` inside the NLL (M:94) generating fresh random L0 per NLL eval; Python substitutes a **deterministic** Euler integrator `_generate_L0_deterministic` (P:180-206) — **fundamentally different likelihood surface**. |
| fcnGeneratePatientParameters.m | fcnGeneratePatientParameters.py | 70 | 125 | appears-complete | Uses global `np.random.seed/rand/randn` — correctly inherits the seed set in the driver. |
| fcnGenerateStochasticTrajectories.m | fcn_generateStochasticTrajectories.py + fcnGenerateStochasticTrajectories.py | 71 | 84 + 65 | drift | **Two near-identical files** exist in `python/`. Importers variously use one or the other; inconsistent aliasing risks double-imports. |
| fcnGetPKPD_parms_est.m | fcnGetPKPD_parms_est.py | 31 | 73 | appears-complete | Reads `PKPD_estimation_results.mat`; not v7.3-aware (raises if -v7.3), which matters since MATLAB a1 saves with `-v7.3` (M:315). |
| fcnPlotKM.m | fcnPlotKM.py | 141 | 153 | appears-complete | Step interpolation via `searchsorted` equivalent to MATLAB `find(...,1,'last')`; 85-point grid preserved. |
| fcnSimulate_DoseChanging.m | fcnSimulate_DoseChanging.py | 90 | 125 | appears-complete | Matches per-patient stochastic dose changes (prob 1/5 per step, dose~U(0,5)). Uses an internal `rng` not tied to global seed (P:26) — reproducibility relies on caller passing `seed=`. |
| fcnSimulate_N_Patients.m | fcnSimulate_N_Patients.py | 132 | 212 | appears-complete | Inner `_fcnRunSimulation` faithfully mirrors PI controller, anti-windup, hazard sampling, break semantics. Uses `np.random.rand()` (global). |
| fcnSimulate_StructuredDoses.m | fcnSimulate_StructuredDoses.py | 112 | 131 | appears-complete | 4 fixed protocols cycled; schedule indexing uses ceil(j/8.5) as in MATLAB. |
| fcnSingleSwimmerPlot_v4.m | fcnSingleSwimmerPlot_v4.py | 377 | 261 | partial | Heatmap-style swimmer plot; formatting simplified but functional. |
| fcn_bootstrapBySID.m | fcn_bootstrapBySID.py | 69 | 77 | appears-complete | `np.random.choice` honors global seed; prints mirror MATLAB. |
| fcn_estimate_parmsL.m | fcn_estimate_parmsL.py | 209 | 251 | drift | MATLAB uses `fmincon` interior-point with OptimalityTolerance 1e-6 (M:78-79); Python uses L-BFGS-B ftol=1e-9 (P:112-115). Different optimizer — parameter estimates will differ. Gaussian NLL includes normalizing constant in both (parity). |
| fcn_generateTrajectory.m | fcn_generateTrajectory.py | 139 | 118 | appears-complete | Default T=170, dt=0.1 preserved. |
| fcn_kaplanMeier.m | fcn_kaplanMeier.py | 73 | 78 | appears-complete | Greenwood variance accumulator preserved; handles denom=0 case explicitly in Python. |
| run_all.m | — | 80 | — | missing | No Python driver/orchestrator for full pipeline. |
| test_FixedKe_Comprehensive.m | — | 404 | — | missing | Methods-comparison test script; fine to leave unported (integration test). |

### 1b. `CICADA_FIGURES/` — publication figures

| MATLAB file | Python file | M LOC | Py LOC | Status | Produces |
|---|---|---:|---:|---|---|
| a1_SingleTraces.m | — | 116 | — | missing | Fig1_singleTrajectories_3panels.pdf |
| a2_EvaluatePKPD_estimates_figures.m | — | 298 | — | missing | Fig_Combined_PKPD_Analysis.pdf |
| a2_EvaluatePKPD_estimates_text.m | — | 199 | — | missing | PKPD text summary |
| a3a_Fig_Swimmers_RCT.m | — | 347 | — | missing | Fig3_swimmer_survival_plot_RCT.pdf |
| a3b_Fig_Swimmers_Obs_Naive.m | — | 520 | — | missing | Fig4_swimmer_survival_plot_Obs_Naive.pdf |
| a3c_Fig_Swimmers_Obs_g_formula.m | — | 369 | — | missing | Fig_gformula_corrected_survival_curves.pdf |
| a4_HeatMaps_Aggressive_Figs.m | — | 71 | — | missing | heatmap PDF |
| a4_HeatMaps_Combined.m | — | 111 | — | missing | Fig_heatmap_figure.pdf |
| a5_OptimizationCurve.m | — | 611 | — | missing | Fig_optimization_curves_with_survival.pdf |
| fcnDualSwimmerPlot_v2.m | — | 536 | — | missing | helper |
| fcnPlotSurvivalCurves_CIs.m | — | 81 | — | missing | helper |
| fcnSingleSwimmerPlot.m | — | 497 | — | missing | helper (v1) |
| fcnSingleSwimmerPlot_v2.m | — | 448 | — | missing | helper |
| fcnSingleSwimmerPlot_v3.m | — | 431 | — | missing | helper |
| fcnSingleSwimmerPlot_v4.m | (same as `matlab/`) | 380 | 261 | appears-complete | helper — ported in `python/fcnSingleSwimmerPlot_v4.py` |
| plot_survival_curves_only.m | — | 105 | — | missing | diagnostic plot |
| run_all_figures.m | — | 41 | — | missing | figure-orchestrator |

### 1c. `sensitivity/`

| MATLAB file | Python file | M LOC | Py LOC | Status | Produces |
|---|---|---:|---:|---|---|
| alt_censoring/run_alt_censoring.m | — | 118 | — | missing | alt_censoring_results.mat, Fig_alt_censoring.pdf, alt_censoring_summary.txt |
| measurement_error/run_measurement_error.m | — | 101 | — | missing | measurement_error_results.mat, Fig_measurement_error.pdf, summary.txt |
| nuc_injection/run_nuc_sensitivity.m | — | 246 | — | missing | nuc_results.mat, Fig_NUC_sensitivity.pdf, nuc_summary.txt |
| nuc_injection/fcnSimulate_N_Patients_withU.m | — | 94 | — | missing | U-confounder simulator helper |

---

## 2. Critical gaps (all-missing)

Missing entirely from Python port (all `CICADA_FIGURES/` + all `sensitivity/`):

- **10 figure scripts** in `CICADA_FIGURES/` (~2,918 LOC total). Each writes one PDF; `a5_OptimizationCurve.m` (611 LOC) and `a3b_Fig_Swimmers_Obs_Naive.m` (520 LOC) are the heaviest.
- **4 swimmer-plot helpers** in `CICADA_FIGURES/` (`fcnDualSwimmerPlot_v2`, `fcnSingleSwimmerPlot` v1/v2/v3; ~1,912 LOC). Only `v4` is ported (under `python/fcnSingleSwimmerPlot_v4.py`).
- **1 non-figure CI helper** `fcnPlotSurvivalCurves_CIs.m` (81 LOC).
- **3 sensitivity drivers** + 1 sensitivity helper (~559 LOC). Outputs each: one .mat + one PDF + one summary .txt.
- **`matlab/test_FixedKe_Comprehensive.m`** (404 LOC) — integration test.
- **`matlab/run_all.m`** — full-pipeline orchestrator.

Expected total missing-Python LOC: ~6,000 MATLAB lines.

---

## 3. Drift risks (structural divergence)

### a0_GenerateTrialData → generatetrialdata.py (HIGH)
- P:315-329 — the entire g-formula block is commented out, and P:343-349 skips the g-formula summary.  Python only plots RCT vs Naive.  **The PDF output will differ visibly** from MATLAB `Fig_swimmer_survival_plot_Obs_Naive.pdf` because green g-formula curves are never drawn.
- P:34-35 — optional helper imports commented; swimmer plot is skipped.
- No `trial_simulation_results_*.txt` output.

### a0_GenerateDoseSwitchingData → generate_trial_data_dose_changing.py (CRITICAL)
- P:73-82 — calls `fcnSimulate_N_Patients(RCT=1, treatProb=0.5*ones(N))`. MATLAB calls **`fcnSimulate_DoseChanging`** (M:68). Consequence: `trialDataDoseChanging.csv` produced by Python contains **RCT PI-controlled data**, not random-dose-changing data. This totally breaks the downstream ke-estimation identifiability story (a1) because windows around abrupt dose changes (needed for `fcnEstimateKe_Standalone`) don't exist.

### a1_EstimatePKPD → EstimatePKPD.py (MEDIUM)
- Main 4-method EM comparison is faithful.
- **Missing**: L312-521 of MATLAB (detailed `pkpd_estimation_results_*.txt` summary). Only the 3-line terminal summary is produced.
- Python saves with `scipy.io.savemat(..., do_compression=True)` (MATLAB v5), but the downstream `fcnGetPKPD_parms_est.py` (P:26) reads with `loadmat(..., struct_as_record=False)`, which works for v5 but not v7.3 — MATLAB saves as `-v7.3` (M:315), so round-tripping MATLAB-produced results through Python requires hdf5storage or Python re-running a1 first.

### a2_CausalSurvivalAnalysis → CausalSurvivalAnalysis.py (HIGH)
- P:207 — uses inline `fcn_bootstrapBySID_py` (not the imported `fcn_bootstrapBySID`); the inline version uses a fresh unseeded `default_rng` (P:73). **Bootstrap is not reproducible**, even within a single Python run, and is disconnected from `np.random.seed(0)` calls elsewhere.
- Python imports `fcnPlotKM` and `fcn_bootstrapBySID` (P:93-94) yet **shadows them** with local definitions (P:55-82) — the local versions win.
- P:186-188 — `fcnDiseaseModelDiagnostics` call is commented out; a2's diagnostic plots never run.
- Missing entirely: the `causal_survival_results_*.txt` output (MATLAB M:229-341, ~113 lines of `fprintf`).

### a3_ThreeTreatmentTargets → ThreeTreatmentTargets.py (MEDIUM)
- Core loop is faithful.
- Missing `treatment_targets_results_*.txt` export (MATLAB M:46-242).
- P:137-141 `_pad_to_nt` pads/truncates KM curves to 85 — may mis-align with the MATLAB fixed-grid step interpolation if a patient's last event is strictly before 168 (Python uses NaN padding, MATLAB extends last value). Treat end-of-trial ATE with care.

### a4_HeatMap_Agressive / a4_Optimize_Heatmap / a4_OptimalTreatmentTarget (MEDIUM)
- `rng(0)` parity in P:10-14 of HeatMap_Agressive and Optimize_Heatmap.
- `OptimalTreatmentTarget.py` parallelizes via `ProcessPoolExecutor` (P:9), not MATLAB `parfor`. Seeds are passed per-worker so reproducibility is preserved within the Python side, but will differ numerically from MATLAB's parfor stream (already unavoidable).
- Text-file summary exports abbreviated or missing relative to MATLAB.

### fcnEstimatePKPD_StateSpaceMixedEffects_v2 → .py (HIGH)
- **Unseeded RNG inside the estimator** (P:90: `rng = np.random.default_rng()`). The MATLAB version uses `randn(N,1)` which inherits the global seed. This makes the E-step initialization non-deterministic even when the caller seeds.
- **RTS backward smoother replaced by moving average** (P:374: `_smooth_1d(L0_filt, window=5)`). MATLAB M:402-415 implements the proper RTS gain `A_smooth = P_filt(t) / (P_filt(t) + Q)` and updates both state and covariance. Python's moving average is biased toward the mean and throws away the Kalman uncertainty. L0 estimates will systematically differ.
- Optimizer: SLSQP (P:393) vs fmincon-SQP. Different convergence basin near boundaries.

### fcnEstimatePKPD_FixedKe_Optimized.py (MEDIUM)
- Same RNG + optimizer issues as above (shares helpers).

### fcnEstimateParmsPKPD.py (HIGH)
- P:180-206 `_generate_L0_deterministic` — MATLAB M:94 calls **stochastic** `fcn_generateTrajectory` (with `randn` inside). Replacing with deterministic Euler changes the objective landscape and the returned ke/C/g coefficients. This helper is used by `a2_CausalSurvivalAnalysis.m` (M:17 chain via `fcnEstimateParmsL`+`fcnEstimateParmsPKPD`), so `parmsPD_est` can differ substantially.

### fcn_estimate_parmsL.py (MEDIUM)
- L-BFGS-B with ftol=1e-9 (P:112-115) vs MATLAB `fmincon` interior-point with OptimalityTolerance=1e-6 and StepTolerance=1e-6. Different optimizer class, different tolerances. Parameter estimates will differ at the 3rd decimal at best.

### Duplicate trajectory generators
- `fcn_generateStochasticTrajectories.py` (85 LOC) AND `fcnGenerateStochasticTrajectories.py` (65 LOC) both define the same symbol. `generatetrialdata.py` imports the CamelCase module; `a1`/`a2`/`ThreeTreatmentTargets` import the underscore module. They're redundant. Pick one and delete the other.

---

## 4. RNG / optimizer / numerical hazards

### RNG seeds

| Location | MATLAB seed | Python seed | Verdict |
|---|---|---|---|
| `a0_GenerateTrialData` | `rng(0)` (M:14) | `np.random.seed(0)` (P:22) | parity (global). Note: PCG64 ≠ MT19937 so streams diverge. |
| `a0_GenerateDoseSwitchingData` | commented out (M:19) | `np.random.seed(0)` (P:19) | Python is stricter. |
| `a4_HeatMap_Agressive` | `rng(0)` (M:4) | `np.random.seed(0)` (P:10) | parity |
| `a4_Optimize_Heatmap` | `rng(0)` (M:2) | `np.random.seed(0)` (P:14) | parity |
| `a4_OptimalTreatmentTarget` | no explicit seed; `parfor` streams | `np.random.seed(0)` + per-worker seeds | Python stricter. |
| `fcnGeneratePatientParameters` | optional `RandomSeed` name/value | same | parity |
| `fcnEstimatePKPD_StateSpaceMixedEffects_v2` | uses global seed | **local `default_rng()`** at P:90 | **BREAKS reproducibility** |
| `fcnEstimatePKPD_FixedKe_Optimized` | uses global seed | same pattern likely (shared idiom) | likely breaks |
| `fcnEstimateKe_Standalone` bootstrap CI | `randsample` (global) | `default_rng()` at P:141 | breaks |
| `a2_CausalSurvivalAnalysis` bootstrap | `randi` (global) | inline `fcn_bootstrapBySID_py` uses `default_rng()` at P:73 | **breaks** — main numbers unreproducible |
| `fcnSimulate_DoseChanging.py` | global `rand` | local `default_rng(seed)` at P:26 — caller must pass seed | breaks unless caller threads seed |

### Optimizers / ODE / filter primitives

| MATLAB call | Location | Python equivalent | Tolerance match |
|---|---|---|---|
| `fmincon` (SQP, for C/g per-patient) | `fcnEstimatePKPD_StateSpaceMixedEffects_v2.m:437-439` | `scipy.optimize.minimize` method=SLSQP | P uses maxiter=150, ftol=1e-9; M uses default OptimalityTol ~1e-6. Similar but not identical. |
| `fmincon` (for ke) | same file M:488 | `minimize` SLSQP, bounds=[(0.1,1.0)] | P:447-450 |
| `fmincon` (interior-point, parmsL) | `fcn_estimate_parmsL.m:78-82` | `minimize` L-BFGS-B, ftol=1e-9 | **different algorithm family** |
| `fmincon` (PD regression) | `fcnEstimateParmsPKPD.m:41-43` | `minimize` L-BFGS-B | **different algorithm family** |
| EKF forward + **RTS backward smoother** | `fcnEstimatePKPD_StateSpaceMixedEffects_v2.m:352-415` | Python does forward EKF then **5-pt moving average** (P:374) | **not equivalent** |
| EM outer loop (30 iters, momentum=0.7, shrink=0.3) | v2 M:77-197 | P:107-194 | parity |
| `glmfit(..., 'binomial','logit')` | `fcnEstimateDeathParms.m:27` | `statsmodels.GLM` binomial-logit | parity |
| `kstest2` | `fcn_estimate_parmsL.m:107-108` | `scipy.stats.ks_2samp` | parity |
| `polyfit` | several | `np.polyfit` | parity |
| `prctile` / `percentile` | many | `np.percentile` | parity (mild method difference but negligible) |
| `randsample` bootstrap | `fcnEstimateKe_Standalone.m:105` | `rng.choice(..., replace=True)` | parity in semantics, stream differs |

No `ode45`, `lsqnonlin`, `integral` appear in either codebase — disease and PK dynamics are Euler-Maruyama / discrete-time, good.

### EM / Kalman specifically
- Python's `fcnEstimatePKPD_StateSpaceMixedEffects_v2.runEKF_improved` (P:316-376) drops the RTS smoother pass present in MATLAB lines 402-415.  Replacement: 5-point centered moving average on the forward-filtered L0.  Covariance `P_smooth` returned to caller but caller discards it.  For the EM M-step the two look the same, but the smoothed L0 that feeds the M-step will differ systematically.  This is the single largest source of expected parameter drift for the paper's headline PKPD numbers.

---

## 5. I/O and output artifacts

### CSV / data inputs

| Artifact | Produced by (MATLAB) | Produced by (Python) | Match? |
|---|---|---|---|
| `trialData0.csv` | a0_GenerateTrialData.m | generatetrialdata.py | yes |
| `trialData1.csv` | a0_GenerateTrialData.m | generatetrialdata.py | yes |
| `trialDataDoseChanging.csv` | a0_GenerateDoseSwitchingData.m | generate_trial_data_dose_changing.py | **NO** — content is different (RCT data vs dose-switching data). |
| `trialDataStructured.csv` | (referenced but no script in matlab/?) | — | not produced in either place from these scripts |

### .mat files

| Artifact | MATLAB script | Python script | Match? |
|---|---|---|---|
| `parmsTrue.mat` | a0_GenerateTrialData.m L75 | generatetrialdata.py L176-191 | yes (-v7.3 in MATLAB, v5 in Python) |
| `parmsTrue_DoseChanging.mat` | a0_GenerateDoseSwitchingData.m L77 | generate_trial_data_dose_changing.py L92-107 | content differs (upstream sim is wrong) |
| `parmsTrue_Structured.mat` | — | — | neither produces |
| `L0data.mat` | a0_GenerateDoseSwitchingData.m L74 | generate_trial_data_dose_changing.py L89 | yes |
| `L0data_structured.mat` | — | — | neither produces |
| `PKPD_estimation_results.mat` | a1_EstimatePKPD.m L315 | EstimatePKPD.py L424 | yes (v5 vs v7.3 format mismatch) |
| `EstimatedParameters.mat` | a2_CausalSurvivalAnalysis.m L36 | CausalSurvivalAnalysis.py L146-148 | yes |
| `ThreeCurves.mat` | a3_ThreeTreatmentTargets.m L44 | ThreeTreatmentTargets.py L155-169 | yes |
| `HeatMapAggressive.mat` | a4_HeatMap_Agressive.m | HeatMap_Agressive.py | yes (assumed, script writes savemat) |
| `HeatMapData.mat` | a4_Optimize_Heatmap.m | Optimize_Heatmap.py | yes (assumed) |
| `bootstrap_confidence_bands.mat` | a2_CausalSurvivalAnalysis.m L224-225 | CausalSurvivalAnalysis.py L306-321 | yes |
| `bootstrap_confidence_bands_v2.mat` | a2_CausalSurvivalAnalysis.m L106 | — | **Python does not produce** |
| `A01Data.mat` | unknown source | — | — |

### Text summaries

| .txt file | MATLAB | Python | Status |
|---|---|---|---|
| `trial_simulation_results_*.txt` | a0 M:354-477 | — | **missing in Python** |
| `pkpd_estimation_results_*.txt` | a1 M:343-521 | — | **missing in Python** |
| `causal_survival_results_*.txt` | a2 M:230-341 | — | **missing in Python** |
| `treatment_targets_results_*.txt` | a3 M:47-242 | — | **missing in Python** |
| `heatmap_sensitivity_results_*.txt` | a4_HeatMap_Agressive | abbreviated? | partial |
| `optimization_heatmap_results_*.txt` | a4_Optimize_Heatmap | abbreviated? | partial |
| `optimal_treatment_target_results_*.txt` | a4_OptimalTreatmentTarget | abbreviated | partial |
| sensitivity `*_summary.txt` (3×) | sensitivity scripts | — | **missing** |

### PDF figures

All 10 publication PDFs under `CICADA_FIGURES/` and all 3 sensitivity PDFs are **not produced** by the Python port.  Only `Fig_swimmer_survival_plot_Obs_Naive.pdf` is produced by `generatetrialdata.py` (with g-formula curves missing).

---

## 6. Recommended Phase 3 port order

Dependency-ordered punch list.  S/M/L/XL estimates assume working knowledge of the MATLAB sources.

1. **Fix `generate_trial_data_dose_changing.py`** — swap in `fcnSimulate_DoseChanging` call. Without this, a1 PKPD identifiability is broken.  **Size: S** (1-line logic fix + re-test).  Blocks a1.
2. **Consolidate trajectory generators** — remove `fcnGenerateStochasticTrajectories.py` (or the other); update imports.  **S.** Independent.
3. **Seed all local RNGs consistently** — in v2 estimator (P:90), in fixed-ke estimator, in `fcn_bootstrapBySID_py` (inline in a2), in `fcnEstimateKe_Standalone` bootstrap CI, in `fcnSimulate_DoseChanging`.  Replace `np.random.default_rng()` with `np.random` (global) or thread a seed argument.  **S-M.** Blocks reproducible validation.
4. **Restore RTS smoother in `fcnEstimatePKPD_StateSpaceMixedEffects_v2.runEKF_improved`** — port MATLAB lines 402-415 verbatim.  **M.** Blocks PKPD numerical parity (headline numbers).
5. **Fix `fcnEstimateParmsPKPD._generate_L0_deterministic`** — should call the stochastic trajectory generator (with a seed, possibly multi-sample mean) to match MATLAB M:94. Consider averaging over N_sim=5 stochastic draws per NLL eval.  **M.** Affects a2 numbers.
6. **Consider swapping SLSQP/L-BFGS-B → `trust-constr` or `SLSQP` with tighter tol** in `fcn_estimate_parmsL` and `fcnEstimateParmsPKPD` to approximate `fmincon` interior-point better. Measure drift first.  **S.** Independent.
7. **Restore text-file exports** in `generatetrialdata.py`, `EstimatePKPD.py`, `CausalSurvivalAnalysis.py`, `ThreeTreatmentTargets.py`, a4_*.py.  Copy MATLAB fprintf blocks 1:1.  **M per file, L total.** Independent.
8. **Restore g-formula curves** in `generatetrialdata.py` (uncomment P:315-329, call estimators).  **S.** Depends on 1-4.
9. **Port swimmer-plot helpers**: v1/v2/v3 from `CICADA_FIGURES/` to `python/` (or just one if v4 suffices for paper).  **L** (swimmer v4 is 377 MATLAB lines, ports to 261 Python lines; v1/v2/v3 each ~400-500 MATLAB lines). Independent.
10. **Port 10 figure scripts** `CICADA_FIGURES/a1_SingleTraces.m`, `a2_EvaluatePKPD_estimates_figures.m`, `a2_EvaluatePKPD_estimates_text.m`, `a3a/b/c_Fig_Swimmers_*`, `a4_HeatMaps_Aggressive_Figs.m`, `a4_HeatMaps_Combined.m`, `a5_OptimizationCurve.m`, `plot_survival_curves_only.m`.  Sizes from S (a4_HeatMaps_Aggressive_Figs 71 LOC) to **XL** (`a5_OptimizationCurve` 611 LOC, `a3b` 520 LOC). Depend on 9.
11. **Port 3 sensitivity drivers** `run_alt_censoring`, `run_measurement_error`, `run_nuc_sensitivity` + `fcnSimulate_N_Patients_withU`.  Each is self-contained once the core estimators are correct.  **M each, L total.** Depends on 1-5.
12. **Port `run_all.m` orchestrator** as a Python `run_all.py` driver.  **S.** Last.

Critical path: 1 → 4 → 5 → 2-3 → 7 → 9 → 10-11 → 12.

---

## 7. Notes on `benchmarks/` and `sensitivity/` scope

### `benchmarks/run_benchmarks.py` (312 LOC)

Implements Naive KM + IPTW (stabilized) + MSM (pooled logistic on discrete-time hazard). Reads `trialData0.csv`, `trialData1.csv`, and `bootstrap_confidence_bands.mat` from the repo root. Writes `benchmarks/benchmark_results.json` and `benchmarks/Fig_benchmark_forest.pdf`.  Parity-test candidate: already production-ready **conditional on** upstream CSVs being correct (see item 1 above — currently `trialDataDoseChanging.csv` is wrong but benchmarks only read `trialData{0,1}.csv` which are correct).  TMLE is deferred in the header comment — that's fine per revision scope ("IPTW+MSM only").

### `sensitivity/` Python scope

- All 3 MATLAB sensitivity scripts (`alt_censoring`, `measurement_error`, `nuc_injection`) call `addpath(genpath(repo_root))` to reuse `fcn*` helpers.  A Python port would need to import from `python/` analogously.
- `run_alt_censoring.m` needs only the core simulator/estimators + 4 different parmsV variants — straightforward once core is fixed.
- `run_measurement_error.m` just perturbs `trialData0.csv` L column and re-runs the estimation pipeline — trivial once g-formula works in Python.
- `run_nuc_sensitivity.m` + `fcnSimulate_N_Patients_withU.m` needs a modified simulator that accepts a hidden U covariate — a straightforward extension of `fcnSimulate_N_Patients.py`.  Plus E-value computation.
- The three expected artifacts per script: `*_results.mat`, `Fig_*.pdf`, `*_summary.txt`.
