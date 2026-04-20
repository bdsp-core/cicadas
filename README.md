# CICADAS

**CICADAS** (Causal Inference for Critical-Care Anti-Seizure Treatment with Disease Dynamics, Automated PKPD/Trial Simulation, and Survival Optimization) is a simulation-based framework for emulating randomized trials of anti-seizure treatment in critically ill ICU patients, combining the parametric g-formula with mechanistic pharmacokinetic–pharmacodynamic models and disease-dynamics simulations.

This repository accompanies:

> McCauley, Westover T., Wynn, Sartipi, et al. *CICADAS: A Simulation-Based Framework for Randomized Trial Emulation in Critical-Care Seizure Treatment.* iScience, in revision, 2026.

## Repository layout

```
cicadas/
├── matlab/                           # MATLAB analysis pipeline (authoritative, used for paper figures)
│   ├── a0_GenerateTrialData.m        # Generate RCT + observational simulated cohorts
│   ├── a0_GenerateDoseSwitchingData.m# Generate dose-switching cohort (for PKPD estimation)
│   ├── a1_EstimatePKPD.m             # Estimate PKPD parameters (EKF/RTS/EM, two-stage w/ bias correction)
│   ├── a2_CausalSurvivalAnalysis.m   # g-formula causal survival analysis + bootstrap CIs
│   ├── a3_ThreeTreatmentTargets.m    # Compare three treatment targets
│   ├── a4_HeatMap_Agressive.m        # Heterogeneous-effects heatmap (aggressive scenario)
│   ├── a4_OptimalTreatmentTarget.m   # Personalized optimal threshold (parallel bootstrap)
│   ├── a4_Optimize_Heatmap.m         # Personalized optimization heatmap
│   ├── fcn*.m                        # Helper functions (PKPD, disease dynamics, plotting, bootstrap)
│   └── run_all.m                     # Orchestrates the full a0 -> a4 -> figures pipeline
├── CICADA_FIGURES/                   # Figure-generation scripts (one script per manuscript figure)
│   ├── a1_SingleTraces.m             # -> Fig1_singleTrajectories_3panels.pdf
│   ├── a2_EvaluatePKPD_estimates_figures.m   # -> Fig_Combined_PKPD_Analysis.pdf
│   ├── a3a_Fig_Swimmers_RCT.m        # -> Fig3_swimmer_survival_plot_RCT.pdf
│   ├── a3b_Fig_Swimmers_Obs_Naive.m  # -> Fig4_swimmer_survival_plot_Obs_Naive.pdf
│   ├── a3c_Fig_Swimmers_Obs_g_formula.m  # -> Fig_gformula_corrected_survival_curves.pdf
│   ├── a4_HeatMaps_Combined.m        # -> Fig_heatmap_figure.pdf
│   ├── a5_OptimizationCurve.m        # -> Fig_optimization_curves_with_survival.pdf
│   └── run_all_figures.m             # Runs all seven figure scripts
├── python/                           # Python port (work in progress; MATLAB is authoritative)
├── sensitivity/                      # Sensitivity analyses (unmeasured confounding, measurement error, alt censoring)
├── benchmarks/                       # IPTW / MSM benchmark implementations (Python)
├── docs/
│   └── figure_map.md                 # Script → manuscript-figure mapping
├── parmsTrue.mat                     # Ground-truth simulation parameters (tracked)
├── LICENSE                           # MIT
├── CITATION.cff                      # Citation metadata
└── README.md
```

## Requirements

- **MATLAB R2024b or R2025b.** The paper was generated under R2024b; R2025b produces qualitatively identical but byte-divergent results due to minor floating-point drift in `fmincon` and the EKF. For byte-level reproducibility, pin R2024b.
  - Optimization Toolbox (`fmincon`)
  - Statistics and Machine Learning Toolbox (`pooled logistic regression`, `bootstrap`)
  - Parallel Computing Toolbox (`parfor` is used by `a4_OptimalTreatmentTarget.m`)
- **Python ≥3.10** for the Python port and for the sensitivity/benchmarking analyses added in the revision. See `python/requirements.txt`.

## Quickstart (MATLAB)

From the repository root, add all subdirectories to the MATLAB path and
then run the orchestrator:

```matlab
addpath(genpath(pwd));

% Full end-to-end pipeline (includes a0 -> a4 -> figure generation):
run('matlab/run_all.m')
```

Or step through manually (same path setup):

```matlab
addpath(genpath(pwd));

% 1. Generate simulated cohorts
a0_GenerateTrialData                 % ~25 s  -> trialData0.csv, trialData1.csv, parmsTrue.mat
a0_GenerateDoseSwitchingData         % ~90 s  -> trialDataDoseChanging.csv

% 2. Estimate PKPD parameters
a1_EstimatePKPD                      % ~85 s  -> PKPD_estimation_results.mat

% 3. g-formula causal survival analysis with bootstrap CIs (1000 replicates)
a2_CausalSurvivalAnalysis            % ~40 min -> bootstrap_confidence_bands.mat

% 4. Treatment-target analyses
a3_ThreeTreatmentTargets             % ~15 s  -> ThreeCurves.mat
a4_HeatMap_Agressive                 % ~3 min -> HeatMapAggressive.mat
a4_OptimalTreatmentTarget            % ~1 h   -> A01Data.mat       (parallel)
a4_Optimize_Heatmap                  % ~45 min -> HeatMapData.mat

% 5. Regenerate figures
cd CICADA_FIGURES
run_all_figures
```

Expected paper results (from `a0` and `a2`):

| Quantity | Paper | Reproduced (R2025b) |
|---|---|---|
| RCT ATE at 168h | +14.8% | +14.8% |
| Naive Kaplan–Meier ATE | −4.8% | −4.8% |
| g-formula ATE | +14.2% | +14.5% |
| Treated survival at 168h | 46.3% | 46.6% |
| Untreated survival at 168h | 32.1% | 32.2% |

## Reproducibility notes

- **RNG seeding.** `a0_GenerateTrialData.m` calls `rng(0)` and produces byte-identical CSVs across runs. Other scripts inherit MATLAB's default RNG state; cold runs under the same MATLAB version reproduce exactly.
- **Bootstrap replicates.** Both `a2_CausalSurvivalAnalysis.m` and `a4_OptimalTreatmentTarget.m` use `Nboot = 1000`.
- **Paper figures.** The authoritative figures in the manuscript were generated under MATLAB R2024b. Regeneration under R2025b produces visually equivalent figures; one figure (`Fig_Combined_PKPD_Analysis`) shows visible drift in Joint-method estimation errors due to floating-point drift in `a0_GenerateDoseSwitchingData.m`.

## License

MIT (see [LICENSE](LICENSE)).

## Citation

See [CITATION.cff](CITATION.cff).

## Contact

M. Brandon Westover, MD, PhD — [mwestove@bidmc.harvard.edu](mailto:mwestove@bidmc.harvard.edu)
Brain Data Science Platform, Beth Israel Deaconess Medical Center / Harvard Medical School
