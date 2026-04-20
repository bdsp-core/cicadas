# Figure map — CICADAS manuscript figures

Each manuscript figure is produced by a single MATLAB script. The scripts read data from the main-level `.mat` and `.csv` files (produced by `a0`–`a4`) and save a PDF next to themselves.

| Manuscript figure (filename)                                   | Script (path)                                                                      | Depends on                                                      |
|----------------------------------------------------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `Fig1_singleTrajectories_3panels.pdf`                          | [`CICADA_FIGURES/a1_SingleTraces.m`](../CICADA_FIGURES/a1_SingleTraces.m)          | `trialData1.csv`, `trialDataDoseChanging.csv` (produced by `matlab/a0_GenerateTrialData.m` + `matlab/a0_GenerateDoseSwitchingData.m`) |
| `Fig_Combined_PKPD_Analysis.pdf`                               | [`CICADA_FIGURES/a2_EvaluatePKPD_estimates_figures.m`](../CICADA_FIGURES/a2_EvaluatePKPD_estimates_figures.m) | `PKPD_estimation_results.mat`                                   |
| `Fig3_swimmer_survival_plot_RCT.pdf`                           | [`CICADA_FIGURES/a3a_Fig_Swimmers_RCT.m`](../CICADA_FIGURES/a3a_Fig_Swimmers_RCT.m)   | `trialData1.csv`, `bootstrap_confidence_bands.mat`              |
| `Fig4_swimmer_survival_plot_Obs_Naive.pdf`                     | [`CICADA_FIGURES/a3b_Fig_Swimmers_Obs_Naive.m`](../CICADA_FIGURES/a3b_Fig_Swimmers_Obs_Naive.m) | `trialData0.csv`, `trialData1.csv`                             |
| `Fig_gformula_corrected_survival_curves.pdf`                   | [`CICADA_FIGURES/a3c_Fig_Swimmers_Obs_g_formula.m`](../CICADA_FIGURES/a3c_Fig_Swimmers_Obs_g_formula.m) | `trialData0.csv`, `trialData1.csv`, `bootstrap_confidence_bands.mat` |
| `Fig_heatmap_figure.pdf`                                       | [`CICADA_FIGURES/a4_HeatMaps_Combined.m`](../CICADA_FIGURES/a4_HeatMaps_Combined.m)  | `HeatMapAggressive.mat`, `HeatMapData.mat`                      |
| `Fig_optimization_curves_with_survival.pdf`                    | [`CICADA_FIGURES/a5_OptimizationCurve.m`](../CICADA_FIGURES/a5_OptimizationCurve.m) | `A01Data.mat`, `ThreeCurves.mat`                                |

## Rebuilding all figures at once

From within `CICADA_FIGURES/`:

```matlab
run_all_figures
```

Or from the repo root as part of the full pipeline:

```matlab
run_all
```

## Paper parity

Under MATLAB R2024b (the version used for the submitted manuscript), cold runs from `rng(0)` should reproduce all figures byte-identically. Under R2025b, numerical drift in `fmincon` and the EKF/RTS/EM pipeline causes `a0_GenerateDoseSwitchingData.m` to diverge, which propagates into `Fig_Combined_PKPD_Analysis.pdf` (Joint-method bar heights differ). All other figures reproduce at pixel parity.
