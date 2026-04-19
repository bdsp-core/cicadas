# CICADAS — Python port

This is a Python port of the MATLAB implementation. **The MATLAB implementation in the repo root is authoritative** — it is what generated the figures in the paper. This port exists for readers without MATLAB access and is the development surface for the revision's new sensitivity and benchmarking analyses.

## Parity

Numerical results should be close to the MATLAB versions but are **not guaranteed to be bit-identical** because of differences in RNG streams, optimizer behavior (`scipy.optimize.minimize trust-constr` vs MATLAB `fmincon`), and EKF/RTS/EM numerics.

## Scripts

Parallel to the MATLAB `a0` / `a1` / `a2` / ... naming:

- `generatetrialdata.py`                     ↔ `a0_GenerateTrialData.m`
- `generate_trial_data_dose_changing.py`     ↔ `a0_GenerateDoseSwitchingData.m`
- `EstimatePKPD.py`                          ↔ `a1_EstimatePKPD.m`
- `CausalSurvivalAnalysis.py`                ↔ `a2_CausalSurvivalAnalysis.m`
- `ThreeTreatmentTargets.py`                 ↔ `a3_ThreeTreatmentTargets.m`
- `HeatMap_Agressive.py`                     ↔ `a4_HeatMap_Agressive.m`
- `OptimalTreatmentTarget.py`                ↔ `a4_OptimalTreatmentTarget.m`
- `Optimize_Heatmap.py`                      ↔ `a4_Optimize_Heatmap.m`
- `fcn*.py`                                  ↔ `fcn*.m` helpers

## Requirements

Python ≥ 3.10.

```
numpy
scipy
pandas
matplotlib
statsmodels
lifelines
```

Install with `pip install -r requirements.txt` once generated, or `pip install numpy scipy pandas matplotlib statsmodels lifelines` in a fresh virtualenv.

## Status (2026-04-19)

This port runs end-to-end but has not been pixel-for-pixel parity-tested against the MATLAB output. Use with that caveat. A full parity study is planned for post-submission.
