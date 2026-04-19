# Causal Survival Analysis with PKPD Modeling - Function Documentation

This repository contains MATLAB scripts and functions for simulating and analyzing clinical trials using causal survival analysis with pharmacokinetic-pharmacodynamic (PKPD) modeling.

## Main Scripts

### Data Generation and Analysis
- **a0_GenerateTrialData.m**: Generates synthetic clinical trial data using logit-based discrete-time hazard models. Creates both RCT (randomized controlled trial) and observational study datasets with PKPD dynamics.

- **a1_CausalSurvivalAnalysis.m**: Main analysis script that estimates disease progression and mortality models from observational data, then simulates causal survival curves under different treatment scenarios.

### Sensitivity Analysis
- **a3_Trial_Given_th.m**: Performs sensitivity analysis by varying the treatment threshold (th) and patient harm parameters (harmE, harmA) to compute average treatment effects (ATE).

- **a4_prob_contours.m**: Creates contour plots showing probability surfaces as a function of harmonic parameters E and A.

- **a5_prob_contours_age_sofa.m**: Similar to a4 but specifically for age and SOFA score dependencies.

### Visualization
- **a0_Figures.m**: Basic figure generation script (appears to be an earlier version).

- **a6_Figures.m**: Creates disease trajectory visualizations comparing treated vs untreated patients.

## Core Functions

### Patient and Trial Simulation
- **fcnGeneratePatientParameters.m**: Generates patient-specific parameters including age, SOFA scores, and PD parameters (C and g values).

- **fcnSimulate_N_Patients.m**: Simulates disease progression and outcomes for N patients under specified trial conditions.

- **fcnRunSimulation.m**: Core simulation engine that generates individual patient trajectories.

- **fcn_generateTrajectory.m**: Generates a single patient's disease trajectory over time.

- **fcn_generateStochasticTrajectories.m**: Creates stochastic disease trajectories with noise components.

### Treatment Assignment
- **fcnBiasedAssignmentProb.m**: Calculates biased treatment assignment probabilities for observational studies based on patient characteristics.

- **fcn_Trial_th.m**: Runs a trial simulation with a specific treatment threshold, returning data for both treatment arms.

### Parameter Estimation
- **fcnEstimateDeathAndDisease.m**: Master estimation function that calls other estimation routines to fit both mortality and disease progression models.

- **fcnEstimateDeathModel.m**: Estimates mortality hazard model parameters from observed data.

- **fcn_estimateParametersL.m**: Estimates disease progression (L) model parameters.

- **fcn_estimateParametersLandPD_TwoStage_AgeSofa.m**: Two-stage estimation procedure for disease and PD parameters with age/SOFA dependencies.

### Utility Functions
- **fcn_harmMapping.m**: Maps harmonic parameters to patient-specific values considering age and SOFA distributions.

- **fcnPlotKM.m**: Plots Kaplan-Meier survival curves and returns survival probabilities.

- **fcn_kaplanMeier.m**: Core Kaplan-Meier estimation function.

- **fcnDualSwimmerPlot.m**: Creates swimmer plots showing individual patient trajectories.

- **fcnDiseaseModelDiagnostics.m**: Generates diagnostic plots to assess model fit quality.

## Data Files

- **parmsTrue.mat**: Contains true parameter values used for data generation.
- **ground_truth_km_logit.mat**: Ground truth Kaplan-Meier curves for validation.
- **trialData0_logit.csv**: Observational study data (with treatment selection bias).
- **trialData1_logit.csv**: RCT data (randomized treatment assignment).

---

## Variable Naming Harmonization Suggestions

### 1. **Standardize Parameter Naming**
   - Current: Mixed use of `parms`, `params`, `parameters`
   - Suggestion: Use consistent prefix `params_` followed by model name
     - `params_Y` (mortality model)
     - `params_V` (censoring model)
     - `params_L` (disease model)
     - `params_PD` (pharmacodynamic model)
     - `params_control` (controller parameters)

### 2. **Clarify Time Variables**
   - Current: `t`, `T`, `Nt`, `dt` used inconsistently
   - Suggestion:
     - `time_points`: vector of time points
     - `dt`: time step (consistent)
     - `n_timepoints`: number of time points
     - `T_data`: table/dataset (capital T for tables only)

### 3. **Patient-Related Variables**
   - Current: Mixed use of `N`, `patient_age`, `age`, `sofa`, `patient_sofa`
   - Suggestion:
     - `n_patients`: number of patients
     - `patient.age`: structure containing patient characteristics
     - `patient.sofa`: structure containing patient characteristics
     - `patient.id` or `patient.sid`: patient identifiers

### 4. **Model Output Variables**
   - Current: `s0`, `s1`, `t0`, `t1`, `ATE0`, `ATE1`
   - Suggestion:
     - `survival_control`, `survival_treated`
     - `time_control`, `time_treated`
     - `ATE_observed`, `ATE_true`

### 5. **Estimation Suffixes**
   - Current: Inconsistent use of `_est`, `_true`, etc.
   - Suggestion:
     - `_true`: true/generating values
     - `_est`: estimated values
     - `_obs`: observed values
     - `_sim`: simulated values

### 6. **Treatment Variables**
   - Current: `Rx`, `treatProb`, `th`
   - Suggestion:
     - `treatment_assigned`: actual treatment (0/1)
     - `treatment_prob`: probability of treatment
     - `treatment_threshold`: threshold for L(t) control

### 7. **Disease/Harm Variables**
   - Current: `L`, `harmE`, `harmA`, `cumsum_L`, `cumsum_A`
   - Suggestion:
     - `disease_burden`: for L(t)
     - `harm_disease`: for harmE
     - `harm_treatment`: for harmA
     - `cumulative_disease`: for cumsum_L
     - `cumulative_treatment`: for cumsum_A

### 8. **Function Return Values**
   - Suggestion: Use consistent ordering and naming for multi-output functions
   - Example: All estimation functions should return: `[params_est, fit_stats, diagnostics]`

### 9. **Boolean/Mode Variables**
   - Current: `RCT` (0/1)
   - Suggestion: Use descriptive boolean names
     - `is_RCT` or `trial_mode` with values 'RCT'/'observational'

### 10. **Documentation Standards**
   - Add consistent function headers with:
     - Inputs (with types and descriptions)
     - Outputs (with types and descriptions)
     - Brief algorithm description
     - References to related functions

These naming conventions would make the codebase more readable and maintainable while reducing confusion about variable purposes and scopes.