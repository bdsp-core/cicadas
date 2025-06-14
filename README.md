# ICU EEG PKPD Emulated Trial Simulations

This repository contains MATLAB simulation code for causal survival analysis with pharmacokinetic-pharmacodynamic (PKPD) modeling in an ICU setting.

## Overview

The code simulates clinical trials with the following features:
- Randomized controlled trial (RCT) or observational study designs
- Patient-specific harm parameters for disease and treatment
- PI-controller based treatment administration
- Survival and hazard function calculations
- Comparison of empirical vs theoretical curves

## Main Workflow

### 1. Data Generation and Visualization
Run `a3_simulate_empirical.m` to:
- Generate synthetic trial data for N patients
- Simulate disease progression and treatment effects
- Create swimmer plots showing patient trajectories
- Export data to `trial_data.csv`

### 2. Survival Analysis
Run `a4_survivalCurves.m` to:
- Read the generated trial data
- Calculate Kaplan-Meier survival curves
- Compute empirical and theoretical hazard functions
- Generate comparison plots

## Key Files

### Main Scripts
- `a3_simulate_empirical.m` - Generate trial data and swimmer plots
- `a4_survivalCurves.m` - Analyze survival and hazard curves
- `a0_simulate_Ideal.m` - Ideal simulation scenarios
- `a1_simulate_Rx_heatmaps.m` - Treatment effect heatmaps
- `a2_simulateData_many.m` - Multiple simulation runs

### Core Functions
- `fcnRunSimulation_GetDataOnly.m` - Main simulation engine
- `fcnRunSimulation_v2.m` - Extended simulation with theoretical curves
- `fcnEmpiricalSurvivalCurves.m` - Kaplan-Meier estimation
- `fcnDualSwimmerPlot.m` - Visualization of patient trajectories
- `fcnSmoothPulse.m` - Disease progression modeling

### Old Versions
The `old/` directory contains previous versions of functions that are no longer in active use but retained for reference.

## Parameters

### Model Parameters
- `C` - Drug potency (default: 3)
- `g` - Dose-response curve steepness (default: 4)
- `kp` - Proportional control gain (default: 0)
- `ki` - Integral control gain (default: 20)
- `Amax` - Maximum pump rate (default: 50)
- `th` - Control target threshold (default: 0.05)

### Simulation Parameters
- `N` - Number of patients (default: 2000)
- `RCT` - Trial type (1=RCT, 0=observational)
- `b0` - Baseline hazard risk (default: 0.1)

## Output Files

- `trial_data.csv` - Long-format trial data with columns:
  - `sid` - Subject ID
  - `t` - Time point
  - `Rx` - Treatment assignment (0/1)
  - `harmE` - Patient-specific disease harm
  - `harmA` - Patient-specific treatment harm
  - `L` - Disease burden
  - `A` - Treatment amount
  - `V` - Control variable
  - `Y` - Outcome (death indicator)

## Requirements

- MATLAB R2020a or later
- Statistics and Machine Learning Toolbox (for survival analysis functions)

## Usage Example

```matlab
% Generate trial data
a3_simulate_empirical

% Analyze survival curves
a4_survivalCurves
```

## License

See LICENSE.txt for details.

## Authors

- Mitchell McCauley
- Brandon Westover

## Acknowledgments

This work is part of the Brain Data Science Platform (BDSP) at Beth Israel Deaconess Medical Center.