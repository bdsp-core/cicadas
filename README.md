# ICU EEG PKPD Emulated Trial Simulations

This repository contains MATLAB simulation code for causal survival analysis with pharmacokinetic-pharmacodynamic (PKPD) modeling in an ICU setting. The simulation generates synthetic clinical trial data and analyzes treatment effectiveness using survival analysis methods.

## Overview

The codebase simulates randomized controlled trials (RCTs) with the following features:
- **Patient-specific disease and treatment harm parameters**
- **PI-controller based automated treatment administration**
- **Mortality modeling with time-dependent hazard functions**
- **Kaplan-Meier survival curve estimation**
- **Empirical vs theoretical curve comparison**
- **Comprehensive visualization and statistical analysis**

## Quick Start

The main workflow consists of two steps:

### 1. Generate Trial Data
```matlab
a0_GenerateData  % Generates trial_data.csv and creates visualizations
```

### 2. Analyze Survival Curves  
```matlab
a1_SurvivalCurves_RCT  % Analyzes survival data and creates comparison plots
```

## Detailed Workflow

### Step 1: Data Generation (`a0_GenerateData.m`)
**Purpose**: Generate synthetic clinical trial data with realistic patient trajectories

**Process**:
- Simulates N=2000 patients with randomized treatment assignment
- Generates patient-specific harm parameters (harmE, harmA)
- Models disease progression using log-normal trajectory functions
- Applies PI-controller for automated treatment administration
- Calculates mortality events based on cumulative disease and treatment exposure
- Exports longitudinal data to `trial_data.csv`

**Outputs**:
- `trial_data.csv` - Complete longitudinal trial dataset
- Figure 1: Disease burden trajectories for all patients
- Figure 2: Dual swimmer plot showing survival outcomes

### Step 2: Survival Analysis (`a1_SurvivalCurves_RCT.m`)
**Purpose**: Analyze treatment effectiveness using survival analysis methods

**Process**:
- Imports trial data and validates completeness
- Calculates Kaplan-Meier survival curves for treated vs control groups
- Computes empirical hazard functions from PKPD model trajectories
- Fits smooth theoretical curves using PCHIP interpolation and moving averages
- Generates comparative visualizations and summary statistics

**Outputs**:
- Figure 1: Kaplan-Meier vs fitted survival curves
- Figure 2: Empirical vs fitted hazard functions
- Console output: Treatment effect estimates and curve summaries

## File Structure

### Main Scripts
- **`a0_GenerateData.m`** - Primary data generation script
- **`a1_SurvivalCurves_RCT.m`** - Survival analysis and visualization script

### Core Supporting Functions
- **`fcnRunSimulation_GetDataOnly.m`** - Individual patient simulation engine
- **`fcnEmpiricalSurvivalCurves.m`** - Kaplan-Meier survival curve estimation
- **`fcnDualSwimmerPlot.m`** - Patient trajectory visualization (swimmer plots)
- **`fcnSmoothPulse.m`** - Disease natural history modeling

### Data Files
- **`trial_data.csv`** - Generated longitudinal trial dataset (output from `a0_GenerateData.m`, input to `a1_SurvivalCurves_RCT.m`)

### Legacy Files
Files moved to `old/` directory contain previous implementations and test functions that are no longer part of the main workflow but retained for reference.

## Model Parameters

### PI Controller Parameters (in `a0_GenerateData.m`)
- **`C`** = 3 - Drug potency (higher = more effective treatment)
- **`g`** = 4 - Dose-response curve steepness  
- **`kp`** = 0 - Proportional control gain (pure integral control)
- **`ki`** = 20 - Integral control gain (aggressive disease suppression)
- **`Amax`** = 50 - Maximum pump rate (treatment upper bound)
- **`th`** = 0.05 - Control target threshold (very low for tight control)

### Disease and Hazard Parameters
- **`b0`** = 0.1 - Baseline mortality risk
- **`harmE`** ~ N(30, 10²) - Patient-specific disease harm (truncated at 0)
- **`harmA`** ~ N(10, 5²) - Patient-specific treatment harm (truncated at 0)
- **Pulse parameters** - Disease natural history trajectory (log-normal shape)

### Trial Design Parameters
- **`N`** = 2000 - Number of patients to simulate
- **`RCT`** = 1 - Trial type (1=RCT, 0=observational study)

#### RCT Parameter Details
The `RCT` parameter controls **treatment assignment** while keeping all underlying biological processes identical:

- **`RCT = 1` (Randomized Controlled Trial)**:
  - Treatment assigned randomly (50% probability)
  - Independent of patient characteristics
  - Eliminates selection bias and confounding
  
- **`RCT = 0` (Observational Study)**:
  - Treatment assignment biased by patient characteristics
  - Patients with higher disease-to-treatment harm ratio more likely treated
  - Introduces realistic selection bias and confounding

**Important**: The underlying disease progression, treatment response, and mortality models are **identical** between RCT = 0 and RCT = 1. Only the treatment assignment mechanism differs, allowing direct comparison of randomized vs observational study designs on the same population.

## Data Structure

The generated `trial_data.csv` contains longitudinal data with the following variables:

| Variable | Description |
|----------|-------------|
| `sid` | Subject ID (1 to N) |
| `t` | Time point (hours) |
| `Rx` | Treatment assignment (0=control, 1=treated) |
| `harmE` | Patient-specific disease harm parameter |
| `harmA` | Patient-specific treatment harm parameter |
| `b0` | Baseline risk (constant across patients) |
| `L` | Disease burden at time t |
| `A` | Treatment amount at time t |
| `V` | PI controller variable at time t |
| `Y` | Death indicator (0=alive, 1=death) |
| `pulseAmpR`, `pulseMuR`, `pulseWidthR`, `pulseCR` | Patient-specific pulse parameters |

## Methods and Algorithms

### Survival Analysis Methods
- **Kaplan-Meier estimation** for empirical survival curves
- **PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)** for smooth survival curve fitting
- **Moving average smoothing** for hazard function estimation
- **Patient-specific hazard calculation** using cumulative PKPD model exposure

### Disease Progression Model
- **Log-normal trajectory functions** for natural disease history
- **PI controller** for automated treatment administration based on disease burden
- **Time-dependent mortality hazard** incorporating both disease and treatment effects

### Statistical Outputs
- **Hazard ratios** (treated vs control)
- **Survival probability differences** at end of follow-up
- **Mean hazard rates** over the study period
- **Treatment group balance** and mortality event counts

## System Requirements

- **MATLAB R2020a or later**
- **Statistics and Machine Learning Toolbox** (for survival analysis functions)
- **Sufficient memory** for N=2000 patient simulation (~40MB CSV output)

## Example Usage

```matlab
% Complete analysis workflow
a0_GenerateData          % Generate synthetic trial data (~2-3 minutes)
a1_SurvivalCurves_RCT   % Analyze survival curves (~30 seconds)
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See `LICENSE.txt` for full details.

**Commercial use is prohibited.**

## Authors and Acknowledgments

**Authors:**
- Mitchell McCauley
- Brandon Westover

**Institution:**
Brain Data Science Platform (BDSP)  
Beth Israel Deaconess Medical Center

**Last Updated:** June 15, 2025