#!/usr/bin/env python3
"""
Simple G-Formula Example for ICU PKPD Trial Data
===============================================

A simplified example showing how to use pygformula for causal survival analysis.

This script demonstrates:
1. Basic data preparation
2. G-formula setup for time-varying treatment
3. Estimation of causal survival curves
4. Comparison of treatment strategies
"""

import pandas as pd
import numpy as np
from pygformula import ParametricGformula
from pygformula.data import load_sample_data
import matplotlib.pyplot as plt

# Load your trial data
print("Loading trial data...")
df = pd.read_csv('trial_data.csv')

# Basic data inspection
print(f"\nData shape: {df.shape}")
print(f"Patients: {df['sid'].nunique()}")
print(f"Time points: {df['t'].nunique()}")
print(f"Deaths: {df['Y'].sum()}")

# Data must be sorted by ID and time
df = df.sort_values(['sid', 't'])

# Create lagged variables (previous time point values)
print("\nCreating time-lagged variables...")
df['L_lag'] = df.groupby('sid')['L'].shift(1).fillna(0)
df['Rx_lag'] = df.groupby('sid')['Rx'].shift(1).fillna(0)

# Define the g-formula components
print("\nSetting up g-formula...")

# 1. Specify variable types
time_varying_confounders = ['L']  # Disease burden
time_varying_treatments = ['Rx']  # Treatment status
outcome = ['Y']  # Death indicator

# 2. Define models for each time-varying variable
models = {
    # Model for disease burden L
    'L': 'linear',  # Use linear regression
    
    # Model for treatment Rx (only needed for natural course)
    'Rx': 'logistic',  # Use logistic regression
    
    # Model for outcome Y
    'Y': 'logistic'  # Use logistic regression for binary outcome
}

# 3. Define interventions to compare
interventions = [
    # Intervention 1: Always treat everyone
    {
        'name': 'Always Treat',
        'intervention': {'Rx': 1}  # Set Rx=1 for all patients at all times
    },
    # Intervention 2: Never treat anyone
    {
        'name': 'Never Treat', 
        'intervention': {'Rx': 0}  # Set Rx=0 for all patients at all times
    },
    # Intervention 3: Natural course (no intervention)
    {
        'name': 'Natural Course',
        'intervention': {}  # No intervention, use observed treatment patterns
    }
]

# 4. Initialize the g-formula estimator
gformula = ParametricGformula(
    df=df,
    id_col='sid',
    time_col='t',
    time_varying_confounders=time_varying_confounders,
    time_varying_treatments=time_varying_treatments,
    outcome=outcome,
    models=models,
    interventions=interventions,
    monte_carlo_simulations=500,  # Number of MC simulations
    n_bootstraps=50  # Number of bootstraps for confidence intervals
)

# 5. Fit the models
print("\nFitting models...")
gformula.fit()

# 6. Get results
print("\nEstimating causal effects...")
results = gformula.predict()

# 7. Extract and plot survival curves
print("\nPlotting results...")

plt.figure(figsize=(10, 6))

# Get unique time points
time_points = sorted(df['t'].unique())

# Plot survival curves for each intervention
colors = {'Always Treat': 'red', 'Never Treat': 'blue', 'Natural Course': 'green'}

for intervention in interventions:
    name = intervention['name']
    
    # Calculate survival probability at each time point
    # (1 - cumulative incidence of death)
    survival_probs = []
    
    for t in time_points:
        # Get cumulative mortality up to time t
        mortality_data = results[name]
        cum_mortality = mortality_data[mortality_data['t'] <= t]['Y'].mean()
        survival_probs.append(1 - cum_mortality)
    
    plt.plot(time_points, survival_probs, 
             label=name, 
             color=colors.get(name, 'black'),
             linewidth=2)

plt.xlabel('Time (hours)')
plt.ylabel('Survival Probability')
plt.title('G-Formula Estimated Survival Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, max(time_points))
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('gformula_survival_curves.png', dpi=300)
plt.show()

# 8. Calculate and report treatment effects
print("\n" + "="*50)
print("CAUSAL EFFECT ESTIMATES")
print("="*50)

# Get final survival probabilities
final_time = max(time_points)
always_treat_survival = 1 - results['Always Treat'][results['Always Treat']['t'] == final_time]['Y'].mean()
never_treat_survival = 1 - results['Never Treat'][results['Never Treat']['t'] == final_time]['Y'].mean()
natural_survival = 1 - results['Natural Course'][results['Natural Course']['t'] == final_time]['Y'].mean()

print(f"\nSurvival at t={final_time} hours:")
print(f"  Always Treat: {always_treat_survival:.3f}")
print(f"  Never Treat: {never_treat_survival:.3f}")
print(f"  Natural Course: {natural_survival:.3f}")

# Average Treatment Effect
ate = always_treat_survival - never_treat_survival
print(f"\nAverage Treatment Effect (ATE): {ate:.3f}")
print(f"Relative Risk Reduction: {ate/never_treat_survival:.1%}")

# Number Needed to Treat
if ate > 0:
    nnt = 1 / ate
    print(f"Number Needed to Treat (NNT): {nnt:.1f}")

print("\nAnalysis complete!")