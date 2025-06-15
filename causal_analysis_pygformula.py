#!/usr/bin/env python3
"""
Causal Survival Analysis using pygformula
=========================================

This script uses the parametric g-formula to estimate causal treatment effects
from the simulated trial data, accounting for time-varying treatment and confounding.

Author: Generated for causal inference analysis
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygformula import ParametricGformula
from pygformula.parametric_gformula.model_builders import get_model_builders
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data(filename='trial_data.csv'):
    """
    Load trial data and prepare it for g-formula analysis.
    
    Parameters:
    -----------
    filename : str
        Path to the trial data CSV file
        
    Returns:
    --------
    df : pd.DataFrame
        Prepared dataframe for analysis
    """
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    
    # Basic data info
    n_patients = df['sid'].nunique()
    n_obs = len(df)
    print(f"Loaded {n_obs} observations from {n_patients} patients")
    
    # Check treatment patterns
    n_treated_initial = df[df['t'] == 0]['Rx'].sum()
    n_control_initial = n_patients - n_treated_initial
    print(f"Initial treatment assignment: {n_treated_initial} treated, {n_control_initial} control")
    
    # Check mortality
    n_deaths = df.groupby('sid')['Y'].max().sum()
    print(f"Total deaths: {n_deaths} ({100*n_deaths/n_patients:.1f}%)")
    
    # Create lagged variables for time-varying confounding
    print("\nCreating lagged variables...")
    df = df.sort_values(['sid', 't'])
    
    # Lag disease burden (L) - important confounder
    df['L_lag1'] = df.groupby('sid')['L'].shift(1)
    df['L_lag1'] = df['L_lag1'].fillna(0)  # Baseline
    
    # Lag treatment (Rx) - for treatment history
    df['Rx_lag1'] = df.groupby('sid')['Rx'].shift(1)
    df['Rx_lag1'] = df['Rx_lag1'].fillna(0)  # Baseline
    
    # Create cumulative variables
    df['L_cumsum'] = df.groupby('sid')['L'].cumsum()
    df['A_cumsum'] = df.groupby('sid')['A'].cumsum()
    
    # Create time squared for flexible time modeling
    df['t_squared'] = df['t'] ** 2
    
    # Create outcome indicator for survival analysis
    # For survival, outcome is death at each time point
    df['Y_event'] = df['Y'].astype(int)
    
    # Create censoring indicator (dropout)
    df['C'] = df['V'].astype(int)  # V is dropout indicator
    
    print("Data preparation complete.")
    return df


def run_gformula_analysis(df, intervention_type='static'):
    """
    Run parametric g-formula analysis for causal survival estimation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Prepared trial data
    intervention_type : str
        Type of intervention to evaluate ('static', 'dynamic', or 'natural')
        
    Returns:
    --------
    results : dict
        G-formula results including survival curves and treatment effects
    """
    print(f"\n{'='*60}")
    print(f"Running g-formula analysis with {intervention_type} intervention")
    print(f"{'='*60}")
    
    # Define variable types for g-formula
    time_varying_confounders = ['L', 'L_lag1', 'L_cumsum']
    time_varying_treatments = ['Rx']
    outcomes = ['Y_event']
    competing_events = ['C']  # Censoring/dropout
    
    # Define baseline confounders (none in this simplified case)
    baseline_confounders = []
    
    # Set up intervention
    if intervention_type == 'static':
        # Static interventions: always treat vs never treat
        interventions = {
            'Always Treat': {'Rx': lambda x: 1},
            'Never Treat': {'Rx': lambda x: 0}
        }
    elif intervention_type == 'dynamic':
        # Dynamic intervention: treat if disease burden > threshold
        interventions = {
            'Treat if L > 0.1': {'Rx': lambda x: (x['L'] > 0.1).astype(int)},
            'Treat if L > 0.05': {'Rx': lambda x: (x['L'] > 0.05).astype(int)},
            'Never Treat': {'Rx': lambda x: 0}
        }
    else:  # natural
        interventions = {
            'Natural Course': {}  # No intervention
        }
    
    # Model specifications
    models = {
        # Time-varying confounder models
        'L': {
            'model_type': 'linear',
            'formula': 'L ~ L_lag1 + Rx_lag1 + t + t_squared'
        },
        'L_lag1': {
            'model_type': 'linear', 
            'formula': 'L_lag1 ~ L_lag1 + Rx_lag1 + t'
        },
        'L_cumsum': {
            'model_type': 'linear',
            'formula': 'L_cumsum ~ L_lag1 + L_cumsum + Rx_lag1 + t'
        },
        
        # Treatment model (for natural course)
        'Rx': {
            'model_type': 'logistic',
            'formula': 'Rx ~ L + L_lag1 + Rx_lag1 + t'
        },
        
        # Outcome model
        'Y_event': {
            'model_type': 'logistic',
            'formula': 'Y_event ~ L + L_cumsum + Rx + A_cumsum + t + t_squared'
        },
        
        # Competing event (censoring) model
        'C': {
            'model_type': 'logistic',
            'formula': 'C ~ L + L_lag1 + Rx + t'
        }
    }
    
    # Initialize g-formula
    gf = ParametricGformula(
        data=df,
        id_column='sid',
        time_column='t',
        time_varying_confounders=time_varying_confounders,
        time_varying_treatments=time_varying_treatments,
        outcomes=outcomes,
        competing_events=competing_events,
        baseline_confounders=baseline_confounders,
        models=models,
        interventions=interventions,
        n_bootstrap=100,  # Number of bootstrap samples for confidence intervals
        random_seed=42
    )
    
    print("\nFitting models...")
    gf.fit()
    
    print("\nSimulating counterfactual outcomes...")
    results = gf.simulate(n_simulations=1000)
    
    # Extract survival curves
    survival_curves = {}
    for intervention_name in interventions.keys():
        # Calculate survival probability at each time point
        survival_probs = []
        time_points = sorted(df['t'].unique())
        
        for t in time_points:
            # Survival = 1 - cumulative incidence of death by time t
            death_by_t = results[intervention_name]['Y_event'][:, :int(t*2)+1].max(axis=1).mean()
            survival_probs.append(1 - death_by_t)
        
        survival_curves[intervention_name] = {
            'time': time_points,
            'survival': survival_probs
        }
    
    # Calculate treatment effects
    if intervention_type == 'static':
        ate_survival = (survival_curves['Always Treat']['survival'][-1] - 
                       survival_curves['Never Treat']['survival'][-1])
        print(f"\nAverage Treatment Effect on Survival: {ate_survival:.3f}")
    
    return {
        'gformula': gf,
        'results': results,
        'survival_curves': survival_curves,
        'interventions': interventions
    }


def plot_survival_curves(survival_curves, title="G-Formula Estimated Survival Curves"):
    """
    Plot survival curves from g-formula results.
    
    Parameters:
    -----------
    survival_curves : dict
        Dictionary of survival curves by intervention
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (intervention_name, curve_data) in enumerate(survival_curves.items()):
        plt.plot(curve_data['time'], 
                curve_data['survival'], 
                label=intervention_name,
                color=colors[i % len(colors)],
                linewidth=2)
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Survival Probability')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(curve_data['time']))
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


def compare_with_observed(df, survival_curves):
    """
    Compare g-formula estimates with observed Kaplan-Meier curves.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original trial data
    survival_curves : dict
        G-formula estimated survival curves
    """
    print("\nComparing with observed data...")
    
    # Calculate observed Kaplan-Meier for initially treated vs untreated
    from lifelines import KaplanMeierFitter
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: G-formula estimates
    plt.subplot(1, 2, 1)
    for intervention_name, curve_data in survival_curves.items():
        plt.plot(curve_data['time'], curve_data['survival'], 
                label=f"G-formula: {intervention_name}", linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Survival Probability')
    plt.title('G-Formula Estimates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Observed Kaplan-Meier
    plt.subplot(1, 2, 2)
    
    # Get data in survival format
    survival_data = df.groupby('sid').agg({
        't': 'max',  # Follow-up time
        'Y': 'max',  # Death indicator
        'Rx': 'first'  # Initial treatment
    }).reset_index()
    
    kmf = KaplanMeierFitter()
    
    # Treated group
    treated = survival_data[survival_data['Rx'] == 1]
    kmf.fit(treated['t'], treated['Y'], label='Observed: Initially Treated')
    kmf.plot_survival_function(color='red')
    
    # Control group
    control = survival_data[survival_data['Rx'] == 0]
    kmf.fit(control['t'], control['Y'], label='Observed: Initially Control')
    kmf.plot_survival_function(color='blue')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Survival Probability')
    plt.title('Observed Kaplan-Meier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_complete_analysis():
    """
    Run complete causal survival analysis pipeline.
    """
    print("="*60)
    print("CAUSAL SURVIVAL ANALYSIS USING G-FORMULA")
    print("="*60)
    
    # Load and prepare data
    df = load_and_prepare_data('trial_data.csv')
    
    # Run static intervention analysis
    static_results = run_gformula_analysis(df, intervention_type='static')
    
    # Plot results
    plot_survival_curves(static_results['survival_curves'], 
                        title="G-Formula: Always Treat vs Never Treat")
    
    # Compare with observed data
    compare_with_observed(df, static_results['survival_curves'])
    
    # Run dynamic intervention analysis
    print("\nTrying dynamic treatment strategies...")
    dynamic_results = run_gformula_analysis(df, intervention_type='dynamic')
    plot_survival_curves(dynamic_results['survival_curves'],
                        title="G-Formula: Dynamic Treatment Strategies")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    # Static intervention effects
    always_treat_survival = static_results['survival_curves']['Always Treat']['survival'][-1]
    never_treat_survival = static_results['survival_curves']['Never Treat']['survival'][-1]
    ate = always_treat_survival - never_treat_survival
    
    print(f"\nStatic Interventions (at end of follow-up):")
    print(f"  Always Treat: {always_treat_survival:.3f} survival")
    print(f"  Never Treat: {never_treat_survival:.3f} survival")
    print(f"  Average Treatment Effect: {ate:.3f}")
    print(f"  Relative Risk Reduction: {ate/never_treat_survival:.1%}")
    
    # Dynamic intervention effects (if run)
    if 'dynamic_results' in locals():
        print(f"\nDynamic Interventions (at end of follow-up):")
        for intervention_name, curve_data in dynamic_results['survival_curves'].items():
            final_survival = curve_data['survival'][-1]
            print(f"  {intervention_name}: {final_survival:.3f} survival")
    
    return {
        'data': df,
        'static_results': static_results,
        'dynamic_results': dynamic_results if 'dynamic_results' in locals() else None
    }


if __name__ == "__main__":
    # Run the complete analysis
    results = run_complete_analysis()
    
    print("\nAnalysis complete! Results stored in 'results' dictionary.")
    print("\nTo run this analysis with your own parameters, you can:")
    print("1. Modify the intervention definitions")
    print("2. Change the model specifications") 
    print("3. Adjust the number of bootstrap samples")
    print("4. Add additional time-varying confounders")