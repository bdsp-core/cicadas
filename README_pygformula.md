# Causal Survival Analysis with pygformula

This directory contains Python scripts for performing causal survival analysis on the ICU PKPD trial data using the parametric g-formula.

## Overview

The g-formula is a powerful method for estimating causal effects with time-varying treatments and confounders. It's particularly well-suited for this dataset because:

1. **Handles time-varying treatment** (Rx can change over time due to switching/stopping)
2. **Accounts for time-varying confounding** (disease burden L affects both treatment and outcome)
3. **Estimates counterfactual survival** under different treatment strategies
4. **Provides valid causal inference** from observational data (RCT = 0)

## Files

- `causal_analysis_pygformula.py` - Comprehensive analysis script with multiple intervention types
- `gformula_simple_example.py` - Simplified example focusing on core concepts
- `requirements_pygformula.txt` - Python package dependencies

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_pygformula.txt
```

## Quick Start

1. Generate trial data using MATLAB:
```matlab
% In MATLAB
a0_GenerateData  % Creates trial_data.csv
```

2. Run g-formula analysis:
```bash
python gformula_simple_example.py
```

## Understanding the G-Formula

### Key Concepts

1. **Time-Varying Treatment**: Treatment status (Rx) can change over time
2. **Time-Varying Confounders**: Disease burden (L) affects both treatment and survival
3. **Counterfactual Outcomes**: What would happen under different treatment strategies

### Interventions Tested

1. **Always Treat**: All patients receive treatment at all times
2. **Never Treat**: No patients receive treatment at any time
3. **Natural Course**: Observed treatment patterns (for comparison)
4. **Dynamic Strategies**: Treat based on disease burden thresholds

### Model Components

The g-formula requires models for:
- **Confounders** (L): How disease burden evolves over time
- **Treatment** (Rx): Treatment assignment mechanism (for natural course)
- **Outcome** (Y): Mortality risk at each time point

## Expected Results

When comparing RCT = 0 (observational) data:

1. **Natural Course** will show confounded results (sicker patients get treated)
2. **Always Treat vs Never Treat** estimates the true causal effect
3. This should approximate the RCT = 1 treatment effect

## Advanced Usage

### Custom Interventions

```python
# Example: Treat only if disease burden exceeds threshold
custom_intervention = {
    'name': 'Threshold Strategy',
    'intervention': lambda df: df['L'] > 0.1  # Treat if L > 0.1
}
```

### Model Specification

```python
# More complex models with interactions
models = {
    'L': {
        'formula': 'L ~ L_lag + Rx_lag + t + t^2 + L_lag:Rx_lag',
        'family': 'gaussian'
    },
    'Y': {
        'formula': 'Y ~ L + Rx + L:Rx + cumsum(L) + cumsum(A)',
        'family': 'binomial'
    }
}
```

### Bootstrap Confidence Intervals

```python
# Increase bootstrap samples for more precise CIs
gformula = ParametricGformula(..., n_bootstraps=200)
```

## Validation

To validate the g-formula estimates:

1. Run analysis on RCT = 1 data (randomized)
2. Compare "Always Treat vs Never Treat" with observed treatment groups
3. They should be similar (validating the g-formula implementation)

Then:

1. Run analysis on RCT = 0 data (observational)
2. The g-formula should recover the true effect despite confounding

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `monte_carlo_simulations` or process in batches
2. **Convergence warnings**: Check model specifications, may need simpler models
3. **Unrealistic estimates**: Verify positivity (all patients can receive all treatments)

### Data Requirements

- Data must be in "long" format (one row per patient-time)
- No missing values in key variables (L, Rx, Y)
- Sufficient follow-up for survival estimation

## References

- Robins JM. (1986). A new approach to causal inference in mortality studies.
- Hernán MA, Robins JM. (2020). Causal Inference: What If.
- pygformula documentation: https://github.com/pygformula/pygformula

## Contact

For questions about the simulation: [Your contact info]
For pygformula issues: https://github.com/pygformula/pygformula/issues