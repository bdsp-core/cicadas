import numpy as np
from typing import Tuple

def fcnBiasedAssignmentProb(age: np.ndarray, sofa: np.ndarray, L_early: np.ndarray) -> np.ndarray:
    """
    Direct translation of MATLAB:
        p = fcnBiasedAssignmentProb(age, sofa, L_early)

    Inputs:
        age     : (n,) array
        sofa    : (n,) array
        L_early : (n, k) array (first k time points of L; in MATLAB k=5)

    Output:
        p       : (n,) array of treatment assignment probabilities in [0.05, 0.95]
    """
    age = np.asarray(age, dtype=float).reshape(-1)
    sofa = np.asarray(sofa, dtype=float).reshape(-1)
    L_early = np.asarray(L_early, dtype=float)
    n_patients = age.shape[0]

    if L_early.shape[0] != n_patients:
        raise ValueError("L_early must have the same number of rows as age/sofa.")
    if L_early.ndim != 2 or L_early.shape[1] < 1:
        raise ValueError("L_early must be a 2D array with at least one time point (columns).")

    p = np.zeros(n_patients, dtype=float)

    for i in range(n_patients):
        L_traj = L_early[i, :]

        # Normalize age and SOFA
        age_norm = (age[i] - 18.0) / (90.0 - 18.0)
        sofa_norm = sofa[i] / 24.0

        # Metrics
        L_initial = float(L_traj[0])
        L_final = float(L_traj[-1])
        L_max = float(np.max(L_traj))
        L_mean = float(np.mean(L_traj))

        # Relative growth with epsilon
        epsilon = 1e-3
        L_relative_growth = (L_final - L_initial) / (L_initial + epsilon)

        # Absolute slope
        L_slope = (L_final - L_initial) / L_traj.size

        # Acceleration
        mid_point = int(np.ceil(L_traj.size / 2.0))
        if L_traj.size >= 3:
            first_half_slope = (L_traj[mid_point - 1] - L_traj[0]) / (mid_point - 1)
            second_half_slope = (L_traj[-1] - L_traj[mid_point - 1]) / (L_traj.size - mid_point)
            acceleration = second_half_slope - first_half_slope
        else:
            acceleration = 0.0

        # Revised normalization
        L_mean_norm = L_mean * 20.0
        L_max_norm = L_max * 10.0
        L_relative_growth_norm = np.tanh(L_relative_growth / 2.0)
        L_slope_norm = L_slope * 100.0
        acceleration_norm = acceleration * 200.0

        # Coefficients
        L_mean_effect = 0.8
        L_max_effect = 0.6
        L_relative_growth_effect = 1.2
        L_slope_effect = 1.0
        acceleration_effect = 0.8
        sofa_effect = 0.5
        age_effect = -0.3

        # Intercept
        intercept = -0.5

        # Log-odds (z)
        z = (
            intercept
            + L_mean_effect * L_mean_norm
            + L_max_effect * L_max_norm
            + L_relative_growth_effect * L_relative_growth_norm
            + L_slope_effect * L_slope_norm
            + acceleration_effect * max(0.0, acceleration_norm)
            + sofa_effect * sofa_norm
            + age_effect * age_norm
        )

        # Add random noise
        z = z + 0.3 * np.random.randn()

        # Probability via logistic
        p[i] = 1.0 / (1.0 + np.exp(-z))

    # Bounds
    p = np.clip(p, 0.05, 0.95)

    # Debug output
    print(
        "Treatment assignment probabilities - "
        f"Mean: {np.mean(p):.3f}, Std: {np.std(p):.3f}, "
        f"Min: {np.min(p):.3f}, Max: {np.max(p):.3f}"
    )

    return p
