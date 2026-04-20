# %% Logit-Based Hazard Model Data Generation
#
# DESCRIPTION:
#   Generates synthetic clinical trial data using logit-based discrete-time
#   hazard models for causal survival analysis with PKPD modeling.
#
# TRIAL MODES:
#   RCT=1: No treatment changes or dropout (yes randomized controlled trial)
#   RCT=0: Treatment changes and dropout allowed (not randomized controlled trial)

# clear; clc; format compact;
# rng(0) % Set random seed for reproducibility

import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt

np.random.seed(0)

# Helper functions
from fcnGeneratePatientParameters import fcnGeneratePatientParameters
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnSimulate_DoseChanging import fcnSimulate_DoseChanging

# %% 1. SIMULATION PARAMETERS ==========================================

N = 1000  # Number of patients to simulate per trial type

# Target L level
th = 0.1  # LOWER threshold - harder to achieve for sick patients, leading to more treatment

# PI Controller Parameters
ki = 10      # Integral control gain (aggressive disease suppression)
Amax = 50    # Maximum pump rate (treatment upper bound)
parmsControl = np.array([ki, Amax])

# PD parameters with age/SOFA dependencies
age, sofa, C, g, parmsPD = fcnGeneratePatientParameters(
    N, 'TargetCMean', 3, 'TargetGMean', 4, 'CV', 0.1
)

# PK parameter - elimination time constant
ke = 0.5

# Mortality (Y) hazard parameters
# logit_y = a0 + a1*(t(j)/170)^2 + (a2*sofa).*(cumsum_L/24)^2 + (a3*(age/90)).*(cumsum_A/207);
a0 = -7
a1 = 0.3
a2 = 20   # harmL
a3 = 5    # harmA
parmsY = np.array([a0, a1, a2, a3])

# Censoring (V) hazard parameters
# logit_v = Rx(j)*(b0 + b1*(cumsum_A/207) + b2*(t(j)/170)^2) + (1-Rx(j))*(b3 + b4*cumsum_L/24 + b5*(t(j)/170)^2);
b0 = -5     # Baseline censoring for treated
b1 = 2.0    # Treatment burden effect
b2 = 0.1    # Time effect for treated
b3 = -5     # Baseline for untreated
b4 = 2      # Disease burden effect
b5 = 1.5    # Time effect for untreated
parmsV = np.array([b0, b1, b2, b3, b4, b5])

# Disease Natural History Parameters and trajectories
# [growth_rate, peak_height, alpha, decay_rate, sigma_early, sigma_late, sigma_transition]
parmsL = np.array([0.25, 1, 0.15, 0.05, 0.15, 0.03, 40])
dt = 2
t = np.arange(0, 168 + dt, dt)  # MATLAB 0:dt:168 inclusive

# %% MAIN LOOP =========================================================
# Mirrors MATLAB a0_GenerateDoseSwitchingData.m which calls
# fcnSimulate_DoseChanging (NOT fcnSimulate_N_Patients). The previous
# Python revision called the RCT simulator here, producing RCT data in
# trialDataDoseChanging.csv; that silently broke the a1 ke-estimation
# identifiability (dose-change windows did not exist). Fixed.

# Generate baseline disease trajectories
L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)

# Simulate dose-changing trajectories. Seed=0 for reproducibility parity
# with the module-level np.random.seed(0) call above.
T = fcnSimulate_DoseChanging(
    N, th, C, g, ke, L0, parmsY, age, sofa, seed=0
)

# Export to CSV for analysis
filename = "trialDataDoseChanging.csv"
T.to_csv(filename, index=False)

# Save L0 as a .mat (mirrors 'save L0data L0')
savemat("L0data.mat", {"L0": L0}, do_compression=True)

# Save "true" values for generating data (parity with MATLAB)
savemat(
    "parmsTrue_DoseChanging.mat",
    {
        "parmsControl": parmsControl,
        "parmsPD": parmsPD,
        "C": C,
        "g": g,
        "ke": ke,
        "parmsY": parmsY,
        "parmsV": parmsV,
        "parmsL": parmsL,
        "age": age,
        "sofa": sofa,
    },
    do_compression=True,
)

# %% extract data and plot it (parity with MATLAB quick-look)
A_mat = np.full((N, t.size), np.nan)
L_mat = np.full((N, t.size), np.nan)

for i in range(1, N + 1):  # MATLAB 1-based subject IDs
    ind = np.where(T["sid"].values == i)[0]
    if ind.size > 0:
        Ai = T.loc[ind, "A"].to_numpy()
        Li = T.loc[ind, "L"].to_numpy()
        # Truncate or pad to Nt to mimic fixed grid
        Ai = Ai[:t.size] if Ai.size >= t.size else np.pad(Ai, (0, t.size - Ai.size), constant_values=np.nan)
        Li = Li[:t.size] if Li.size >= t.size else np.pad(Li, (0, t.size - Li.size), constant_values=np.nan)
        A_mat[i - 1, :] = Ai
        L_mat[i - 1, :] = Li

idx = 1  # MATLAB 1-based
plt.figure(1); plt.clf()
plt.subplot(2, 1, 1); plt.plot(t, A_mat[idx - 1, :], 'k')
plt.subplot(2, 1, 2); plt.plot(t, L_mat[idx - 1, :], 'r')
# plt.show()  # optional

# %% estimation of C, g, ke (hooks preserved to mirror MATLAB)
# run_PKPD_estimation_pipeline
# test_Enhanced_StateSpace
# test_FixedKe_Comprehensive
