# -*- coding: utf-8 -*-
# get survival curves across (A2, A3, Th) grid — Python translation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat

# Reproducibility (mirrors MATLAB rng(0))
np.random.seed(0)

# ---------------------------------------------------------------------
# KM helper (stand-in for fcnPlotKM). If you have a direct translation,
# import it and swap the two calls below.
# ---------------------------------------------------------------------
def _km_curve(times: np.ndarray, events: np.ndarray):
    """Kaplan–Meier S(t); returns (t_out, S_out) starting at (0,1)."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    order = np.argsort(times, kind="mergesort")
    times, events = times[order], events[order]
    uniq_event_times = np.unique(times[events == 1])
    if uniq_event_times.size == 0:
        return np.array([0.0]), np.array([1.0])
    n = len(times); at_risk = n; S = 1.0
    t_out = [0.0]; S_out = [1.0]; idx = 0
    for te in uniq_event_times:
        while idx < n and times[idx] < te:
            at_risk -= 1; idx += 1
        d_i = np.sum((times == te) & (events == 1))
        if at_risk > 0:
            S *= (1.0 - d_i / at_risk)
        t_out.append(te); S_out.append(S)
        k = 0
        while idx + k < n and times[idx + k] == te:
            k += 1
        at_risk -= k; idx += k
    return np.asarray(t_out), np.asarray(S_out)

def _times_events_for_group(Tdf: pd.DataFrame, pids):
    times, events = [], []
    for pid in pids:
        pdata = Tdf[Tdf["sid"] == pid].sort_values("t")
        y = pdata["Y"].values
        if (y > 0).any():
            t_event = float(pdata["t"].values[np.where(y > 0)[0][0]])
            times.append(t_event); events.append(1)
        else:
            times.append(float(np.max(pdata["t"].values))); events.append(0)
    return np.asarray(times, float), np.asarray(events, int)

def fcnPlotKM_py(Tdf: pd.DataFrame):
    """Return (s0, s1, t0, t1) by Rx-at-baseline groups."""
    uniq = np.unique(Tdf["sid"].values)
    treated, untreated = [], []
    for pid in uniq:
        pdata = Tdf[Tdf["sid"] == pid].sort_values("t")
        if int(pdata["Rx"].iloc[0]) == 1:
            treated.append(pid)
        else:
            untreated.append(pid)
    ut_times, ut_events = _times_events_for_group(Tdf, untreated)
    tr_times, tr_events = _times_events_for_group(Tdf, treated)
    t0, s0 = _km_curve(ut_times, ut_events)
    t1, s1 = _km_curve(tr_times, tr_events)
    return s0, s1, t0, t1

# ---------------------------------------------------------------------
# External helpers you translated elsewhere (same signatures as MATLAB)
# ---------------------------------------------------------------------
from fcnGeneratePatientParameters import fcnGeneratePatientParameters
from fcnGetPKPD_parms_est import fcnGetPKPD_parms_est
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnBiasedAssignmentProb import fcnBiasedAssignmentProb
from fcnEstimateDeathParms import fcnEstimateDeathParms
from fcnSimulate_N_Patients import fcnSimulate_N_Patients
# If you have fcnPlotKM in Python, you can import it and replace fcnPlotKM_py.

# ---------------------------------------------------------------------
# Script body (mirrors MATLAB)
# ---------------------------------------------------------------------

# clear; clc; format compact;

# 1) Simulation parameters
N = 1000          # patients per trial type
nTrials = 1       # median across repeated trials (can increase)

th_scalar = 0.1   # control target threshold (not used in sweep below)
ki = 10.0
Amax = 50.0
parmsControl = np.array([ki, Amax], dtype=float)
ke = 0.5

# Mortality (Y) hazard parameters (baseline; will be overwritten in sweep)
a0, a1, a2, a3 = -7.0, 0.3, 20.0, 5.0
parmsY = np.array([a0, a1, a2, a3], dtype=float)

# Censoring hazard parameters
b0, b1, b2, b3, b4, b5 = -5.0, 2.0, 0.1, -5.0, 2.0, 1.5
parmsV = np.array([b0, b1, b2, b3, b4, b5], dtype=float)

# Disease natural history params
parmsL = np.array([0.25, 1.0, 0.15, 0.05, 0.15, 0.03, 40.0], dtype=float)

dt = 2.0
t = np.arange(0.0, 168.0 + dt, dt)
Nt = t.size

# 2) Sweep grids (Th, A2, A3)
Th = np.linspace(0.01, 1.10, 20)  # thresholds to test
A2 = np.linspace(1.0, 50.0, 20)   # harm from L coefficient
A3 = np.linspace(1.0, 50.0, 20)   # harm from A coefficient

# Preallocate heatmap outputs
s0 = np.full((A2.size, A3.size), np.nan)  # untreated survival at best Th
s1 = np.full((A2.size, A3.size), np.nan)  # treated survival at best Th
Th_opt = np.full((A2.size, A3.size), np.nan)

# --- make patients and baseline data ---
age, sofa, C, g, parmsPD = fcnGeneratePatientParameters(
    N, "TargetCMean", 3, "TargetGMean", 4, "CV", 0.1
)
C_est, g_est, ke_est, parmsPD_est = fcnGetPKPD_parms_est(age, sofa)
L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)
treatProbBiased = fcnBiasedAssignmentProb(age, sofa, L0[:, :5])
T0_df = pd.read_csv("trialData0.csv")
parmsY_est = fcnEstimateDeathParms(T0_df)

# 3) Grid sweep
for ii in range(A2.size):
    for jj in range(A3.size):
        print("*****************")
        print([A2[ii], A3[jj]])
        print("*****************")

        # Twiddle mortality params
        parmsY_mod = np.array([-7.0, 0.3, A2[ii], A3[jj]], dtype=float)
        parmsY_est_mod = np.asarray(parmsY_est, dtype=float).copy()
        if parmsY_est_mod.size >= 4:
            parmsY_est_mod[2] = A2[ii]
            parmsY_est_mod[3] = A3[jj]

        # Trials storage (|Th| x nTrials)
        S0_trials = np.full((Th.size, nTrials), np.nan)
        S1_trials = np.full((Th.size, nTrials), np.nan)

        for trial in range(nTrials):
            for k, th in enumerate(Th):
                # Ideal data (RCT=1) using TRUE parmsY_mod (not used directly below)
                RCT = 1
                treatProb = np.full(N, 0.5)
                _T1 = fcnSimulate_N_Patients(
                    N, RCT, treatProb, th, C, g, ke, L0, parmsControl, parmsY_mod, parmsV, age, sofa
                )

                # Observational run (RCT=0) — not used for KM here but kept for parity
                RCT = 0
                _T0_obs = fcnSimulate_N_Patients(
                    N, RCT, treatProbBiased, th, C, g, ke, L0, parmsControl, parmsY_mod, parmsV, age, sofa
                )

                # Simulated RCT using estimated models (no censoring)
                RCT = 1
                treatProb = np.full(N, 0.5)
                parmsV_est = np.zeros(6, dtype=float)
                T1_est = fcnSimulate_N_Patients(
                    N, RCT, treatProb, th,
                    C_est, g_est, ke_est, L0, parmsControl, parmsY_est_mod, parmsV_est, age, sofa
                )

                # KM on estimated-model RCT
                s0_est, s1_est, _, _ = fcnPlotKM_py(T1_est)
                S1_trials[k, trial] = s1_est[-1]
                S0_trials[k, trial] = s0_est[-1]

        # Median across trials
        S0_med = np.nanmedian(S0_trials, axis=1)
        S1_med = np.nanmedian(S1_trials, axis=1)

        # Optimal threshold index (max treatment effect)
        diff = S1_med - S0_med
        jj_star = int(np.nanargmax(diff))
        th_opt = Th[jj_star]
        Th_opt[ii, jj] = th_opt
        s0[ii, jj] = S0_med[jj_star]
        s1[ii, jj] = S1_med[jj_star]

        # Live heatmaps (replicates imagesc(...); axis xy via origin="lower")
        plt.figure(1); plt.clf()

        plt.subplot(3, 1, 1)
        im1 = plt.imshow(
            s1.T, origin="lower",
            extent=[A2.min(), A2.max(), A3.min(), A3.max()],
            aspect="auto"
        )
        plt.xlabel("Harm from L"); plt.ylabel("Harm from A"); plt.title("s1 (Treated)")
        plt.colorbar(im1)

        plt.subplot(3, 1, 2)
        im2 = plt.imshow(
            s0.T, origin="lower",
            extent=[A2.min(), A2.max(), A3.min(), A3.max()],
            aspect="auto"
        )
        plt.xlabel("Harm from L"); plt.ylabel("Harm from A"); plt.title("s0 (Untreated)")
        plt.colorbar(im2)

        plt.subplot(3, 1, 3)
        delta = (s1 - s0)
        im3 = plt.imshow(
            delta.T, origin="lower",
            extent=[A2.min(), A2.max(), A3.min(), A3.max()],
            aspect="auto"
        )
        plt.xlabel("Harm from L"); plt.ylabel("Harm from A"); plt.title("s1 - s0")
        plt.colorbar(im3)

        plt.gcf().set_facecolor("white")
        plt.tight_layout()
        plt.pause(0.001)

    Th_opt3 = Th_opt.copy()
    # Save after each outer-loop iteration (matches MATLAB 'save HeatMapData' inside loop)
    savemat(
        "HeatMapData.mat",
        {
            "A2": A2,
            "A3": A3,
            "s0": s0,
            "s1": s1,
            "delta": (s1 - s0),
            "Th": Th,
            "Th_opt": Th_opt,
            "Th_opt3": Th_opt3,
            "t": t,
            "Nt": Nt,
            "N": N,
        },
        do_compression=True,
    )
