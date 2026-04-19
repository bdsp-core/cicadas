# -*- coding: utf-8 -*-
# %% get survival curves for 2 different scenarios — Python translation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# Reproducibility (mirrors MATLAB rng(0))
np.random.seed(0)

# ---------------------------------------------------------------------
# KM helper (stand-in for fcnPlotKM). If you have a direct translation,
# import it and swap the two calls below.
# ---------------------------------------------------------------------
def _km_curve(times: np.ndarray, events: np.ndarray):
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
# External helpers you already translated elsewhere
# ---------------------------------------------------------------------
from fcnGeneratePatientParameters import fcnGeneratePatientParameters
from fcnGetPKPD_parms_est import fcnGetPKPD_parms_est
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnBiasedAssignmentProb import fcnBiasedAssignmentProb
from fcnEstimateDeathParms import fcnEstimateDeathParms
from fcnSimulate_N_Patients import fcnSimulate_N_Patients
# If you have fcnPlotKM, you can import and use it instead of fcnPlotKM_py:
# from fcnPlotKM import fcnPlotKM

# ---------------------------------------------------------------------
# Script body (mirrors MATLAB)
# ---------------------------------------------------------------------

# clear; clc; format compact;
dt = 2.0
t = np.arange(0.0, 168.0 + dt, dt)
Nt = t.size

T0_df = pd.read_csv("trialData0.csv")
N = np.unique(T0_df["sid"].values).size

# load parmsTrue (parmsControl, parmsPD, C, g, parmsY, parmsV, parmsL, age, sofa, ke)
m = loadmat("parmsTrue.mat", squeeze_me=True, struct_as_record=False)
parmsControl = np.asarray(m["parmsControl"], dtype=float).reshape(-1)
parmsPD = np.asarray(m["parmsPD"], dtype=float).reshape(-1)
C = np.asarray(m["C"], dtype=float).reshape(-1)
g = np.asarray(m["g"], dtype=float).reshape(-1)
parmsY_true = np.asarray(m["parmsY"], dtype=float).reshape(-1)
parmsV = np.asarray(m["parmsV"], dtype=float).reshape(-1)
parmsL = np.asarray(m["parmsL"], dtype=float).reshape(-1)
age = np.asarray(m["age"], dtype=float).reshape(-1)
sofa = np.asarray(m["sofa"], dtype=float).reshape(-1)
ke = float(np.asarray(m["ke"]).reshape(())) if "ke" in m else 0.5  # default fallback

# Th = 0.1; (scalar now, as in your MATLAB)
Th = np.array([0.1], dtype=float)
A2 = np.linspace(1, 50, 20)
A3 = np.linspace(1, 50, 20)

# Preallocate heatmap outputs (match MATLAB sizes)
s0 = np.full((A2.size, A3.size), np.nan)
s1 = np.full((A2.size, A3.size), np.nan)

# make patients
age_sim, sofa_sim, C_sim, g_sim, parmsPD_sim = fcnGeneratePatientParameters(
    N, "TargetCMean", 3, "TargetGMean", 4, "CV", 0.1
)
C_est, g_est, ke_est, parmsPD_est = fcnGetPKPD_parms_est(age_sim, sofa_sim)
L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)
treatProbBiased = fcnBiasedAssignmentProb(age_sim, sofa_sim, L0[:, :5])
T0_for_Y = pd.read_csv("trialData0.csv")
parmsY_est = fcnEstimateDeathParms(T0_for_Y)

nTrials = 1  # Number of trials to take median over

for ii in range(A2.size):
    for jj in range(A3.size):
        print("*****************")
        print([A2[ii], A3[jj]])
        print("*****************")

        # Twiddle parmsY and parmsY_est (elements 3 and 4 in MATLAB are 1-based)
        parmsY = np.array([-7.0, 0.3, A2[ii], A3[jj]], dtype=float)
        parmsY_est_mod = np.asarray(parmsY_est, dtype=float).copy()
        if parmsY_est_mod.size >= 4:
            parmsY_est_mod[2] = A2[ii]
            parmsY_est_mod[3] = A3[jj]

        # Arrays across trials (|Th| x nTrials); also used below (prealloc fix)
        S0_trials = np.full((Th.size, nTrials), np.nan)
        S1_trials = np.full((Th.size, nTrials), np.nan)

        for trial in range(nTrials):
            for kk, th in enumerate(Th):
                # Ideal data (RCT=1) with TRUE parms
                RCT = 1
                treatProb = np.full(N, 0.5)
                T1 = fcnSimulate_N_Patients(
                    N, RCT, treatProb, th, C, g, ke, L0, parmsControl, parmsY, parmsV, age, sofa
                )

                # Observational data (RCT=0) — not used directly for KM here
                RCT = 0
                T0_obs = fcnSimulate_N_Patients(
                    N, RCT, treatProbBiased, th, C, g, ke, L0, parmsControl, parmsY, parmsV, age, sofa
                )

                # Simulated RCT using estimated models (no censoring)
                RCT = 1
                treatProb = np.full(N, 0.5)
                parmsV_est = np.zeros(6, dtype=float)
                T1_est = fcnSimulate_N_Patients(
                    N, RCT, treatProb, th, C_est, g_est, ke_est, L0, parmsControl, parmsY_est_mod, parmsV_est, age, sofa
                )

                # s0_est, s1_est from T1_est
                # If you have fcnPlotKM, use it; else use KM helper:
                # s0_est, s1_est, _, _ = fcnPlotKM(T1_est)
                s0_est, s1_est, _, _ = fcnPlotKM_py(T1_est)

                S1_trials[kk, trial] = s1_est[-1]
                S0_trials[kk, trial] = s0_est[-1]

        # Median across trials (still just one trial unless you increase nTrials)
        S0 = np.nanmedian(S0_trials, axis=1)
        S1 = np.nanmedian(S1_trials, axis=1)

        # find optimal ATE (argmax over Th)
        diff = S1 - S0
        jj_star = int(np.nanargmax(diff))
        th_opt = Th[jj_star]

        # Store for heatmaps
        # Note: MATLAB uses s1(i,j) with imagesc(A2, A3, s1') and axis xy
        # We replicate by plotting s1.T with extent and origin='lower'
        s0[ii, jj] = S0[jj_star]
        s1[ii, jj] = S1[jj_star]

        # Live heatmap update (matches MATLAB three subplots)
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

    Th_opt3 = np.full_like(s1, th_opt)  # store last-threshold snapshot like MATLAB line

# Save like MATLAB "save HeatMapAggressive"
savemat(
    "HeatMapAggressive.mat",
    {
        "A2": A2,
        "A3": A3,
        "s0": s0,
        "s1": s1,
        "delta": (s1 - s0),
        "Th": Th,
        "Th_opt3": Th_opt3,
        "t": t,
        "Nt": Nt,
        "N": N,
    },
    do_compression=True,
)
