# -*- coding: utf-8 -*-
# %% get 3 curves — Python translation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# ---------------------------------------------------------------------
# KM helpers (to stand in for fcnPlotKM)
# ---------------------------------------------------------------------
def _km_curve(times: np.ndarray, events: np.ndarray):
    """Kaplan–Meier S(t) step function; returns (t_out, S_out) starting at (0,1)."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    order = np.argsort(times, kind="mergesort")
    times, events = times[order], events[order]
    uniq_event_times = np.unique(times[events == 1])
    if uniq_event_times.size == 0:
        return np.array([0.0]), np.array([1.0])
    n = len(times)
    at_risk = n
    S = 1.0
    t_out = [0.0]
    S_out = [1.0]
    idx = 0
    for te in uniq_event_times:
        while idx < n and times[idx] < te:
            at_risk -= 1
            idx += 1
        d_i = np.sum((times == te) & (events == 1))
        if at_risk > 0:
            S *= (1.0 - d_i / at_risk)
        t_out.append(te); S_out.append(S)
        k = 0
        while idx + k < n and times[idx + k] == te:
            k += 1
        at_risk -= k
        idx += k
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
    """Return (s0, s1, t0, t1) for untreated/treated groups via KM."""
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
# External helpers (translated elsewhere)
# ---------------------------------------------------------------------
from fcnEstimateDeathParms import fcnEstimateDeathParms
from fcnEstimateParmsL import fcnEstimateParmsL
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnSimulate_N_Patients import fcnSimulate_N_Patients
# If you have a direct translation of fcnPlotKM, you can import and use it instead:
# from fcnPlotKM import fcnPlotKM  # returns (s0, s1, t0, t1)

# ---------------------------------------------------------------------
# Script body (mirrors MATLAB)
# ---------------------------------------------------------------------
# clear all; clc; format compact;
plt.figure(3); plt.clf()

dt = 2.0
t = np.arange(0.0, 168.0 + dt, dt)  # 0:dt:168
Nt = t.size

T0 = pd.read_csv("trialData0.csv")
N = np.unique(T0["sid"].values).size

# load parmsTrue
m = loadmat("parmsTrue.mat", squeeze_me=True, struct_as_record=False)
parmsControl = np.asarray(m["parmsControl"], dtype=float).reshape(-1)
parmsPD = np.asarray(m["parmsPD"], dtype=float).reshape(-1)
C = np.asarray(m["C"], dtype=float).reshape(-1)
g = np.asarray(m["g"], dtype=float).reshape(-1)
parmsY = np.asarray(m["parmsY"], dtype=float).reshape(-1)
parmsV = np.asarray(m["parmsV"], dtype=float).reshape(-1)
parmsL = np.asarray(m["parmsL"], dtype=float).reshape(-1)
age = np.asarray(m["age"], dtype=float).reshape(-1)
sofa = np.asarray(m["sofa"], dtype=float).reshape(-1)
ke = float(np.asarray(m["ke"]).reshape(())) if "ke" in m else np.nan

th = np.array([0.0, 0.02, 0.1, 0.8], dtype=float)

# Estimate parameters
parmsY_est = fcnEstimateDeathParms(T0)
parmsL_est, LL, AA, patient_age, patient_sofa, t_local = fcnEstimateParmsL(T0)

# Setup for RCT simulation
RCT = 1
treatProb = np.full(N, 0.5)
parmsV_est = np.zeros(6, dtype=float)
L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)

# Local arrays
ATEest = np.zeros(len(th), dtype=float)
ATEref = np.zeros(len(th), dtype=float)
S1est = np.zeros((len(th), Nt), dtype=float)
S0est = np.zeros((len(th), Nt), dtype=float)
S1ref = np.zeros((len(th), Nt), dtype=float)
S0ref = np.zeros((len(th), Nt), dtype=float)

for i, thi in enumerate(th):
    T_ref = fcnSimulate_N_Patients(N, RCT, treatProb, thi, C, g, ke, L0, parmsControl, parmsY, parmsV_est, age, sofa)
    T_est = fcnSimulate_N_Patients(N, RCT, treatProb, thi, C, g, ke, L0, parmsControl, parmsY_est, parmsV_est, age, sofa)

    # s0,s1 from ref; s0_est,s1_est from est
    # If you have fcnPlotKM, call it; otherwise use fcnPlotKM_py:
    # s0, s1, _, _ = fcnPlotKM(T_ref); s0_est, s1_est, _, _ = fcnPlotKM(T_est)
    s0, s1, _, _ = fcnPlotKM_py(T_ref)
    s0_est, s1_est, _, _ = fcnPlotKM_py(T_est)

    # Align to common grid length Nt (defensive; KM returns step times)
    def _pad_to_nt(s):
        s = np.asarray(s, dtype=float).reshape(-1)
        if s.size >= Nt:
            return s[:Nt]
        return np.pad(s, (0, Nt - s.size), constant_values=np.nan)

    s0g = _pad_to_nt(s0); s1g = _pad_to_nt(s1)
    s0eg = _pad_to_nt(s0_est); s1eg = _pad_to_nt(s1_est)

    ATEest[i] = s1eg[~np.isnan(s1eg)][-1] - s0eg[~np.isnan(s0eg)][-1]
    ATEref[i] = s1g[~np.isnan(s1g)][-1] - s0g[~np.isnan(s0g)][-1]

    S1est[i, :] = s1eg
    S0est[i, :] = s0eg
    S1ref[i, :] = s1g
    S0ref[i, :] = s0g

# Save like MATLAB "save ThreeCurves"
savemat(
    "ThreeCurves.mat",
    {
        "th": th,
        "ATEest": ATEest,
        "ATEref": ATEref,
        "S1est": S1est,
        "S0est": S0est,
        "S1ref": S1ref,
        "S0ref": S0ref,
        "t": t,
        "Nt": Nt,
    },
    do_compression=True,
)

# %%
# Plot (fixed minor indexing typo from MATLAB: S0ref/S0est should use [i, :])
plt.figure(1); plt.clf()
for i in range(4):
    ax = plt.subplot(4, 1, i + 1)
    ax.plot(t, S1ref[i, :], 'r--', label='Treated (True)')
    ax.plot(t, S0ref[i, :], 'b--', label='Untreated (True)')
    ax.plot(t, S1est[i, :], 'r', label='Treated (Est)')
    ax.plot(t, S0est[i, :], 'b', label='Untreated (Est)')
    ax.set_xlim(0, 168); ax.set_ylim(0, 1)
    ax.grid(True)
    if i == 0:
        ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
# plt.show()  # optional
