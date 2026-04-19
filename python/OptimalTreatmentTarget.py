# -*- coding: utf-8 -*-
# Bootstrap ATE over thresholds — Python translation of your MATLAB script

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------
# Reproducibility (mirrors MATLAB rng(0) concept for the driver)
# ---------------------------------------------------------------------
np.random.seed(0)

# ---------------------------------------------------------------------
# Helpers: KM and bootstrap-by-sid (stand-ins for fcnPlotKM / fcn_bootstrapBySID)
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

def fcn_bootstrapBySID_py(T0: pd.DataFrame, N: int, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap by patient (sid) with replacement; reindex to 1..N."""
    orig_ids = np.unique(T0["sid"].values)
    sampled_ids = rng.choice(orig_ids, size=N, replace=True)
    frames = []
    for new_sid, old_sid in enumerate(sampled_ids, start=1):
        df = T0[T0["sid"] == old_sid].copy()
        df["sid"] = new_sid
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------------
# External helpers (your translations)
# ---------------------------------------------------------------------
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnEstimateDeathParms import fcnEstimateDeathParms
from fcnSimulate_N_Patients import fcnSimulate_N_Patients
# If you have fcnPlotKM and fcn_bootstrapBySID in Python, you can import and swap them in.

# ---------------------------------------------------------------------
# Worker for parallel bootstrap
# ---------------------------------------------------------------------
def _bootstrap_worker(
    seed: int,
    T0_csv_path: str,
    N: int,
    th_vec: np.ndarray,
    t: np.ndarray,
    L0: np.ndarray,
    parmsControl: np.ndarray,
    parmsY_true: np.ndarray,
    age: np.ndarray,
    sofa: np.ndarray,
    C: np.ndarray,
    g: np.ndarray,
    ke: float,
):
    rng = np.random.default_rng(seed)
    T0 = pd.read_csv(T0_csv_path)
    # Bootstrap by sid
    T0_boot = fcn_bootstrapBySID_py(T0, N, rng)
    # Estimate Y parameters on bootstrap sample
    parmsY_est = fcnEstimateDeathParms(T0_boot)

    RCT = 1
    treatProb = np.full(N, 0.5)
    parmsV_est = np.zeros(6, dtype=float)

    ATEest = np.zeros(th_vec.size, dtype=float)
    ATEref = np.zeros(th_vec.size, dtype=float)

    for i, th in enumerate(th_vec):
        T_ref = fcnSimulate_N_Patients(
            N, RCT, treatProb, th, C, g, ke, L0, parmsControl, parmsY_true, parmsV_est, age, sofa
        )
        T_est = fcnSimulate_N_Patients(
            N, RCT, treatProb, th, C, g, ke, L0, parmsControl, parmsY_est, parmsV_est, age, sofa
        )
        # KM curves (using inline helper)
        s0, s1, _, _ = fcnPlotKM_py(T_ref)
        s0_est, s1_est, _, _ = fcnPlotKM_py(T_est)

        ATEest[i] = s1_est[-1] - s0_est[-1]
        ATEref[i] = s1[-1] - s0[-1]

    return ATEest, ATEref

# ---------------------------------------------------------------------
# Main (mirrors the MATLAB script)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # clear; clc; format compact;
    dt = 2.0
    t = np.arange(0.0, 168.0 + dt, dt)
    Nt = t.size

    T0 = pd.read_csv("trialData0.csv")
    N = np.unique(T0["sid"].values).size

    # load parmsTrue (expects keys used below)
    m = loadmat("parmsTrue.mat", squeeze_me=True, struct_as_record=False)
    parmsControl = np.asarray(m["parmsControl"], dtype=float).reshape(-1)
    parmsPD = np.asarray(m["parmsPD"], dtype=float).reshape(-1)  # not used directly here
    C = np.asarray(m["C"], dtype=float).reshape(-1)
    g = np.asarray(m["g"], dtype=float).reshape(-1)
    parmsY = np.asarray(m["parmsY"], dtype=float).reshape(-1)
    parmsV = np.asarray(m["parmsV"], dtype=float).reshape(-1)    # not used directly here
    parmsL = np.asarray(m["parmsL"], dtype=float).reshape(-1)
    age = np.asarray(m["age"], dtype=float).reshape(-1)
    sofa = np.asarray(m["sofa"], dtype=float).reshape(-1)
    ke = float(np.asarray(m["ke"]).reshape(())) if "ke" in m else 0.5

    Nboot = 1000
    th = np.linspace(0.0, 1.0, 50)

    # Pre-allocate results
    A0 = np.zeros((Nboot, th.size), dtype=float)  # ATEest
    A1 = np.zeros((Nboot, th.size), dtype=float)  # ATEref

    # Generate L0 once (as in MATLAB)
    L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)

    # Parallel bootstrap loop (uses all cores by default)
    num_workers = os.cpu_count() or 1
    seeds = np.random.SeedSequence(0).spawn(Nboot)
    seed_ints = [int(s.entropy % (2**32 - 1)) for s in seeds]  # simple int seeds

    print(f"Starting parallel bootstrap with {num_workers} workers and {Nboot} iterations...")
    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for n in range(Nboot):
            futures.append(
                ex.submit(
                    _bootstrap_worker,
                    seed_ints[n],
                    "trialData0.csv",  # pass path to avoid DataFrame pickling overhead
                    N,
                    th,
                    t,
                    L0,
                    parmsControl,
                    parmsY,
                    age,
                    sofa,
                    C,
                    g,
                    ke,
                )
            )

        for idx, fut in enumerate(as_completed(futures), start=1):
            ATEest, ATEref = fut.result()
            A0[idx - 1, :] = ATEest
            A1[idx - 1, :] = ATEref
            print(f"Completed bootstrap {idx}")

    # Plot final results (many lines, like MATLAB plot(th, A0', 'k', th, A1', 'r'))
    plt.figure(1); plt.clf()
    for r in range(A0.shape[0]):
        plt.plot(th, A0[r, :], 'k', alpha=0.2)
    for r in range(A1.shape[0]):
        plt.plot(th, A1[r, :], 'r', alpha=0.2)
    plt.xlabel("Threshold (th)")
    plt.ylabel("ATE at end of trial")
    plt.grid(True)
    plt.title("Bootstrap ATE curves across thresholds")
    # plt.show()  # optional

    # Save like MATLAB "save A01Data"
    savemat(
        "A01Data.mat",
        {
            "th": th,
            "A0": A0,  # ATEest (estimated Y)
            "A1": A1,  # ATEref (true Y)
            "Nboot": Nboot,
            "Nt": Nt,
        },
        do_compression=True,
    )
