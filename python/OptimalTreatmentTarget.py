# -*- coding: utf-8 -*-
# Bootstrap ATE over thresholds — Python translation of your MATLAB script

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

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
# Use the full fcnPlotKM which step-interpolates to the 0:2:168 grid (85 points)
# matching MATLAB. The inline `fcnPlotKM_py` above does NOT align to 168 — it
# returns survival at the last raw event time — which inflates s1[-1]−s0[-1].
from fcnPlotKM import fcnPlotKM

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
        # KM curves aligned to 0:2:168 (85 points) — fcnPlotKM is the parity-
        # matched translation of matlab/fcnPlotKM.m. s1[-1] is S(168).
        s0, s1, _, _ = fcnPlotKM(T_ref)
        s0_est, s1_est, _, _ = fcnPlotKM(T_est)

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

    # -----------------------------------------------------------------
    # Export optimal treatment target analysis results to text file
    # (Ported from matlab/a4_OptimalTreatmentTarget.m:57-298)
    # -----------------------------------------------------------------
    try:
        _now = datetime.now()
        _filename = f"optimal_treatment_target_results_{_now.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(_filename, 'w') as f:
            f.write('==========================================================\n')
            f.write('OPTIMAL TREATMENT TARGET ANALYSIS RESULTS FOR PAPER\n')
            f.write(f"Generated on: {_now.strftime('%d-%b-%Y %H:%M:%S')}\n")
            f.write('==========================================================\n\n')

            # Study parameters
            f.write('STUDY PARAMETERS:\n')
            f.write(f"- Sample size: {N} patients\n")
            f.write(f"- Bootstrap iterations: {Nboot}\n")
            f.write('- Study period: 168 hours\n')
            f.write(f"- Time step: {int(dt)} hours\n")
            f.write(f"- Treatment target range: {th.min():.2f} to {th.max():.2f} (50 levels)\n")
            f.write(f"- Target resolution: {th[1] - th[0]:.3f}\n")
            f.write('- Parallel processing: Yes\n\n')

            # Summary statistics across bootstrap
            A0_median = np.median(A0, axis=0)
            A1_median = np.median(A1, axis=0)
            A0_lower = np.percentile(A0, 2.5, axis=0)
            A0_upper = np.percentile(A0, 97.5, axis=0)
            A1_lower = np.percentile(A1, 2.5, axis=0)
            A1_upper = np.percentile(A1, 97.5, axis=0)

            opt_idx_est = int(np.argmax(A0_median))
            opt_idx_true = int(np.argmax(A1_median))
            max_ate_est = float(A0_median[opt_idx_est])
            max_ate_true = float(A1_median[opt_idx_true])

            f.write('OPTIMAL TREATMENT TARGETS:\n')
            f.write('Based on TRUE parameters:\n')
            f.write(f"  Optimal threshold: {th[opt_idx_true]:.3f}\n")
            f.write(f"  Maximum ATE: {max_ate_true:.3f} ({max_ate_true*100:.1f}% survival benefit)\n")
            f.write(f"  95% CI: [{A1_lower[opt_idx_true]:.3f}, {A1_upper[opt_idx_true]:.3f}]\n")

            f.write('\nBased on ESTIMATED parameters:\n')
            f.write(f"  Optimal threshold: {th[opt_idx_est]:.3f}\n")
            f.write(f"  Maximum ATE: {max_ate_est:.3f} ({max_ate_est*100:.1f}% survival benefit)\n")
            f.write(f"  95% CI: [{A0_lower[opt_idx_est]:.3f}, {A0_upper[opt_idx_est]:.3f}]\n")

            target_difference = abs(th[opt_idx_est] - th[opt_idx_true])
            ate_difference = max_ate_est - max_ate_true

            f.write('\nOPTIMAL TARGET AGREEMENT:\n')
            f.write(f"Target difference: {target_difference:.3f}\n")
            f.write(f"ATE difference: {ate_difference:.3f} ({ate_difference*100:.1f}% points)\n")

            if target_difference < 0.05:
                f.write('+ EXCELLENT agreement - targets differ by < 0.05\n')
            elif target_difference < 0.1:
                f.write('+ GOOD agreement - targets differ by < 0.10\n')
            elif target_difference < 0.2:
                f.write('!!! MODERATE agreement - targets differ by < 0.20\n')
            else:
                f.write('x POOR agreement - targets differ by >= 0.20\n')

            # Performance across target range
            f.write('\nPERFORMANCE ACROSS TARGET RANGE:\n')

            bias_median = A0_median - A1_median
            if np.std(A0_median) > 0 and np.std(A1_median) > 0:
                correlation = float(np.corrcoef(A0_median, A1_median)[0, 1])
            else:
                correlation = 0.0
            n_obs = len(A0_median)
            if abs(correlation) < 1 and n_obs > 2:
                t_stat = correlation * np.sqrt((n_obs - 2) / (1 - correlation**2))
                from math import erf, sqrt
                p_value = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
            else:
                p_value = 0.0

            f.write('Overall performance:\n')
            f.write(f"  Correlation (est vs true): {correlation:.3f} (p = {p_value:.3f})\n")
            f.write(f"  Mean bias: {np.mean(bias_median):.3f} ({np.mean(bias_median)*100:.1f}% points)\n")
            f.write(f"  RMS bias: {np.sqrt(np.mean(bias_median**2)):.3f}\n")
            max_abs_bias = float(np.max(np.abs(bias_median)))
            max_abs_bias_idx = int(np.argmax(np.abs(bias_median)))
            f.write(f"  Max absolute bias: {max_abs_bias:.3f} at th = {th[max_abs_bias_idx]:.3f}\n")

            # Key target thresholds
            key_targets = [0.05, 0.1, 0.2, 0.5]
            f.write('\nKEY TARGET THRESHOLDS ANALYSIS:\n')
            f.write(f"{'Target':<8s} {'True ATE':<12s} {'Est ATE':<12s} {'Bias':<12s} {'Est 95% CI':<15s}\n")
            f.write(f"{'------':<8s} {'--------':<12s} {'-------':<12s} {'----':<12s} {'-----------':<15s}\n")

            for target_val in key_targets:
                target_idx = int(np.argmin(np.abs(th - target_val)))
                true_ate = A1_median[target_idx]
                est_ate = A0_median[target_idx]
                bias_v = est_ate - true_ate
                ci_str = f"[{A0_lower[target_idx]:.3f},{A0_upper[target_idx]:.3f}]"
                f.write(f"{target_val:<8.2f} {true_ate:<12.3f} {est_ate:<12.3f} {bias_v:<12.3f} {ci_str:<15s}\n")

            # Bootstrap uncertainty
            f.write('\nBOOTSTRAP UNCERTAINTY ANALYSIS:\n')
            if np.mean(A0[:, opt_idx_est]) != 0:
                cv_est_optimal = np.std(A0[:, opt_idx_est], ddof=1) / abs(np.mean(A0[:, opt_idx_est])) * 100
            else:
                cv_est_optimal = float('nan')
            if np.mean(A1[:, opt_idx_true]) != 0:
                cv_true_optimal = np.std(A1[:, opt_idx_true], ddof=1) / abs(np.mean(A1[:, opt_idx_true])) * 100
            else:
                cv_true_optimal = float('nan')

            f.write('Uncertainty at optimal targets:\n')
            f.write(f"  Estimated optimal (th={th[opt_idx_est]:.3f}): CV = {cv_est_optimal:.1f}%\n")
            f.write(f"  True optimal (th={th[opt_idx_true]:.3f}): CV = {cv_true_optimal:.1f}%\n")

            proportion_est_better = float(np.mean(A0[:, opt_idx_est] > A1[:, opt_idx_true]))
            f.write(f"  Proportion where est > true at optimal: {proportion_est_better*100:.1f}%\n")

            # Target selection robustness
            est_optimal_selections = np.zeros(th.size)
            true_optimal_selections = np.zeros(th.size)
            for n in range(Nboot):
                est_optimal_selections[int(np.argmax(A0[n, :]))] += 1
                true_optimal_selections[int(np.argmax(A1[n, :]))] += 1
            est_optimal_selections = est_optimal_selections / Nboot * 100
            true_optimal_selections = true_optimal_selections / Nboot * 100

            most_freq_est_idx = int(np.argmax(est_optimal_selections))
            most_freq_true_idx = int(np.argmax(true_optimal_selections))
            max_est_freq = float(est_optimal_selections[most_freq_est_idx])
            max_true_freq = float(true_optimal_selections[most_freq_true_idx])

            f.write('\nTARGET SELECTION ROBUSTNESS:\n')
            f.write('Most frequently selected targets across bootstrap iterations:\n')
            f.write(f"  Estimated parameters: th = {th[most_freq_est_idx]:.3f} ({max_est_freq:.1f}% of iterations)\n")
            f.write(f"  True parameters: th = {th[most_freq_true_idx]:.3f} ({max_true_freq:.1f}% of iterations)\n")

            agreement_tolerance = 0.05
            agreement_count = 0
            for n in range(Nboot):
                if abs(th[int(np.argmax(A0[n, :]))] - th[int(np.argmax(A1[n, :]))]) <= agreement_tolerance:
                    agreement_count += 1
            agreement_rate = agreement_count / Nboot * 100
            f.write(f"  Target agreement rate (within {agreement_tolerance:.2f}): {agreement_rate:.1f}%\n")

            # Clinical decision zones
            f.write('\nCLINICAL DECISION ZONES:\n')

            beneficial_threshold = 0.02
            beneficial_est = int(np.sum(A0_median > beneficial_threshold))
            beneficial_true = int(np.sum(A1_median > beneficial_threshold))

            f.write(f"Beneficial treatment zones (ATE > {beneficial_threshold*100:.1f}%):\n")
            f.write(f"  True parameters: {beneficial_true}/{len(th)} targets ({beneficial_true/len(th)*100:.1f}%)\n")
            f.write(f"  Estimated parameters: {beneficial_est}/{len(th)} targets ({beneficial_est/len(th)*100:.1f}%)\n")

            harmful_threshold = -0.01
            harmful_est = int(np.sum(A0_median < harmful_threshold))
            harmful_true = int(np.sum(A1_median < harmful_threshold))

            f.write(f"Potentially harmful zones (ATE < {harmful_threshold*100:.1f}%):\n")
            f.write(f"  True parameters: {harmful_true}/{len(th)} targets ({harmful_true/len(th)*100:.1f}%)\n")
            f.write(f"  Estimated parameters: {harmful_est}/{len(th)} targets ({harmful_est/len(th)*100:.1f}%)\n")

            safe_zone_est = A0_lower > 0.01
            safe_zone_true = A1_lower > 0.01
            safe_targets_est = int(np.sum(safe_zone_est))
            safe_targets_true = int(np.sum(safe_zone_true))

            f.write('Safe operating zones (95% CI lower bound > 1%):\n')
            f.write(f"  True parameters: {safe_targets_true} targets\n")
            f.write(f"  Estimated parameters: {safe_targets_est} targets\n")

            # Clinical recommendations
            f.write('\nCLINICAL RECOMMENDATIONS:\n')
            if target_difference < 0.1 and agreement_rate > 70:
                f.write('+ PROCEED with target optimization using estimated parameters\n')
                f.write(f"  - Target agreement is excellent ({agreement_rate:.1f}% bootstrap agreement)\n")
                f.write(f"  - Recommended target: th = {th[opt_idx_est]:.3f}\n")
                if safe_targets_est > 10:
                    f.write('  - Multiple safe targets available for flexibility\n')
            elif target_difference < 0.2:
                f.write('!!! CAUTION advised for target optimization\n')
                f.write(f"  - Moderate target agreement ({agreement_rate:.1f}% bootstrap agreement)\n")
                f.write(f"  - Consider sensitivity analysis around th = {th[opt_idx_est]:.3f} +/- {target_difference:.2f}\n")
            else:
                f.write('x HIGH RISK for target optimization\n')
                f.write(f"  - Poor target agreement ({agreement_rate:.1f}% bootstrap agreement)\n")
                f.write('  - Improve parameter estimation before optimization\n')
                f.write('  - Consider conservative target selection\n')

            # Parameter estimation priorities
            f.write('\nPARAMETER ESTIMATION PRIORITIES:\n')
            if np.mean(np.abs(bias_median)) < 0.02:
                f.write('- Current parameter estimation accuracy is EXCELLENT\n')
            elif np.mean(np.abs(bias_median)) < 0.05:
                f.write('- Current parameter estimation accuracy is GOOD\n')
            else:
                f.write('- Parameter estimation accuracy needs IMPROVEMENT\n')

            f.write('- Focus on mortality hazard parameter estimation\n')
            f.write('- Bootstrap confidence intervals provide uncertainty quantification\n')
            f.write('- Validate target selection with independent data\n')

            # Technical details
            f.write('\nTECHNICAL DETAILS:\n')
            f.write('- Optimization method: Grid search with bootstrap resampling\n')
            f.write(f"- Target range: [{th.min():.2f}, {th.max():.2f}] with {th[1]-th[0]:.3f} resolution\n")
            f.write('- Bootstrap method: Resampling by patient ID\n')
            f.write('- Confidence intervals: 2.5th and 97.5th percentiles\n')
            f.write('- Endpoint: Survival probability at 168 hours\n')
            f.write('- Simulation: RCT emulation with 50% treatment probability\n')

            # Data availability
            f.write('\nDATA AVAILABILITY:\n')
            f.write('- Complete bootstrap results saved to: A01Data.mat\n')
            f.write('- A0: Bootstrap ATE estimates using estimated parameters\n')
            f.write('- A1: Bootstrap ATE estimates using true parameters\n')
            f.write('- th: Treatment target grid (50 levels)\n')
            f.write('- Confidence bands available for all targets\n')

        print(f"Optimal treatment target analysis results exported to: {_filename}")
    except Exception as _exc:
        print(f"WARNING: Failed to export optimal treatment target results text file: {_exc}")
