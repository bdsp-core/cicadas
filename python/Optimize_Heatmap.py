# -*- coding: utf-8 -*-
# get survival curves across (A2, A3, Th) grid — Python translation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat
from datetime import datetime

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
# Use the proper fcnPlotKM (0:2:168 grid, 85 points) matching MATLAB — the
# inline fcnPlotKM_py above returns survival at raw event times, not S(168),
# which inflates s1[-1]-s0[-1] ATE estimates.
from fcnPlotKM import fcnPlotKM

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
                s0_est, s1_est, _, _ = fcnPlotKM(T1_est)
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

# ---------------------------------------------------------------------
# Export optimization heatmap analysis results to text file for paper
# (Ported from matlab/a4_Optimize_Heatmap.m:131-413)
# ---------------------------------------------------------------------
try:
    _now = datetime.now()
    _filename = f"optimization_heatmap_results_{_now.strftime('%Y%m%d_%H%M%S')}.txt"

    with open(_filename, 'w') as f:
        f.write('==========================================================\n')
        f.write('OPTIMIZATION HEATMAP ANALYSIS RESULTS FOR PAPER\n')
        f.write(f"Generated on: {_now.strftime('%d-%b-%Y %H:%M:%S')}\n")
        f.write('==========================================================\n\n')

        # Study parameters
        f.write('STUDY PARAMETERS:\n')
        f.write(f"- Sample size: {N} patients\n")
        f.write(f"- Number of trials per parameter set: {nTrials}\n")
        f.write('- Study period: 168 hours\n')
        f.write(f"- Time step: {int(dt)} hours\n")
        f.write(f"- PI controller gain: {ki:.1f}\n")
        f.write(f"- Maximum pump rate: {Amax:.1f}\n")
        f.write(f"- Elimination constant: {ke:.2f}\n\n")

        # Parameter optimization ranges
        f.write('PARAMETER OPTIMIZATION RANGES:\n')
        f.write(f"- Treatment thresholds: {Th.min():.2f} to {Th.max():.2f} (20 levels)\n")
        f.write(f"- Disease harm (a2): {A2.min():.1f} to {A2.max():.1f} (20 levels)\n")
        f.write(f"- Treatment harm (a3): {A3.min():.1f} to {A3.max():.1f} (20 levels)\n")
        f.write(f"- Total optimization combinations: {A2.size * A3.size * Th.size}\n\n")

        # Baseline mortality params
        f.write('BASELINE MORTALITY PARAMETERS:\n')
        f.write(f"- Baseline hazard (a0): {a0:.1f}\n")
        f.write(f"- Time effect (a1): {a1:.2f}\n")
        f.write(f"- Disease harm varies: {A2.min():.1f} to {A2.max():.1f}\n")
        f.write(f"- Treatment harm varies: {A3.min():.1f} to {A3.max():.1f}\n\n")

        # Survival outcome analysis
        f.write('SURVIVAL OUTCOME ANALYSIS:\n')
        valid_s0 = s0[~np.isnan(s0)]
        valid_s1 = s1[~np.isnan(s1)]
        ate_matrix = s1 - s0
        valid_ate = ate_matrix[~np.isnan(ate_matrix)]
        valid_th_opt = Th_opt[~np.isnan(Th_opt)]

        f.write('Optimized untreated survival (s0):\n')
        f.write(f"  Range: {valid_s0.min()*100:.1f}% to {valid_s0.max()*100:.1f}%\n")
        f.write(f"  Mean: {valid_s0.mean()*100:.1f}% +/- {valid_s0.std(ddof=1)*100:.1f}%\n")

        f.write('Optimized treated survival (s1):\n')
        f.write(f"  Range: {valid_s1.min()*100:.1f}% to {valid_s1.max()*100:.1f}%\n")
        f.write(f"  Mean: {valid_s1.mean()*100:.1f}% +/- {valid_s1.std(ddof=1)*100:.1f}%\n")

        f.write('Optimized treatment effect (ATE):\n')
        f.write(f"  Range: {valid_ate.min()*100:.1f}% to {valid_ate.max()*100:.1f}%\n")
        f.write(f"  Mean: {valid_ate.mean()*100:.1f}% +/- {valid_ate.std(ddof=1)*100:.1f}%\n")

        f.write('Optimal treatment thresholds:\n')
        f.write(f"  Range: {valid_th_opt.min():.3f} to {valid_th_opt.max():.3f}\n")
        f.write(f"  Mean: {valid_th_opt.mean():.3f} +/- {valid_th_opt.std(ddof=1):.3f}\n")

        # Find global optimum
        max_ate = float(valid_ate.max())
        max_i, max_j = np.where(ate_matrix == max_ate)

        f.write('\nGLOBAL OPTIMUM:\n')
        if max_i.size > 0:
            i0, j0 = int(max_i[0]), int(max_j[0])
            f.write('Best parameter combination:\n')
            f.write(f"  Disease harm (a2): {A2[i0]:.1f}\n")
            f.write(f"  Treatment harm (a3): {A3[j0]:.1f}\n")
            f.write(f"  Optimal threshold: {Th_opt[i0, j0]:.3f}\n")
            f.write(f"  Maximum ATE: {max_ate*100:.1f}% ({s1[i0, j0]*100:.1f}% vs {s0[i0, j0]*100:.1f}% survival)\n")

        # Worst case
        min_ate = float(valid_ate.min())
        min_i, min_j = np.where(ate_matrix == min_ate)

        f.write('\nWORST CASE SCENARIO:\n')
        if min_i.size > 0:
            i0, j0 = int(min_i[0]), int(min_j[0])
            f.write('Worst parameter combination:\n')
            f.write(f"  Disease harm (a2): {A2[i0]:.1f}\n")
            f.write(f"  Treatment harm (a3): {A3[j0]:.1f}\n")
            f.write(f"  Optimal threshold: {Th_opt[i0, j0]:.3f}\n")
            f.write(f"  Minimum ATE: {min_ate*100:.1f}% ({s1[i0, j0]*100:.1f}% vs {s0[i0, j0]*100:.1f}% survival)\n")

        # Treatment optimization effectiveness
        f.write('\nTREATMENT OPTIMIZATION EFFECTIVENESS:\n')
        beneficial_scenarios = int(np.sum(valid_ate > 0))
        harmful_scenarios = int(np.sum(valid_ate < 0))
        neutral_scenarios = int(np.sum(np.abs(valid_ate) < 0.01))
        total_scenarios = int(valid_ate.size)

        f.write('After optimization, treatment is:\n')
        f.write(f"  Beneficial (ATE > 0): {beneficial_scenarios}/{total_scenarios} scenarios ({beneficial_scenarios/total_scenarios*100:.1f}%)\n")
        f.write(f"  Harmful (ATE < 0): {harmful_scenarios}/{total_scenarios} scenarios ({harmful_scenarios/total_scenarios*100:.1f}%)\n")
        f.write(f"  Neutral (|ATE| < 1%): {neutral_scenarios}/{total_scenarios} scenarios ({neutral_scenarios/total_scenarios*100:.1f}%)\n")

        # High-impact optimization zones
        high_benefit_threshold = float(np.percentile(valid_ate, 75))
        high_benefit_scenarios = int(np.sum(valid_ate >= high_benefit_threshold))

        f.write('\nHIGH-IMPACT OPTIMIZATION ZONES:\n')
        f.write('High benefit zone (top 25% after optimization):\n')
        f.write(f"  ATE threshold: >= {high_benefit_threshold*100:.1f}%\n")
        f.write(f"  Number of scenarios: {high_benefit_scenarios} ({high_benefit_scenarios/total_scenarios*100:.1f}%)\n")

        # Optimal threshold distribution
        f.write('\nOPTIMAL THRESHOLD DISTRIBUTION:\n')
        threshold_bins = np.array([0.01, 0.1, 0.2, 0.5, 1.0, 1.1])
        threshold_counts, _ = np.histogram(valid_th_opt, bins=threshold_bins)
        total_counts = threshold_counts.sum()
        threshold_percentages = (threshold_counts / total_counts * 100) if total_counts > 0 else np.zeros_like(threshold_counts, dtype=float)

        f.write('Optimal threshold preferences:\n')
        for i in range(len(threshold_counts)):
            range_str = f"{threshold_bins[i]:.2f}-{threshold_bins[i+1]:.2f}"
            f.write(f"  {range_str}: {int(threshold_counts[i])} scenarios ({threshold_percentages[i]:.1f}%)\n")

        mode_idx = int(np.argmax(threshold_counts))
        f.write(f"Most common threshold range: {threshold_bins[mode_idx]:.2f}-{threshold_bins[mode_idx+1]:.2f} ({threshold_percentages[mode_idx]:.1f}% of scenarios)\n")

        # Parameter sensitivity for optimization
        disease_th_corr = 0.0
        treatment_th_corr = 0.0
        disease_ate_corr = 0.0
        treatment_ate_corr = 0.0
        if A2.size > 1 and A3.size > 1:
            f.write('\nPARAMETER SENSITIVITY FOR OPTIMIZATION:\n')

            mean_th_by_disease = np.nanmean(Th_opt, axis=1)
            mean_th_by_treatment = np.nanmean(Th_opt, axis=0)

            def _corr_nan(x, y):
                x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 2 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
                    return 0.0, float('nan')
                r = float(np.corrcoef(x[mask], y[mask])[0, 1])
                n = int(mask.sum())
                if abs(r) >= 1 or n <= 2:
                    p = 0.0 if abs(r) >= 1 else 1.0
                else:
                    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                    from math import erf, sqrt
                    p = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
                return r, p

            disease_th_corr, disease_th_p = _corr_nan(A2, mean_th_by_disease)
            treatment_th_corr, treatment_th_p = _corr_nan(A3, mean_th_by_treatment)

            f.write('Disease harm (a2) vs Optimal Threshold:\n')
            f.write(f"  Correlation: {disease_th_corr:.3f} (p = {disease_th_p:.3f})\n")
            if disease_th_corr > 0.5:
                f.write('  -> Higher disease harm requires HIGHER optimal thresholds\n')
            elif disease_th_corr < -0.5:
                f.write('  -> Higher disease harm requires LOWER optimal thresholds\n')
            else:
                f.write('  -> Disease harm has moderate effect on optimal thresholds\n')

            f.write('Treatment harm (a3) vs Optimal Threshold:\n')
            f.write(f"  Correlation: {treatment_th_corr:.3f} (p = {treatment_th_p:.3f})\n")
            if treatment_th_corr > 0.5:
                f.write('  -> Higher treatment harm requires HIGHER optimal thresholds\n')
            elif treatment_th_corr < -0.5:
                f.write('  -> Higher treatment harm requires LOWER optimal thresholds\n')
            else:
                f.write('  -> Treatment harm has moderate effect on optimal thresholds\n')

            mean_ate_by_disease = np.nanmean(ate_matrix, axis=1)
            mean_ate_by_treatment = np.nanmean(ate_matrix, axis=0)

            disease_ate_corr, disease_ate_p = _corr_nan(A2, mean_ate_by_disease)
            treatment_ate_corr, treatment_ate_p = _corr_nan(A3, mean_ate_by_treatment)

            f.write('\nDisease harm (a2) vs Optimized ATE:\n')
            f.write(f"  Correlation: {disease_ate_corr:.3f} (p = {disease_ate_p:.3f})\n")
            f.write('Treatment harm (a3) vs Optimized ATE:\n')
            f.write(f"  Correlation: {treatment_ate_corr:.3f} (p = {treatment_ate_p:.3f})\n")

        # Optimization robustness
        f.write('\nOPTIMIZATION ROBUSTNESS:\n')
        if valid_th_opt.mean() != 0:
            cv_thresholds = valid_th_opt.std(ddof=1) / valid_th_opt.mean() * 100
        else:
            cv_thresholds = float('nan')
        f.write(f"Optimal threshold variability: CV = {cv_thresholds:.1f}%\n")

        if cv_thresholds < 20:
            f.write('  -> Optimal thresholds are CONSISTENT across parameter space\n')
        elif cv_thresholds < 50:
            f.write('  -> Optimal thresholds show MODERATE variability\n')
        else:
            f.write('  -> Optimal thresholds are HIGHLY variable across parameters\n')

        # Safe optimization zones
        conservative_benefit_threshold = 0.03
        safe_optimization_scenarios = int(np.sum(valid_ate > conservative_benefit_threshold))

        f.write('\nSAFE OPTIMIZATION ZONES:\n')
        f.write('Conservative benefit threshold (ATE > 3%):\n')
        f.write(f"  Safe scenarios: {safe_optimization_scenarios}/{total_scenarios} ({safe_optimization_scenarios/total_scenarios*100:.1f}%)\n")

        # Clinical decision making
        f.write('\nCLINICAL DECISION MAKING:\n')

        if safe_optimization_scenarios / total_scenarios > 0.8:
            f.write('+ OPTIMIZATION is broadly beneficial across parameter space\n')
            f.write('  - Aggressive optimization strategies are safe\n')
            f.write('  - Standard threshold optimization protocols recommended\n')
        elif beneficial_scenarios / total_scenarios > 0.7:
            f.write('!!! OPTIMIZATION benefits are parameter-dependent\n')
            f.write('  - Careful parameter estimation required before optimization\n')
            f.write('  - Consider adaptive optimization protocols\n')
        else:
            f.write('x HIGH RISK for optimization without accurate parameters\n')
            f.write('  - Conservative approaches recommended\n')
            f.write('  - Extensive parameter validation required\n')

        # Parameter estimation priorities
        f.write('\nPARAMETER ESTIMATION PRIORITIES FOR OPTIMIZATION:\n')
        if abs(disease_th_corr) > abs(treatment_th_corr):
            f.write('1. Disease harm parameters (a2) - highest impact on thresholds\n')
            f.write('2. Treatment harm parameters (a3)\n')
        else:
            f.write('1. Treatment harm parameters (a3) - highest impact on thresholds\n')
            f.write('2. Disease harm parameters (a2)\n')

        if abs(disease_ate_corr) > abs(treatment_ate_corr):
            f.write('3. Disease harm also most critical for ATE optimization\n')
        else:
            f.write('3. Treatment harm also most critical for ATE optimization\n')

        # Practical recommendations
        f.write('\nPRACTICAL OPTIMIZATION RECOMMENDATIONS:\n')

        if A2.size > 1 and A3.size > 1:
            mean_ate_val = float(valid_ate.mean())
            robust_optimization_mask = ate_matrix > mean_ate_val
            robust_optimization_count = int(np.sum(robust_optimization_mask))
            total_cells = int(ate_matrix.size)

            f.write('Robust optimization regions (above-average ATE):\n')
            f.write(f"  Count: {robust_optimization_count}/{total_cells} scenarios ({robust_optimization_count/total_cells*100:.1f}%)\n")

        f.write('\nFor clinical implementation:\n')
        mean_th_opt = float(valid_th_opt.mean())
        if mean_th_opt < 0.2:
            f.write('- Favor AGGRESSIVE optimization (low thresholds)\n')
        elif mean_th_opt > 0.5:
            f.write('- Favor CONSERVATIVE optimization (high thresholds)\n')
        else:
            f.write('- Use MODERATE optimization thresholds\n')

        p25 = float(np.percentile(valid_th_opt, 25))
        p75 = float(np.percentile(valid_th_opt, 75))
        f.write(f"- Recommended threshold range: {p25:.3f} to {p75:.3f}\n")
        f.write('- Monitor treatment response and adjust adaptively\n')
        f.write('- Validate optimization performance with real-world data\n')

        # Technical details
        f.write('\nTECHNICAL DETAILS:\n')
        f.write('- Optimization method: Exhaustive grid search\n')
        f.write('- Grid resolution: 20 x 20 x 20 (parameter x parameter x threshold)\n')
        f.write(f"- Total evaluations: {A2.size * A3.size * Th.size}\n")
        f.write('- Optimization criterion: Maximum ATE at 168 hours\n')
        f.write('- Simulation: RCT emulation with observational parameter estimation\n')
        f.write(f"- Aggregation: Median across {nTrials} trial(s) per combination\n")

        # Data availability
        f.write('\nDATA AVAILABILITY:\n')
        f.write('- Complete optimization results saved to: HeatMapData.mat\n')
        f.write('- s0: Optimized untreated survival matrix (20x20)\n')
        f.write('- s1: Optimized treated survival matrix (20x20)\n')
        f.write('- Th_opt: Optimal threshold matrix (20x20)\n')
        f.write('- Parameter grids: A2 (disease harm), A3 (treatment harm), Th (thresholds)\n')

    print(f"Optimization heatmap analysis results exported to: {_filename}")
except Exception as _exc:
    print(f"WARNING: Failed to export optimization heatmap results text file: {_exc}")
