# -*- coding: utf-8 -*-
# %% get survival curves for 2 different scenarios — Python translation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import datetime

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
# Use the proper fcnPlotKM (0:2:168 grid, 85 points) matching MATLAB — the
# inline fcnPlotKM_py above returns survival at raw event times, not S(168),
# which inflates s1[-1]-s0[-1] ATE estimates.
from fcnPlotKM import fcnPlotKM

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
                s0_est, s1_est, _, _ = fcnPlotKM(T1_est)

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

# ---------------------------------------------------------------------
# Export heatmap sensitivity analysis results to text file for paper
# (Ported from matlab/a4_HeatMap_Agressive.m:134-372)
# ---------------------------------------------------------------------
try:
    _now = datetime.now()
    _filename = f"heatmap_sensitivity_results_{_now.strftime('%Y%m%d_%H%M%S')}.txt"

    # Baseline mortality params (MATLAB hard-coded same values)
    _a0, _a1 = -7.0, 0.3
    _ki = 10.0
    _Amax = 50.0
    _dt = int(dt)
    # Use last target threshold value from the Th sweep
    _th_last = float(Th[-1])

    with open(_filename, 'w') as f:
        f.write('==========================================================\n')
        f.write('HEATMAP SENSITIVITY ANALYSIS RESULTS FOR PAPER\n')
        f.write(f"Generated on: {_now.strftime('%d-%b-%Y %H:%M:%S')}\n")
        f.write('==========================================================\n\n')

        # Study parameters
        f.write('STUDY PARAMETERS:\n')
        f.write(f"- Sample size: {N} patients\n")
        f.write(f"- Number of trials per parameter set: {nTrials}\n")
        f.write('- Study period: 168 hours\n')
        f.write(f"- Time step: {_dt} hours\n")
        f.write(f"- Control target threshold: {_th_last:.2f}\n")
        f.write(f"- PI controller gain: {_ki:.1f}\n")
        f.write(f"- Maximum pump rate: {_Amax:.1f}\n")
        f.write(f"- Elimination constant: {ke:.2f}\n\n")

        # Parameter ranges
        f.write('PARAMETER SENSITIVITY RANGES:\n')
        f.write(f"- Harm from disease (a2): {A2.min():.1f} to {A2.max():.1f} (20 levels)\n")
        f.write(f"- Harm from treatment (a3): {A3.min():.1f} to {A3.max():.1f} (20 levels)\n")
        f.write(f"- Total parameter combinations: {A2.size * A3.size}\n\n")

        # Baseline mortality parameters
        f.write('BASELINE MORTALITY PARAMETERS:\n')
        f.write(f"- Baseline hazard (a0): {_a0:.1f}\n")
        f.write(f"- Time effect (a1): {_a1:.2f}\n")
        f.write(f"- Disease harm varies: {A2.min():.1f} to {A2.max():.1f}\n")
        f.write(f"- Treatment harm varies: {A3.min():.1f} to {A3.max():.1f}\n\n")

        # Survival outcome ranges
        f.write('SURVIVAL OUTCOME RANGES:\n')
        valid_s0 = s0[~np.isnan(s0)]
        valid_s1 = s1[~np.isnan(s1)]
        ate_matrix = s1 - s0
        valid_ate = ate_matrix[~np.isnan(ate_matrix)]

        f.write('Untreated survival (s0):\n')
        f.write(f"  Range: {valid_s0.min()*100:.1f}% to {valid_s0.max()*100:.1f}%\n")
        f.write(f"  Mean: {valid_s0.mean()*100:.1f}% +/- {valid_s0.std(ddof=1)*100:.1f}%\n")

        f.write('Treated survival (s1):\n')
        f.write(f"  Range: {valid_s1.min()*100:.1f}% to {valid_s1.max()*100:.1f}%\n")
        f.write(f"  Mean: {valid_s1.mean()*100:.1f}% +/- {valid_s1.std(ddof=1)*100:.1f}%\n")

        f.write('Treatment effect (ATE):\n')
        f.write(f"  Range: {valid_ate.min()*100:.1f}% to {valid_ate.max()*100:.1f}%\n")
        f.write(f"  Mean: {valid_ate.mean()*100:.1f}% +/- {valid_ate.std(ddof=1)*100:.1f}%\n")

        # Optimal treatment zones
        max_ate = float(valid_ate.max())
        min_ate = float(valid_ate.min())

        opt_i, opt_j = np.where(ate_matrix == max_ate)
        worst_i, worst_j = np.where(ate_matrix == min_ate)

        f.write('\nOPTIMAL TREATMENT ZONES:\n')
        f.write('Best treatment scenario:\n')
        if opt_i.size > 0:
            i0, j0 = int(opt_i[0]), int(opt_j[0])
            f.write(f"  Parameters: Disease harm = {A2[i0]:.1f}, Treatment harm = {A3[j0]:.1f}\n")
            f.write(f"  ATE: {max_ate*100:.1f}% ({s1[i0,j0]*100:.1f}% vs {s0[i0,j0]*100:.1f}% survival)\n")

        f.write('Worst treatment scenario:\n')
        if worst_i.size > 0:
            i0, j0 = int(worst_i[0]), int(worst_j[0])
            f.write(f"  Parameters: Disease harm = {A2[i0]:.1f}, Treatment harm = {A3[j0]:.1f}\n")
            f.write(f"  ATE: {min_ate*100:.1f}% ({s1[i0,j0]*100:.1f}% vs {s0[i0,j0]*100:.1f}% survival)\n")

        # Treatment benefit analysis
        beneficial_count = int(np.sum(valid_ate > 0))
        harmful_count = int(np.sum(valid_ate < 0))
        neutral_count = int(np.sum(np.abs(valid_ate) < 0.01))
        total_scenarios = int(valid_ate.size)

        f.write('\nTREATMENT BENEFIT ANALYSIS:\n')
        f.write('Scenarios where treatment is:\n')
        f.write(f"  Beneficial (ATE > 0): {beneficial_count}/{total_scenarios} ({beneficial_count/total_scenarios*100:.1f}%)\n")
        f.write(f"  Harmful (ATE < 0): {harmful_count}/{total_scenarios} ({harmful_count/total_scenarios*100:.1f}%)\n")
        f.write(f"  Neutral (|ATE| < 1%): {neutral_count}/{total_scenarios} ({neutral_count/total_scenarios*100:.1f}%)\n")

        # Parameter sensitivity
        disease_corr = 0.0
        treatment_corr = 0.0
        if A2.size > 1 and A3.size > 1:
            mean_ate_by_disease = np.nanmean(ate_matrix, axis=1)    # length A2
            mean_ate_by_treatment = np.nanmean(ate_matrix, axis=0)  # length A3

            f.write('\nPARAMETER SENSITIVITY:\n')

            # Correlations (ignoring NaNs)
            def _corr_nan(x, y):
                x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 2 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
                    return 0.0, float('nan')
                r = float(np.corrcoef(x[mask], y[mask])[0, 1])
                n = int(mask.sum())
                # two-sided p-value via t-stat (approximation)
                if abs(r) >= 1 or n <= 2:
                    p = 0.0 if abs(r) >= 1 else 1.0
                else:
                    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                    # Use normal approximation for p-value
                    from math import erf, sqrt
                    p = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
                return r, p

            disease_corr, disease_p = _corr_nan(A2, mean_ate_by_disease)
            f.write('Disease harm (a2) vs Treatment Effect:\n')
            f.write(f"  Correlation: {disease_corr:.3f} (p = {disease_p:.3f})\n")
            if disease_corr < -0.5:
                f.write('  -> Higher disease harm REDUCES treatment benefit\n')
            elif disease_corr > 0.5:
                f.write('  -> Higher disease harm INCREASES treatment benefit\n')
            else:
                f.write('  -> Disease harm has moderate effect on treatment benefit\n')

            treatment_corr, treatment_p = _corr_nan(A3, mean_ate_by_treatment)
            f.write('Treatment harm (a3) vs Treatment Effect:\n')
            f.write(f"  Correlation: {treatment_corr:.3f} (p = {treatment_p:.3f})\n")
            if treatment_corr < -0.5:
                f.write('  -> Higher treatment harm REDUCES treatment benefit\n')
            elif treatment_corr > 0.5:
                f.write('  -> Higher treatment harm INCREASES treatment benefit\n')
            else:
                f.write('  -> Treatment harm has moderate effect on treatment benefit\n')

        # Risk-benefit zones
        f.write('\nRISK-BENEFIT ZONES:\n')
        high_benefit_threshold = float(np.percentile(valid_ate, 75))
        high_benefit_count = int(np.sum(valid_ate >= high_benefit_threshold))
        f.write('High benefit zone (top 25% of scenarios):\n')
        f.write(f"  ATE threshold: >= {high_benefit_threshold*100:.1f}%\n")
        f.write(f"  Number of scenarios: {high_benefit_count}\n")

        high_risk_threshold = float(np.percentile(valid_ate, 25))
        high_risk_count = int(np.sum(valid_ate <= high_risk_threshold))
        f.write('High risk zone (bottom 25% of scenarios):\n')
        f.write(f"  ATE threshold: <= {high_risk_threshold*100:.1f}%\n")
        f.write(f"  Number of scenarios: {high_risk_count}\n")

        # Clinical decision making
        f.write('\nCLINICAL DECISION MAKING:\n')

        safe_threshold = 0.05
        safe_scenarios = int(np.sum(valid_ate > safe_threshold))

        f.write('Treatment recommendations:\n')
        f.write(f"  Safe treatment zone (ATE > 5%): {safe_scenarios} scenarios ({safe_scenarios/total_scenarios*100:.1f}%)\n")

        dangerous_threshold = -0.02
        dangerous_scenarios = int(np.sum(valid_ate < dangerous_threshold))
        f.write(f"  Dangerous treatment zone (ATE < -2%): {dangerous_scenarios} scenarios ({dangerous_scenarios/total_scenarios*100:.1f}%)\n")

        # Parameter estimation implications
        f.write('\nPARAMETER ESTIMATION IMPLICATIONS:\n')
        f.write('For clinical implementation:\n')
        if abs(disease_corr) > 0.5:
            f.write('- Accurate estimation of disease harm (a2) is CRITICAL\n')
        else:
            f.write('- Accurate estimation of disease harm (a2) is IMPORTANT\n')
        if abs(treatment_corr) > 0.5:
            f.write('- Accurate estimation of treatment harm (a3) is CRITICAL\n')
        else:
            f.write('- Accurate estimation of treatment harm (a3) is IMPORTANT\n')

        if beneficial_count / total_scenarios > 0.8:
            f.write('- Treatment is broadly beneficial across parameter space\n')
        elif harmful_count / total_scenarios > 0.3:
            f.write('- Treatment can be harmful - careful parameter estimation required\n')
        else:
            f.write('- Treatment benefit is parameter-dependent\n')

        # Robust treatment strategies
        f.write('\nROBUST TREATMENT STRATEGIES:\n')
        if A2.size > 1 and A3.size > 1:
            robust_benefit_mask = ate_matrix > 0.02
            robust_regions = int(np.sum(robust_benefit_mask))
            total_cells = int(ate_matrix.size)

            f.write('Robust benefit regions (ATE > 2% regardless of exact parameters):\n')
            f.write(f"  Count: {robust_regions}/{total_cells} scenarios ({robust_regions/total_cells*100:.1f}%)\n")

            if robust_regions > 0.5 * total_cells:
                f.write('  -> Treatment is ROBUST across most parameter combinations\n')
            else:
                f.write('  -> Treatment benefit is SENSITIVE to parameter values\n')

        # Practical recommendations
        f.write('\nPRACTICAL RECOMMENDATIONS:\n')
        f.write('For clinical practice:\n')
        if safe_scenarios / total_scenarios > 0.7:
            f.write('+ Treatment is generally safe and beneficial\n')
            f.write('  - Consider aggressive treatment protocols\n')
            f.write('  - Standard parameter estimation may be sufficient\n')
        else:
            f.write('!!! Treatment benefit is parameter-dependent\n')
            f.write('  - Invest in accurate parameter estimation\n')
            f.write('  - Consider adaptive treatment protocols\n')
            f.write('  - Monitor for treatment-related harm\n')

        f.write('\nFor parameter estimation:\n')
        f.write('- Priority should be given to accurate estimation of:\n')
        if abs(disease_corr) > abs(treatment_corr):
            f.write('  1. Disease harm parameters (higher impact)\n')
            f.write('  2. Treatment harm parameters\n')
        else:
            f.write('  1. Treatment harm parameters (higher impact)\n')
            f.write('  2. Disease harm parameters\n')

        # Technical details
        f.write('\nTECHNICAL DETAILS:\n')
        f.write('- Analysis type: Grid search sensitivity analysis\n')
        f.write('- Grid resolution: 20 x 20 parameter combinations\n')
        f.write('- Mortality model: Logistic hazard with time-varying effects\n')
        f.write('- Treatment assignment: 50% randomization (RCT simulation)\n')
        f.write('- Endpoint: Survival probability at 168 hours\n')
        f.write(f"- Aggregation: Median across {nTrials} trial(s) per parameter set\n")

        # Data availability
        f.write('\nDATA AVAILABILITY:\n')
        f.write('- Complete results saved to: HeatMapAggressive.mat\n')
        f.write('- Survival matrices: s0 (untreated), s1 (treated)\n')
        f.write('- Parameter grids: A2 (disease harm), A3 (treatment harm)\n')
        f.write('- Optimal thresholds: Th_opt (if computed)\n')

    print(f"Heatmap sensitivity analysis results exported to: {_filename}")
except Exception as _exc:
    print(f"WARNING: Failed to export heatmap sensitivity results text file: {_exc}")
