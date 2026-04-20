# %% ToDo:
# - make bias more severe for selection; make it based on L0(t) first few
#   hours; generate L0(t) outside of sim function
# - implement other biases

# %% Logit-Based Hazard Model Data Generation
#
# DESCRIPTION:
#   Generates synthetic clinical trial data using logit-based discrete-time
#   hazard models for causal survival analysis with PKPD modeling.
#
# TRIAL MODES:
#   RCT=1: No treatment changes or dropout (yes randomized controlled trial)
#   RCT=0: Treatment changes and dropout allowed (not randomized controlled trial)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat
from datetime import datetime

# --- Reproducibility (mirrors MATLAB rng(0)) ---
np.random.seed(0)

# --- Translated helpers you already have / will add ---
from fcnGeneratePatientParameters import fcnGeneratePatientParameters
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnSimulate_N_Patients import fcnSimulate_N_Patients
from fcnBiasedAssignmentProb import fcnBiasedAssignmentProb


# These are additional MATLAB helpers referenced in this script.
# Provide Python translations of these (or comment out the calls below until ready).
# from fcnBiasedAssignmentProb import fcnBiasedAssignmentProb
# from fcnSingleSwimmerPlot_v4 import fcnSingleSwimmerPlot_v4
# from fcnPlotKM import fcnPlotKM
# from fcnEstimateDeathParms import fcnEstimateDeathParms
# from fcnEstimateParmsL import fcnEstimateParmsL
# from fcnEstimateParmsPKPD import fcnEstimateParmsPKPD

# -------------------------------------------------------
# Internal KM helper (equivalent to MATLAB ecdf with censoring)
# -------------------------------------------------------
def _km_curve(times: np.ndarray, events: np.ndarray):
    """
    Compute Kaplan–Meier step function S(t) given:
      times  : (n,) event or censor times
      events : (n,) 1=event (death), 0=censored

    Returns:
      t_out, S_out  as stepwise arrays starting with (0,1).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    # Sort by time (stable)
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]

    # Unique event times only (KM jumps only at event times)
    uniq_event_times = np.unique(times[events == 1])
    if uniq_event_times.size == 0:
        # No events -> survival is 1 everywhere
        return np.array([0.0]), np.array([1.0])

    n = len(times)
    at_risk = n
    S = 1.0
    t_out = [0.0]
    S_out = [1.0]
    idx = 0

    for te in uniq_event_times:
        # Remove censored and events prior to te from risk set accounting
        # Count events and censors at each exact time
        d_i = np.sum((times == te) & (events == 1))
        c_i = np.sum((times == te) & (events == 0))

        # Number at risk just prior to te:
        # subtract all individuals with time < te (both censored and events)
        # Move idx forward to first row at te
        while idx < n and times[idx] < te:
            at_risk -= 1
            idx += 1

        # Now times[idx] == te
        # KM multiplicative step
        if at_risk > 0:
            S *= (1.0 - d_i / at_risk)
        else:
            S *= 1.0

        t_out.append(te)
        S_out.append(S)

        # After processing the jump, remove all at te (events and censors) from risk set
        # advance through all rows at te
        # (we already used d_i; now account them and censors)
        k = 0
        while idx + k < n and times[idx + k] == te:
            k += 1
        at_risk -= k
        idx += k

    return np.asarray(t_out, dtype=float), np.asarray(S_out, dtype=float)


# %% 1. SIMULATION PARAMETERS ==========================================

N = 2000  # Number of patients to simulate per trial type

# Target L level
th = 0.1  # LOWER threshold - harder to achieve for sick patients, leading to more treatment

# PI Controller Parameters
ki = 10     # Integral control gain (aggressive disease suppression)
Amax = 50   # Maximum pump rate (treatment upper bound)
parmsControl = np.array([ki, Amax])

# PD parameters with age/SOFA dependencies
age, sofa, C, g, parmsPD = fcnGeneratePatientParameters(
    N, 'TargetCMean', 3, 'TargetGMean', 4, 'CV', 0.1
)

# PK parameter - elimination time constant
ke = 0.5

# Mortality (Y) hazard parameters
# logit_y = a0 + a1*(t(j)/170)^2 + (a2*sofa).*(cumsum_L/24)^2 + (a3*(age/90)).*(cumsum_A/207);
a0 = -7.0
a1 = 0.3
a2 = 20.0  # harmL
a3 = 5.0   # harmA
parmsY = np.array([a0, a1, a2, a3])

# Censoring (V) hazard parameters
# logit_v = Rx(j)*(b0 + b1*(cumsum_A/207) + b2*(t(j)/170)^2) + (1-Rx(j))*(b3 + b4*cumsum_L/24 + b5*(t(j)/170)^2);
b0 = -5.0   # Baseline censoring for treated
b1 = 2.0    # Treatment burden effect
b2 = 0.1    # Time effect for treated
b3 = -5.0   # Baseline for untreated
b4 = 2.0    # Disease burden effect
b5 = 1.5    # Time effect for untreated
parmsV = np.array([b0, b1, b2, b3, b4, b5])

# Disease Natural History Parameters and trajectories
# [growth_rate, peak_height, alpha, decay_rate, sigma_early, sigma_late, sigma_transition]
parmsL = np.array([0.25, 1.0, 0.15, 0.05, 0.15, 0.03, 40.0])
dt = 2.0
t = np.arange(0.0, 168.0 + dt, dt)  # MATLAB 0:dt:168 inclusive

# %% MAIN LOOP FOR BOTH RCT MODES =====================================

for RCT in [0, 1]:

    # Generate baseline disease trajectories
    L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)

    # Treatment assignment probabilities
    if RCT == 1:
        treatProb = np.full(N, 0.5)                 # row vector in MATLAB → (N,) here
    else:
        treatProb = fcnBiasedAssignmentProb(age, sofa, L0[:, :5])  # L0(:,1:5)


    # Simulate N patients
    T = fcnSimulate_N_Patients(
        N, RCT, treatProb, th, C, g, ke, L0, parmsControl, parmsY, parmsV, age, sofa
    )

    # Export to CSV for analysis
    filename = f"trialData{RCT}.csv"
    T.to_csv(filename, index=False)

    # Save "true" values for generating data (overwrites per loop, as in MATLAB)
    savemat(
        "parmsTrue.mat",
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

# %% plot survival curves - RCT = 1 vs naive vs g-formula

# FIGURE: OBSERVATIONAL DATA with NAIVE KAPLAN-MEIER (showing bias without g-formula)
plt.figure(1); plt.clf()
T0 = pd.read_csv("trialData0.csv")

# Optional swimmer plot (provide translation before enabling)
# fcnSingleSwimmerPlot_v4(T0)

# Try to fetch positions from existing axes to align survival curves
fig = plt.gcf()
axes_list = fig.get_axes()
if len(axes_list) > 0:
    positions = np.array([ax.get_position().bounds for ax in axes_list])
    lowest_swimmer_bottom = np.min(positions[:, 1])
    swimmer_left = positions[0, 0]
    swimmer_width = positions[0, 2]
else:
    # Fallback if swimmer plot not created yet
    lowest_swimmer_bottom = 0.45
    swimmer_left = 0.13
    swimmer_width = 0.775

# %% Calculate NAIVE Kaplan-Meier curves from observational data
unique_patients = np.unique(T0["sid"].values)
n_patients = unique_patients.size

treated_patients = []
untreated_patients = []

# Determine treatment group by initial Rx
for pid in unique_patients:
    pdata = T0[T0["sid"] == pid].sort_values("t")
    if int(pdata["Rx"].iloc[0]) == 1:
        treated_patients.append(pid)
    else:
        untreated_patients.append(pid)

# Gather times & events: death -> first Y==1 time; else censored at last t
def _times_events_for_group(Tdf, pids):
    times, events = [], []
    for pid in pids:
        pdata = Tdf[Tdf["sid"] == pid].sort_values("t")
        death_idx = np.argmax(pdata["Y"].values > 0)
        has_death = (pdata["Y"].values > 0).any()
        if has_death:
            t_event = float(pdata["t"].values[np.where(pdata["Y"].values > 0)[0][0]])
            times.append(t_event)
            events.append(1)
        else:
            times.append(float(np.max(pdata["t"].values)))
            events.append(0)
    return np.asarray(times, dtype=float), np.asarray(events, dtype=int)

treated_times, treated_events = _times_events_for_group(T0, treated_patients)
untreated_times, untreated_events = _times_events_for_group(T0, untreated_patients)

# KM curves (equivalent to MATLAB ecdf with 'Censoring')
t0_naive, s0_naive = _km_curve(untreated_times, untreated_events)
t1_naive, s1_naive = _km_curve(treated_times, treated_events)

# Debug stats
print("Naive KM Statistics:")
print(f"Untreated: {len(untreated_times)} patients, {int(untreated_events.sum())} deaths")
print(f"Treated:   {len(treated_times)} patients, {int(treated_events.sum())} deaths")
print(f"Untreated survival at t=168: {s0_naive[-1]:.3f}")
print(f"Treated survival at t=168:   {s1_naive[-1]:.3f}")

# ATE at end of trial
ate_naive = s1_naive[-1] - s0_naive[-1]
print("\n=== AVERAGE TREATMENT EFFECT AT 168 HOURS ===")
print(f"Naive KM estimate: ATE = {100*ate_naive:.1f}% ({s1_naive[-1]:.3f} - {s0_naive[-1]:.3f})")
if ate_naive < 0:
    print("  → Treatment appears HARMFUL in naive analysis")
else:
    print("  → Treatment appears beneficial in naive analysis")

# %% Get RCT ground truth curves for comparison
T1 = pd.read_csv("trialData1.csv")

# If you have fcnPlotKM translated, use it; else compute here
def _plotKM_py(Tdf):
    """Return (s0, s1, t0, t1) for untreated/treated groups using KM."""
    # classify by Rx at baseline (RCT: fixed anyway)
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

# s0_true, s1_true, t0_true, t1_true = fcnPlotKM(T1)
s0_true, s1_true, t0_true, t1_true = _plotKM_py(T1)

# True ATE (RCT)
ate_true = s1_true[-1] - s0_true[-1]
print(f"\nRCT (ground truth): ATE = {100*ate_true:.1f}% ({s1_true[-1]:.3f} - {s0_true[-1]:.3f})")
print("  → Treatment is truly HARMFUL" if ate_true < 0 else "  → Treatment is truly BENEFICIAL")

# Compare bias
bias = ate_naive - ate_true
print("\n=== BIAS IN NAIVE ESTIMATE ===")
print(f"Bias = {100*bias:.1f} percentage points")
wrong_direction = (np.sign(ate_naive) != np.sign(ate_true)) and (ate_true != 0)
if wrong_direction:
    print("⚠️  CRITICAL: Treatment effect is in the WRONG DIRECTION!")
    if ate_naive < 0:
        print("   Naive analysis suggests treatment is harmful")
    else:
        print("   Naive analysis suggests treatment is beneficial")
    if ate_true < 0:
        print("   But RCT shows treatment is actually harmful")
    else:
        print("   But RCT shows treatment is actually beneficial")

# %% Get g-formula estimates (requires your translated estimators)
# T0 = pd.read_csv('trialData0.csv')
# parmsY_est = fcnEstimateDeathParms(T0)
# parmsL_est, LL, AA, age_est, sofa_est, t_est = fcnEstimateParmsL(T0)
# ke_est, C_est, g_est, theta_est = fcnEstimateParmsPKPD(parmsL_est, LL, AA, age_est, sofa_est, t_est)
#
# RCT = 1
# treatProb_est = np.full(N, 0.5)
# L0_est = fcnGenerateStochasticTrajectories(t_est, parmsL_est, N)
# T1_est = fcnSimulate_N_Patients(N, RCT, treatProb_est, th, C, g, ke, L0_est, parmsControl, parmsY_est, np.zeros(6), age_est, sofa_est)
# s0_gf, s1_gf, t0_gf, t1_gf = _plotKM_py(T1_est)
#
# ate_gformula = s1_gf[-1] - s0_gf[-1]
# print(f"\nG-formula estimate: ATE = {100*ate_gformula:.1f}% ({s1_gf[-1]:.3f} - {s0_gf[-1]:.3f})")
# print("  → G-formula correctly adjusts for confounding")

# %% Summary comparison (if g-formula computed, uncomment related lines)
print("\n=== SUMMARY: AVERAGE TREATMENT EFFECTS ===")
print("Method          | ATE at 168h | Interpretation")
print("----------------|-------------|----------------")

rct_interp = "Harmful" if ate_true < 0 else "Beneficial"
print(f"RCT (truth)     | {100*ate_true:+6.1f}%    | {rct_interp}")

naive_interp = "Harmful" if ate_naive < 0 else "Beneficial"
naive_suffix = " (WRONG!)" if wrong_direction else ""
print(f"Naive KM        | {100*ate_naive:+6.1f}%    | {naive_interp}{naive_suffix}")

# if 'ate_gformula' in locals():
#     gf_interp = "Harmful" if ate_gformula < 0 else "Beneficial"
#     print(f"G-formula       | {100*ate_gformula:+6.1f}%    | {gf_interp}")

print(f"\nNaive bias: {100*(ate_naive - ate_true):.1f} percentage points")
# if 'ate_gformula' in locals():
#     print(f"G-formula bias: {100*(ate_gformula - ate_true):.1f} percentage points")

# %% Position for bottom plot - use remaining space at bottom
vertical_gap = 0.015
bottom_margin = 0.05
height = max(0.15, lowest_swimmer_bottom - vertical_gap - bottom_margin)

# Create axes aligned with swimmer plots (or fallback)
ax4 = plt.axes([swimmer_left, bottom_margin, swimmer_width, height])

# Plot survival curves
plt.sca(ax4)
plt.grid(True)
plt.xlabel("Time (hours)", fontsize=12)
plt.ylabel("Survival Probability", fontsize=12)
plt.xlim(0, 168)
plt.ylim(0, 1)

# RCT ground truth curves (black, solid to match the MATLAB 'k' examples)
t0 = np.arange(0, 168 + 2, 2.0)
t1 = t0
plt.plot(t0_true, s0_true, 'k', linewidth=2.5)
plt.plot(t1_true, s1_true, 'k', linewidth=2.5)

# Naive observational curves (red)
plt.step(t0_naive, s0_naive, where='post', linewidth=2.5, color='r')
plt.step(t1_naive, s1_naive, where='post', linewidth=2.5, color='r')

# g-formula curves (green) if computed
# if 't0_gf' in locals():
#     plt.step(t0_gf, s0_gf, where='post', linewidth=2.5, color='g')
#     plt.step(t1_gf, s1_gf, where='post', linewidth=2.5, color='g')

# Label positions
label_time = 130.0
def _nearest_idx(tvec, t0):
    return int(np.argmin(np.abs(np.asarray(tvec) - t0)))

idx0_true = _nearest_idx(t0_true, label_time)
idx1_true = _nearest_idx(t1_true, label_time)
idx0_naive = _nearest_idx(t0_naive, label_time)
idx1_naive = _nearest_idx(t1_naive, label_time)

plt.text(label_time-10, s0_true[idx0_true] + 0.05, "Untreated (RCT)",
         color=(0.4,0.4,0.8), fontsize=10, fontweight='bold', ha='right')
plt.text(label_time-10, s0_naive[idx0_naive] - 0.05, "Untreated (Naive)",
         color=(0.4,0.4,0.8), fontsize=10, fontweight='bold', ha='right')

plt.text(label_time+10, s1_true[idx1_true] + 0.05, "Treated (RCT)",
         color=(0,0,0.5), fontsize=10, fontweight='bold', ha='left')
plt.text(label_time+10, s1_naive[idx1_naive] - 0.05, "Treated (Naive)",
         color=(0,0,0.5), fontsize=10, fontweight='bold', ha='left')

# Tick formatting (every 24h)
ax4.set_xticks(np.arange(0, 168+1, 24))
ax4.set_xticklabels([str(x) for x in range(0, 169, 24)])
ax4.set_xlim(0, 168)
ax4.set_ylim(0, 1)

# Figure size and export as PDF
fig_width, fig_height = 6, 9
fig = plt.gcf()
fig.set_size_inches(fig_width, fig_height)
plt.savefig("Fig_swimmer_survival_plot_Obs_Naive.pdf", format="pdf", dpi=300, bbox_inches="tight")

# ---------------------------------------------------------------------
# Export trial simulation results to text file for paper
# (Ported from matlab/a0_GenerateTrialData.m:354-476)
# ---------------------------------------------------------------------
try:
    _now = datetime.now()
    _filename = f"trial_simulation_results_{_now.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(_filename, 'w') as f:
        f.write('==========================================================\n')
        f.write('TRIAL SIMULATION RESULTS FOR PAPER\n')
        f.write(f"Generated on: {_now.strftime('%d-%b-%Y %H:%M:%S')}\n")
        f.write('==========================================================\n\n')

        # Simulation parameters
        f.write('SIMULATION PARAMETERS:\n')
        f.write(f"- Sample size: {N} patients per trial type\n")
        f.write('- Study period: 168 hours\n')
        f.write('- Time step: 2 hours\n')
        f.write(f"- Target threshold: {th:.3f}\n")
        f.write(f"- Elimination constant (ke): {ke:.3f}\n")
        f.write(f"- PI controller gain (ki): {ki:.1f}\n")
        f.write(f"- Maximum pump rate (Amax): {Amax:.1f}\n\n")

        # True model parameters
        f.write('TRUE MODEL PARAMETERS:\n')
        f.write('Mortality hazard (Y) parameters:\n')
        for i, v in enumerate(parmsY):
            f.write(f"  a{i}: {v:.3f}\n")

        f.write('\nCensoring hazard (V) parameters:\n')
        param_names_V = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
        for i, v in enumerate(parmsV):
            f.write(f"  {param_names_V[i]}: {v:.3f}\n")

        f.write('\nDisease natural history (L) parameters:\n')
        param_names_L = ['growth_rate', 'peak_height', 'alpha', 'decay_rate',
                         'sigma_early', 'sigma_late', 'sigma_transition']
        for i, v in enumerate(parmsL):
            f.write(f"  {param_names_L[i]}: {v:.3f}\n")

        # Patient population characteristics
        f.write('\nPATIENT POPULATION:\n')
        f.write(f"Age: {np.mean(age):.1f} +/- {np.std(age, ddof=1):.1f} years\n")
        f.write(f"SOFA score: {np.mean(sofa):.1f} +/- {np.std(sofa, ddof=1):.1f}\n")
        f.write(f"Clearance (C): {np.mean(C):.2f} +/- {np.std(C, ddof=1):.2f}\n")
        f.write(f"Potency (g): {np.mean(g):.2f} +/- {np.std(g, ddof=1):.2f}\n\n")

        # Treatment assignment bias (observational study)
        f.write('TREATMENT ASSIGNMENT:\n')
        f.write('RCT (trial=1): Random 50% assignment\n')
        f.write('Observational (trial=0): Biased assignment based on age, SOFA, initial disease severity\n')
        f.write(f"Treated patients in observational study: {len(treated_patients)} ({100*len(treated_patients)/N:.1f}%)\n")
        f.write(f"Untreated patients in observational study: {len(untreated_patients)} ({100*len(untreated_patients)/N:.1f}%)\n\n")

        # Primary results - Average Treatment Effects
        f.write('PRIMARY RESULTS - AVERAGE TREATMENT EFFECTS AT 168 HOURS:\n')
        f.write(f"{'Method':<20s} {'ATE':<12s} {'Interpretation':<15s} {'Bias vs RCT':<15s}\n")
        f.write(f"{'-'*20:<20s} {'-'*12:<12s} {'-'*15:<15s} {'-'*15:<15s}\n")

        rct_interp = 'Harmful' if ate_true < 0 else 'Beneficial'
        f.write(f"{'RCT (ground truth)':<20s} {100*ate_true:+7.1f}%     {rct_interp:<15s} {'0.0% (reference)':<15s}\n")

        naive_interp = 'Harmful' if ate_naive < 0 else 'Beneficial'
        wrong_direction_local = (np.sign(ate_naive) != np.sign(ate_true)) and (ate_true != 0)
        naive_suffix = ' (WRONG!)' if wrong_direction_local else ''
        bias_naive = 100 * (ate_naive - ate_true)
        f.write(f"{'Naive Kaplan-Meier':<20s} {100*ate_naive:+7.1f}%     {naive_interp + naive_suffix:<15s} {bias_naive:+7.1f}% pts\n")

        # G-formula (only if computed in Python scope)
        if 'ate_gformula' in dir():
            gf_interp = 'Harmful' if ate_gformula < 0 else 'Beneficial'
            bias_gformula = 100 * (ate_gformula - ate_true)
            f.write(f"{'G-formula':<20s} {100*ate_gformula:+7.1f}%     {gf_interp:<15s} {bias_gformula:+7.1f}% pts\n")
        else:
            # TODO: add ate_gformula when available
            bias_gformula = None

        # Survival rates at 168 hours
        f.write('\nSURVIVAL RATES AT 168 HOURS:\n')
        f.write(f"{'Method':<20s} {'Untreated':<15s} {'Treated':<15s}\n")
        f.write(f"{'-'*20:<20s} {'-'*15:<15s} {'-'*15:<15s}\n")
        f.write(f"{'RCT (truth)':<20s} {s0_true[-1]*100:<14.1f}% {s1_true[-1]*100:<14.1f}%\n")
        f.write(f"{'Naive observational':<20s} {s0_naive[-1]*100:<14.1f}% {s1_naive[-1]*100:<14.1f}%\n")
        if 's0_gf' in dir() and 's1_gf' in dir():
            f.write(f"{'G-formula':<20s} {s0_gf[-1]*100:<14.1f}% {s1_gf[-1]*100:<14.1f}%\n")
        # else: TODO: add s0_gf/s1_gf when available

        # Key findings
        f.write('\nKEY FINDINGS:\n')
        if wrong_direction_local:
            f.write(f"!!! CRITICAL BIAS: Naive analysis suggests treatment is {naive_interp},\n")
            f.write(f"   but RCT shows treatment is actually {rct_interp}\n")
            f.write('   -> This demonstrates the importance of causal inference methods\n')
        if bias_gformula is not None:
            f.write(f"- G-formula reduces bias from {abs(bias_naive):.1f} to {abs(bias_gformula):.1f} percentage points\n")
            if abs(bias_naive) > 0:
                f.write(f"- Bias reduction: {100*(abs(bias_naive) - abs(bias_gformula))/abs(bias_naive):.1f}% improvement\n")
        # else: TODO: add bias_gformula when g-formula computed

        # Study implications
        f.write('\nSTUDY IMPLICATIONS:\n')
        f.write('- Observational studies with biased treatment assignment can lead to\n')
        f.write('  severely misleading conclusions about treatment effectiveness\n')
        f.write('- G-formula/parametric g-computation can correct for confounding\n')
        f.write('  when all confounders are measured and properly modeled\n')
        f.write('- This simulation validates the causal inference methodology\n')
        f.write('  for use in real clinical data analysis\n')

        # Methods summary
        f.write('\nMETHODS:\n')
        f.write('- Data generation: Discrete-time hazard models with PKPD dynamics\n')
        f.write('- Treatment assignment: Biased based on age, SOFA, disease severity\n')
        f.write('- Causal inference: G-formula with estimated parameters\n')
        f.write('- Comparison: RCT (truth) vs Naive analysis vs G-formula\n')

    print(f"\nTrial simulation results exported to: {_filename}")
except Exception as _exc:
    print(f"WARNING: Failed to export trial simulation results text file: {_exc}")
