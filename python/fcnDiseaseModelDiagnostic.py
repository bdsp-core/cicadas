# fcnDiseaseModelDiagnostics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import ks_2samp

# --- Optional imports of your existing helpers; fall back to locals if missing ---
try:
    from fcn_kaplanMeier import fcn_kaplanMeier
except Exception:
    fcn_kaplanMeier = None

try:
    from fcnPlotKM import fcnPlotKM as fcnPlotKM_ext
except Exception:
    fcnPlotKM_ext = None

try:
    from fcn_generateTrajectory import fcn_generateTrajectory
except Exception:
    fcn_generateTrajectory = None


def fcnDiseaseModelDiagnostics(
    T1_est: pd.DataFrame,
    true_params: np.ndarray,
    C_est_all: np.ndarray,
    g_est_all: np.ndarray,
    theta_est: np.ndarray,
    L_trajectories: np.ndarray,
    A_trajectories: np.ndarray,
    age: np.ndarray,
    sofa: np.ndarray,
    t: np.ndarray,
):
    """
    Diagnostic plots and metrics for disease model parameter estimation.

    Parameter order (length 14):
      [γ, H, α, δ, σ_early, σ_late, τ, b0_C, b1_C, b2_C, b0_g, b1_g, b2_g, ke]
    """

    # Match MATLAB script: fix grid (even if t was passed)
    dt = 2.0
    t = np.arange(0.0, 168.0 + dt, dt)  # Nt = 85
    Nt = t.size

    # Treated indicator from A trajectories
    is_treated = np.nanmax(A_trajectories, axis=1) > 0
    n_treated = int(np.sum(is_treated))
    n_untreated = int(np.sum(~is_treated))

    # Parameter names
    param_names = [
        "Growth rate (γ)", "Peak height (H)", "Mean reversion (α)",
        "Decay rate (δ)", "Early volatility (σ_early)",
        "Late volatility (σ_late)", "Volatility decay (τ)",
        "C intercept (b0_C)", "C age coef (b1_C)", "C SOFA coef (b2_C)",
        "g intercept (b0_g)", "g age coef (b1_g)", "g SOFA coef (b2_g)",
        "PK value (ke)"
    ]

    theta_est = np.asarray(theta_est, dtype=float).reshape(-1)
    print("\n\nFinal estimated parameters:")
    for i in range(min(14, theta_est.size)):
        print(f"  {param_names[i]}: {theta_est[i]:.4f}")

    # Parameter errors against truth
    true_params = np.asarray(true_params, dtype=float).reshape(-1)
    if true_params.size >= 14:
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.where(np.abs(true_params[:14]) > 0, np.abs(true_params[:14]), np.nan)
            param_errors = np.abs(theta_est[:14] - true_params[:14]) / denom * 100.0
        print("\nParameter estimation errors:")
        for i in range(14):
            pe = param_errors[i]
            print(f"  {param_names[i]}: {np.nan_to_num(pe):.1f}%")
        print(f"\nMean absolute percentage error: {np.nanmean(param_errors):.1f}%")

    # Goodness-of-fit metrics (no plots)
    _compute_goodness_of_fit_metrics(L_trajectories, A_trajectories, t, theta_est, is_treated)

    # === Survival curves: True vs Estimated ===
    # Ground-truth curves from trialData1.csv
    T1 = pd.read_csv("trialData1.csv")
    if fcnPlotKM_ext is not None:
        s0_true, s1_true, t0_true, t1_true = fcnPlotKM_ext(T1)
    else:
        s0_true, s1_true, t0_true, t1_true = _fcnPlotKM_py(T1)

    # Estimated curves from T1_est
    if fcnPlotKM_ext is not None:
        s0_est, s1_est, t0_est, t1_est = fcnPlotKM_ext(T1_est)
    else:
        s0_est, s1_est, t0_est, t1_est = _fcnPlotKM_py(T1_est)

    plt.figure(3); plt.clf()
    plt.plot(t0_true, s0_true, 'b--', t1_true, s1_true, 'r--', linewidth=2)
    plt.plot(t0_est, s0_est, 'b', linewidth=2, label="Untreated (Est)")
    plt.plot(t1_est, s1_est, 'r', linewidth=2, label="Treated (Est)")
    plt.gcf().set_facecolor("w")
    plt.xlabel("Hours")
    plt.ylabel("% Alive")
    plt.title("Survival Curves: True vs Estimated")
    plt.legend(["Untreated (True)", "Treated (True)", "Untreated (Est)", "Treated (Est)"], loc="lower left")
    plt.grid(True); plt.box(False)

    # === Additional analysis ===
    m = loadmat("parmsTrue.mat", squeeze_me=True, struct_as_record=False)
    C_true_all = np.asarray(m["C"], dtype=float).reshape(-1)
    g_true_all = np.asarray(m["g"], dtype=float).reshape(-1)

    T0 = pd.read_csv("trialData0.csv")
    unique_patients = np.unique(T0["sid"].values)
    n_patients = unique_patients.size

    # Patient-level data (age/sofa/treatment-at-baseline)
    patient_age = np.zeros(n_patients)
    patient_sofa = np.zeros(n_patients)
    patient_treated = np.zeros(n_patients, dtype=int)

    for idx_i, pid in enumerate(unique_patients):
        pdata = T0[T0["sid"] == pid].sort_values("t")
        patient_age[idx_i] = float(pdata["age"].iloc[0])
        patient_sofa[idx_i] = float(pdata["sofa"].iloc[0])
        patient_treated[idx_i] = int(pdata["Rx"].iloc[0])

    # Trajectory matrices
    LL = np.full((n_patients, Nt), np.nan)
    AA = np.full((n_patients, Nt), np.nan)
    for idx_i, pid in enumerate(unique_patients):
        pdata = T0[T0["sid"] == pid].sort_values("t")
        n_obs = len(pdata)
        t_idx = np.clip((pdata["t"].to_numpy() / dt).astype(int), 0, Nt - 1)
        LL[idx_i, t_idx] = pdata["L"].to_numpy()
        AA[idx_i, t_idx] = pdata["A"].to_numpy()

    # Patient-level parameter accuracy
    C_est_all = np.asarray(C_est_all, dtype=float).reshape(-1)
    g_est_all = np.asarray(g_est_all, dtype=float).reshape(-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        C_pct_err = np.abs(C_est_all - C_true_all) / np.where(C_true_all != 0, np.abs(C_true_all), np.nan) * 100.0
        g_pct_err = np.abs(g_est_all - g_true_all) / np.where(g_true_all != 0, np.abs(g_true_all), np.nan) * 100.0

    print("\n*** PATIENT-LEVEL PARAMETER ACCURACY ***")
    print("C values:")
    print(f"  Mean absolute error: {np.nanmean(np.abs(C_est_all - C_true_all)):.3f} ({np.nanmean(C_pct_err):.1f}%)")
    print(f"  Median absolute error: {np.nanmedian(np.abs(C_est_all - C_true_all)):.3f} ({np.nanmedian(C_pct_err):.1f}%)")

    print("\ng values:")
    print(f"  Mean absolute error: {np.nanmean(np.abs(g_est_all - g_true_all)):.3f} ({np.nanmean(g_pct_err):.1f}%)")
    print(f"  Median absolute error: {np.nanmedian(np.abs(g_est_all - g_true_all)):.3f} ({np.nanmedian(g_pct_err):.1f}%)")

    print("\nFor TREATED patients only:")
    treated_mask = patient_treated == 1
    print(f"  C mean error: {np.nanmean(C_pct_err[treated_mask]):.1f}%, median: {np.nanmedian(C_pct_err[treated_mask]):.1f}%")
    print(f"  g mean error: {np.nanmean(g_pct_err[treated_mask]):.1f}%, median: {np.nanmedian(g_pct_err[treated_mask]):.1f}%")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70 + "\n")

    # Survival rates at key time points
    time_points = [24, 48, 72, 96, 120, 144, 168]
    print("Survival rates at key time points:")
    print("Time (h) | True Untreated | True Treated | Est Untreated | Est Treated")
    print("-" * 70)

    def _near_idx(tt, tp):
        return int(np.argmin(np.abs(np.asarray(tt) - tp)))

    for tp in time_points:
        i0t = _near_idx(t0_true, tp)
        i1t = _near_idx(t1_true, tp)
        i0e = _near_idx(t0_est, tp)
        i1e = _near_idx(t1_est, tp)
        print(f"{tp:8d} | {s0_true[i0t]*100:14.1f}% | {s1_true[i1t]*100:12.1f}% | {s0_est[i0e]*100:13.1f}% | {s1_est[i1e]*100:11.1f}%")

    # Treatment effect (mean survival difference over grid)
    treatment_effect_true = float(np.mean(s1_true) - np.mean(s0_true))
    treatment_effect_est = float(np.mean(s1_est) - np.mean(s0_est))
    err_abs = abs(treatment_effect_est - treatment_effect_true)
    err_pct = (err_abs / abs(treatment_effect_true) * 100.0) if treatment_effect_true != 0 else np.nan

    print("\nTreatment effect (mean survival difference):")
    print(f"  True: {treatment_effect_true:.3f}")
    print(f"  Estimated: {treatment_effect_est:.3f}")
    print(f"  Error: {err_abs:.3f} ({err_pct:.1f}%)")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


# ---------------------------
# Local helpers (fallbacks)
# ---------------------------

def _fcnPlotKM_py(Tdf: pd.DataFrame):
    """
    KM curve on a fixed grid 0,2,...,168 (returns s0,s1,t0,t1 with length 85 each).
    Group by initial Rx; first Y==1 is event time else censored at last t.
    """
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

    if fcn_kaplanMeier is not None:
        t_ut_raw, s_ut_raw, _, _ = fcn_kaplanMeier(ut_times, ut_events)
        t_tr_raw, s_tr_raw, _, _ = fcn_kaplanMeier(tr_times, tr_events)
    else:
        # Minimal KM (events only)
        t_ut_raw, s_ut_raw = _km_curve_simple(ut_times, ut_events)
        t_tr_raw, s_tr_raw = _km_curve_simple(tr_times, tr_events)

    # prepend (0,1)
    s0_raw = np.r_[1.0, s_ut_raw]
    t0_raw = np.r_[0.0, t_ut_raw]
    s1_raw = np.r_[1.0, s_tr_raw]
    t1_raw = np.r_[0.0, t_tr_raw]

    # align to fixed grid using "previous" (step) interpolation
    t_grid = np.arange(0.0, 168.0 + 2.0, 2.0)
    s0 = _step_align(t0_raw, s0_raw, t_grid)
    s1 = _step_align(t1_raw, s1_raw, t_grid)

    # extend flat if last observed time < 168
    if t0_raw.max() < 168:
        s0[t_grid > t0_raw.max()] = s0_raw[-1]
    if t1_raw.max() < 168:
        s1[t_grid > t1_raw.max()] = s1_raw[-1]

    # sanity (85 points)
    assert s0.size == 85 and s1.size == 85
    return s0, s1, t_grid, t_grid


def _times_events_for_group(Tdf: pd.DataFrame, pids):
    times, events = [], []
    for pid in pids:
        pdata = Tdf[Tdf["sid"] == pid].sort_values("t")
        y = pdata["Y"].values
        if (y == 1).any():
            t_event = float(pdata["t"].values[np.where(y == 1)[0][0]])
            times.append(t_event); events.append(1)
        else:
            times.append(float(pdata["t"].values[-1])); events.append(0)
    return np.asarray(times, float), np.asarray(events, int)


def _km_curve_simple(times: np.ndarray, events: np.ndarray):
    # Compute KM at unique event times; returns (t_event, S_at_event)
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    uniq = np.unique(times[events == 1])
    if uniq.size == 0:
        return np.array([0.0]), np.array([1.0])
    n = times.size
    S = 1.0
    t_out, s_out = [], []
    for te in uniq:
        n_risk = np.sum(times >= te)
        d_i = np.sum((times == te) & (events == 1))
        if n_risk > 0:
            S *= (1.0 - d_i / n_risk)
        t_out.append(te); s_out.append(S)
    return np.asarray(t_out), np.asarray(s_out)


def _step_align(t_raw: np.ndarray, s_raw: np.ndarray, t_grid: np.ndarray):
    # previous-value step interpolation
    s = np.zeros_like(t_grid, dtype=float)
    for i, tg in enumerate(t_grid):
        idx = np.where(t_raw <= tg)[0]
        s[i] = s_raw[idx[-1]] if idx.size > 0 else 1.0
    return s


def _compute_goodness_of_fit_metrics(L_trajectories, A_trajectories, t, theta_est, is_treated):
    # Extract L params and simulate comparison trajectories
    L_params_est = np.asarray(theta_est[:7], dtype=float)
    if fcn_generateTrajectory is None:
        # If the generator isn't available, skip with a note
        print("\n\nGoodness of Fit Metrics:")
        print("(Skipped detailed L simulation: fcn_generateTrajectory not found)")
        return

    dt = t[1] - t[0]
    N_sim = 100
    L_sim, _ = fcn_generateTrajectory(L_params_est, N_sim, t[-1], dt) if _expects_2(L_params_est) \
               else (fcn_generateTrajectory(L_params_est, N_sim, t[-1], dt), None)

    L_obs_untreated = L_trajectories[~is_treated, :]
    obs_peaks = np.nanmax(L_obs_untreated, axis=1)
    sim_peaks = np.nanmax(L_sim, axis=1)

    obs_onset = _computeOnsetTimes(L_obs_untreated, t)
    sim_onset = _computeOnsetTimes(L_sim, t)

    obs_mean = np.nanmean(L_obs_untreated, axis=0)
    sim_mean = np.nanmean(L_sim, axis=0)

    obs_var = np.nanvar(L_obs_untreated, axis=0)
    sim_var = np.nanvar(L_sim, axis=0)

    # KS tests
    _, p_peaks = ks_2samp(obs_peaks[~np.isnan(obs_peaks)], sim_peaks[~np.isnan(sim_peaks)])
    _, p_onset = ks_2samp(obs_onset[~np.isnan(obs_onset)], sim_onset[~np.isnan(sim_onset)])

    mse_mean = np.nanmean((obs_mean - sim_mean) ** 2)
    var_error = np.nanmean(np.abs(obs_var - sim_var) / (obs_var + 1e-6))

    print("\n\nGoodness of Fit Metrics:")
    print(f"Peak values K-S test p-value: {p_peaks:.3f}")
    print(f"Onset times K-S test p-value: {p_onset:.3f}")
    print(f"MSE of mean trajectories: {mse_mean:.4f}")
    print(f"Mean relative error in variance: {var_error:.3f}")


def _computeOnsetTimes(trajectories: np.ndarray, t: np.ndarray):
    N = trajectories.shape[0]
    onset_times = np.zeros(N)
    for i in range(N):
        row = trajectories[i, :]
        peak = np.nanmax(row)
        thr = 0.1 * peak
        idx = np.where(row >= thr)[0]
        onset_times[i] = t[idx[0]] if idx.size > 0 else np.nan
    return onset_times


def _expects_2(L_params_est):
    # Helper to handle fcn_generateTrajectory returning (trajectories,t) vs trajectories only
    try:
        _ = fcn_generateTrajectory(L_params_est, 1, 10, 2)
        return False  # returned single array → no tuple
    except TypeError:
        # Might return tuple
        try:
            trj, _ = fcn_generateTrajectory(L_params_est, 1, 10, 2)
            _ = trj  # suppress lint
            return True
        except Exception:
            return False
