import numpy as np
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass

# Optional plotting for the "verbose" summary section
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


def fcnEstimateKe_Standalone(L_obs: np.ndarray, A_obs: np.ndarray, *args, **kwargs) -> Tuple[float, Dict[str, Any]]:
    """
    Standalone ke Estimation from Dose Change Dynamics
    Direct translation of MATLAB:
        [ke_est, results] = fcnEstimateKe_Standalone(L_obs, A_obs, varargin)

    Name–value params:
        'KeRange'           : [low, high] (default [0.3, 0.9])
        'WindowSize'        : int (default 10)       # time points after dose change
        'MinDoseChange'     : float (default 1.0)    # minimum A jump to consider significant
        'EstimationMethod'  : 'robust'|'mle' (default 'robust')
        'AssumeC'           : float (default 3.0)
        'AssumeG'           : float (default 4.0)
        'Verbose'           : bool (default True)
        'dt'                : float (default 2)
    """

    # --------------------------
    # Parse name–value args
    # --------------------------
    opts: Dict[str, Any] = {
        "KeRange": np.array([0.3, 0.9], dtype=float),
        "WindowSize": 10,
        "MinDoseChange": 1.0,
        "EstimationMethod": "robust",
        "AssumeC": 3.0,
        "AssumeG": 4.0,
        "Verbose": True,
        "dt": 2.0,
    }

    if len(args) % 2 != 0:
        raise ValueError("Name–value pairs must be key,value,... (even number of *args).")
    for k, v in zip(args[::2], args[1::2]):
        if not isinstance(k, str):
            raise ValueError("Name–value keys must be strings.")
        opts[k] = v
    opts.update(kwargs)

    ke_range = np.asarray(opts["KeRange"], dtype=float).reshape(-1)
    if ke_range.size != 2:
        raise ValueError("KeRange must be length-2.")
    window_size = int(opts["WindowSize"])
    min_dose_change = float(opts["MinDoseChange"])
    method = str(opts["EstimationMethod"]).lower()
    C_approx = float(opts["AssumeC"])
    g_approx = float(opts["AssumeG"])
    verbose = bool(opts["Verbose"])
    dt = float(opts["dt"])

    L_obs = np.asarray(L_obs, dtype=float)
    A_obs = np.asarray(A_obs, dtype=float)
    if L_obs.shape != A_obs.shape:
        raise ValueError("L_obs and A_obs must have the same shape.")
    N, T = L_obs.shape

    if verbose:
        print("\n========================================")
        print("STANDALONE ke ESTIMATION")
        print("========================================")
        print("Method: Dose change response analysis")
        print(f"Patients: {N}, Time points: {T}")
        print(f"ke search range: [{ke_range[0]:.2f}, {ke_range[1]:.2f}]")

    # --------------------------
    # Step 1: Detect dose changes
    # --------------------------
    dose_changes = _detectDoseChanges(A_obs, min_dose_change)
    if verbose:
        total_changes = int(dose_changes.sum())
        print(f"Detected {total_changes} significant dose changes")

    # --------------------------
    # Step 2: Extract response windows
    # --------------------------
    windows = _extractResponseWindows(L_obs, A_obs, dose_changes, window_size)
    if verbose:
        print(f"Extracted {len(windows)} valid response windows")

    # --------------------------
    # Step 3: Estimate ke per window
    # --------------------------
    ke_estimates: List[float] = []
    window_weights: List[float] = []

    for win in windows:
        ke_w, weight_w = _estimateKeFromWindow(win, C_approx, g_approx, ke_range, dt)
        if (ke_w is not None) and np.isfinite(ke_w) and (ke_w > 0):
            ke_estimates.append(float(ke_w))
            window_weights.append(float(weight_w))

    if verbose:
        print(f"Successfully estimated ke from {len(ke_estimates)} windows")

    # --------------------------
    # Step 4: Aggregate estimates
    # --------------------------
    ke_est: float
    ke_std: float

    if len(ke_estimates) == 0:
        # No estimates available -> return NaN-like results
        ke_est = np.nan
        ke_std = np.nan
        ci_95 = np.array([np.nan, np.nan])
        results = _assemble_results(ke_est, ke_std, ci_95, ke_estimates, window_weights,
                                    len(windows), method, dose_changes)
        if verbose:
            print("No valid windows for ke estimation. Returning NaN.")
        return ke_est, results

    v = np.asarray(ke_estimates, dtype=float)
    w = np.asarray(window_weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(v)

    if method == "robust":
        ke_est = _weightedMedian(v, w)
        ke_std = np.median(np.abs(v - np.median(v))) * 1.4826  # MAD-based robust std
    else:  # 'mle' in the MATLAB script uses weighted mean
        ke_est = float(np.sum(v * w) / np.sum(w))
        ke_std = float(np.std(v, ddof=0))

    # --------------------------
    # Step 5: Bootstrap CI
    # --------------------------
    if len(v) > 10:
        n_boot = 1000
        rng = np.random.default_rng()
        boot_vals = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.choice(len(v), size=len(v), replace=True)
            boot_vals[b] = _weightedMedian(v[idx], w[idx])
        ci_95 = np.percentile(boot_vals, [2.5, 97.5])
    else:
        ci_95 = ke_est + np.array([-1.96, 1.96]) * ke_std

    # --------------------------
    # Compile results + prints
    # --------------------------
    results = _assemble_results(ke_est, ke_std, ci_95, ke_estimates, window_weights,
                                len(windows), method, dose_changes)

    if verbose:
        print("\n========================================")
        print("RESULTS:")
        print("========================================")
        print(f"Estimated ke: {ke_est:.3f} ± {ke_std:.3f}")
        print(f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
        print(f"CV: {results['cv'] * 100:.2f}%")
        print(f"Success rate: {results['convergence'] * 100:.1f}%")

        # Distribution dump
        print("\nDistribution of ke estimates:")
        print(f"  Min: {np.min(v):.3f}")
        print(f"  25%: {np.percentile(v, 25):.3f}")
        print(f"  50%: {np.median(v):.3f} (median)")
        print(f"  75%: {np.percentile(v, 75):.3f}")
        print(f"  Max: {np.max(v):.3f}")

        # Optional visualization (if matplotlib is available)
        if _HAS_PLT and (len(v) > 5):
            fig = plt.figure("ke Estimation Results", figsize=(10, 8))
            plt.clf()

            ax1 = fig.add_subplot(2, 2, 1)
            ax1.hist(v, bins=20)
            ax1.axvline(ke_est, color='r', linewidth=2)
            ax1.set_xlabel("ke estimate"); ax1.set_ylabel("Count")
            ax1.set_title("Distribution of ke Estimates")

            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(v, 'o-')
            ax2.axhline(ke_est, color='r', linewidth=2)
            ax2.set_xlabel("Window index"); ax2.set_ylabel("ke estimate")
            ax2.set_title("ke Estimates by Window")

            ax3 = fig.add_subplot(2, 2, 3)
            ax3.scatter(w, v)
            ax3.set_xlabel("Window weight"); ax3.set_ylabel("ke estimate")
            ax3.set_title("Estimates vs Weights")

            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis("off")
            ax4.text(0.1, 0.9, f"Final ke: {ke_est:.3f}", fontsize=14)
            ax4.text(0.1, 0.7, f"Std: {ke_std:.3f}", fontsize=12)
            ax4.text(0.1, 0.5, f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]", fontsize=12)
            ax4.text(0.1, 0.3, f"N windows: {len(v)}", fontsize=12)
            fig.suptitle("Standalone ke Estimation")

            # plt.show()  # optional

    return float(ke_est), results


# ======================================================================
# Helper functions (direct translations)
# ======================================================================

def _detectDoseChanges(A_obs: np.ndarray, min_change: float) -> np.ndarray:
    """Detect significant dose changes (boolean mask)"""
    N, T = A_obs.shape
    dose_changes = np.zeros((N, T), dtype=bool)
    for i in range(N):
        for t in range(1, T):
            if abs(A_obs[i, t] - A_obs[i, t - 1]) >= min_change:
                dose_changes[i, t] = True
    return dose_changes


def _extractResponseWindows(L_obs: np.ndarray, A_obs: np.ndarray,
                            dose_changes: np.ndarray, window_size: int) -> List[Dict[str, Any]]:
    """Extract windows around dose changes."""
    windows: List[Dict[str, Any]] = []
    N, T = L_obs.shape

    for i in range(N):
        change_times = np.where(dose_changes[i, :])[0]
        for ct in change_times:
            if ct + window_size <= T:
                t_win = np.arange(ct, ct + window_size, dtype=int)

                L_win = L_obs[i, t_win]
                A_win = A_obs[i, t_win]

                # At least 80% valid L
                if np.sum(~np.isnan(L_win)) >= window_size * 0.8:
                    win = {
                        "L": L_win.copy(),
                        "A": A_win.copy(),
                        "t": t_win.copy(),
                        "patient": int(i + 1),          # MATLAB-style 1-based patient id (informational)
                        "change_time": int(ct),
                        "dose_jump": float(A_obs[i, ct] - A_obs[i, ct - 1]) if ct > 0 else float(A_obs[i, ct]),
                    }
                    windows.append(win)
    return windows


def _estimateKeFromWindow(win: Dict[str, Any], C: float, g: float,
                          ke_range: np.ndarray, dt: float) -> Tuple[float, float]:
    """Estimate ke from a single response window via coarse grid + bounded refine."""
    from scipy.optimize import minimize_scalar

    def obj(ke_val: float) -> float:
        return _computeWindowError(ke_val, win, C, g, dt)

    # Coarse grid search
    ke_grid = np.linspace(ke_range[0], ke_range[1], 20)
    errors = np.array([obj(kev) for kev in ke_grid])
    min_idx = int(np.argmin(errors))
    ke_coarse = float(ke_grid[min_idx])
    min_error = float(errors[min_idx])

    # Refine around coarse min within ±0.1, clipped to range
    low = max(ke_range[0], ke_coarse - 0.1)
    high = min(ke_range[1], ke_coarse + 0.1)
    res = minimize_scalar(obj, bounds=(low, high), method="bounded")
    ke_est = float(res.x) if res.success else ke_coarse

    # Weight: signal strength × fit quality
    signal_strength = abs(float(win["dose_jump"]))
    fit_quality = 1.0 / (1.0 + min(min_error, obj(ke_est)))
    weight = signal_strength * fit_quality

    return ke_est, weight


def _smooth_nanaware(x: np.ndarray, span: int) -> np.ndarray:
    """NaN-aware moving average (like MATLAB smooth with small span)."""
    x = np.asarray(x, dtype=float)
    if span <= 1 or x.size < 2:
        return x.copy()
    w = np.ones(span, dtype=float)
    valid = np.isfinite(x).astype(float)
    num = np.convolve(np.nan_to_num(x, nan=0.0), w, mode="same")
    den = np.convolve(valid, w, mode="same")
    out = np.divide(num, den, out=np.full_like(x, np.nan), where=den > 0)
    return out


def _computeWindowError(ke_val: float, win: Dict[str, Any], C: float, g: float, dt: float) -> float:
    """Compute robust MAE on L within a window for a given ke."""
    L = np.asarray(win["L"], dtype=float)
    A = np.asarray(win["A"], dtype=float)
    T = L.size

    # Simulate drug concentration X with this ke
    X = np.zeros(T, dtype=float)
    for t in range(1, T):
        X[t] = ke_val * X[t - 1] + A[t]

    # Estimate L0 assuming this ke (simple inverse)
    L0_est = np.zeros(T, dtype=float)
    for t in range(T):
        if X[t] > 0:
            sX = 1.0 - 1.0 / (((C / X[t]) ** g) + 1.0)
            if 0.1 < sX < 0.9:
                L0_est[t] = L[t] / sX
            else:
                L0_est[t] = L[t]
        else:
            L0_est[t] = L[t]

    # Smooth L0 (span=3) if enough valid points
    if np.sum(np.isfinite(L0_est)) > 3:
        L0_est = _smooth_nanaware(L0_est, span=3)

    # Predict L using estimated L0 and given ke
    L_pred = np.zeros(T, dtype=float)
    for t in range(T):
        if X[t] > 0:
            sX = 1.0 - 1.0 / (((C / X[t]) ** g) + 1.0)
            L_pred[t] = L0_est[t] * sX
        else:
            L_pred[t] = L0_est[t]

    # Robust MAE on relative error
    valid = np.isfinite(L) & np.isfinite(L_pred) & (L > 0)
    if np.sum(valid) > 3:
        return float(np.median(np.abs(L[valid] - L_pred[valid]) / L[valid]))
    else:
        return float(np.inf)


def _weightedMedian(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median (exactly mirrors MATLAB logic used)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return np.nan
    if np.any(np.isfinite(weights)) and weights.sum() > 0:
        w = weights.copy()
    else:
        w = np.ones_like(values)

    order = np.argsort(values, kind="mergesort")
    v_sorted = values[order]
    w_sorted = w[order]
    w_norm = w_sorted / np.sum(w_sorted)
    cdf = np.cumsum(w_norm)
    idx = np.searchsorted(cdf, 0.5, side="left")
    if idx >= v_sorted.size:
        idx = v_sorted.size - 1
    return float(v_sorted[idx])


def _assemble_results(ke_est: float, ke_std: float, ci_95: np.ndarray,
                      all_estimates: List[float], weights: List[float],
                      n_windows: int, method: str, dose_changes_mask: np.ndarray) -> Dict[str, Any]:
    """Pack results struct/dict mirroring MATLAB fields."""
    total_changes = float(dose_changes_mask.sum())
    convergence = (len(all_estimates) / total_changes) if total_changes > 0 else np.nan
    results: Dict[str, Any] = {
        "ke_est": ke_est,
        "ke_std": ke_std,
        "ke_ci95": np.asarray(ci_95, dtype=float),
        "all_estimates": np.asarray(all_estimates, dtype=float),
        "weights": np.asarray(weights, dtype=float),
        "n_windows": n_windows,
        "method": method,
        "cv": (ke_std / ke_est) if (ke_est not in (0, np.nan) and np.isfinite(ke_est)) else np.nan,
        "convergence": convergence,
    }
    return results
