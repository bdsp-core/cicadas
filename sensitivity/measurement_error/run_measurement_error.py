"""run_measurement_error.py -- Python port of run_measurement_error.m.

Measurement-error sensitivity: add Gaussian noise to observed L_t in
trialData0.csv at a grid of sigmas, re-run the g-formula pipeline, and
report the recovered ATE vs the RCT truth from trialData1.csv.

Outputs (written next to this script):
    measurement_error_results.mat
    Fig_measurement_error.pdf
    measurement_error_summary.txt
"""

import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # <repo>/cicadas
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")

sys.path.insert(0, _PYTHON_DIR)
np.random.seed(0)

from fcnPlotKM import fcnPlotKM  # noqa: E402
from fcnEstimateDeathParms import fcnEstimateDeathParms  # noqa: E402
from fcnEstimateParmsL import fcnEstimateParmsL  # noqa: E402
from fcnEstimateParmsPKPD import fcnEstimateParmsPKPD  # noqa: E402
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories  # noqa: E402
from fcnSimulate_N_Patients import fcnSimulate_N_Patients  # noqa: E402


def main() -> int:
    p_T0 = os.path.join(_REPO_ROOT, "trialData0.csv")
    p_T1 = os.path.join(_REPO_ROOT, "trialData1.csv")
    for p in (p_T0, p_T1):
        if not os.path.exists(p):
            print(f"[WARN] missing required input {p}; exiting cleanly.")
            return 0

    T0_base = pd.read_csv(p_T0)
    T1 = pd.read_csv(p_T1)
    s0_true, s1_true, _, _ = fcnPlotKM(T1)
    ate_truth = float(s1_true[-1] - s0_true[-1])
    print(f"Baseline RCT truth ATE = {ate_truth:.3f}")

    sigma_vals = np.array([0.0, 0.01, 0.025, 0.05, 0.1, 0.2])
    n_seeds = 3

    ate_obs = np.full((sigma_vals.size, n_seeds), np.nan)
    bias_obs = np.full_like(ate_obs, np.nan)

    dt = 2.0
    t = np.arange(0.0, 168.0 + dt, dt)
    parmsControl = np.array([10.0, 50.0])
    th = 0.1

    t_all = time.time()
    for is_, sigma in enumerate(sigma_vals):
        for k in range(n_seeds):
            seed_val = 100 + 10 * (is_ + 1) + (k + 1)
            np.random.seed(seed_val)

            T0 = T0_base.copy()
            noise = sigma * np.random.randn(len(T0))
            T0["L"] = np.maximum(0.0, T0["L"].to_numpy() + noise)

            try:
                parmsY_est = fcnEstimateDeathParms(T0)
                parmsL_est, LL, AA, age_e, sofa_e, t_e = fcnEstimateParmsL(T0)
                ke_est, C_est, g_est, _ = fcnEstimateParmsPKPD(
                    parmsL_est, LL, AA, age_e, sofa_e, t_e,
                )
                parmsL_est = np.asarray(parmsL_est, dtype=float).reshape(-1)[:7]

                N = int(np.unique(T0["sid"].values).size)
                # age/sofa per MATLAB: first N rows' age/sofa (one value per sid,
                # but in MATLAB each row is a time step; take first N rows).
                age = T0["age"].to_numpy()[:N]
                sofa = T0["sofa"].to_numpy()[:N]

                L0_est = fcnGenerateStochasticTrajectories(t, parmsL_est, N)
                T1_est = fcnSimulate_N_Patients(
                    N, 1, 0.5 * np.ones(N), th, C_est, g_est, ke_est, L0_est,
                    parmsControl, parmsY_est, np.zeros(6), age, sofa,
                )
                s0_gf, s1_gf, _, _ = fcnPlotKM(T1_est)
                ate_obs[is_, k] = float(s1_gf[-1] - s0_gf[-1])
                bias_obs[is_, k] = ate_obs[is_, k] - ate_truth
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] sigma={sigma:.3f} seed={k+1}: {e}")
                traceback.print_exc()

            print(
                f"sigma={sigma:.3f} seed={k+1}: "
                f"ATE={ate_obs[is_, k]:.3f}, bias={bias_obs[is_, k]:+.3f}"
            )

    print(f"\nSweep total: {(time.time() - t_all)/60:.1f} min")

    ate_mean = np.nanmean(ate_obs, axis=1)
    ate_sd = np.nanstd(ate_obs, axis=1, ddof=0)
    bias_mean = np.nanmean(bias_obs, axis=1)

    out_file = os.path.join(_SCRIPT_DIR, "measurement_error_results.mat")
    savemat(
        out_file,
        {
            "sigma_vals": sigma_vals,
            "ate_obs": ate_obs,
            "bias_obs": bias_obs,
            "ate_mean": ate_mean,
            "ate_sd": ate_sd,
            "bias_mean": bias_mean,
            "ate_truth": ate_truth,
            "n_seeds": n_seeds,
        },
        do_compression=True,
    )
    print(f"Saved {out_file}")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(sigma_vals, 100 * ate_mean, yerr=100 * ate_sd,
                marker="o", linestyle="-", linewidth=2, markersize=8)
    ax.axhline(100 * ate_truth, color="red", linestyle="--", linewidth=1.5,
               label=f"RCT truth = {100*ate_truth:.1f}%")
    ax.set_xlabel(r"Gaussian noise SD on $L_t$ (disease-severity proxy)")
    ax.set_ylabel("G-formula ATE at 168 h (percentage points)")
    ax.set_title(
        "Measurement-error sensitivity of the CICADAS g-formula\n"
        rf"mean $\pm$ SD across {n_seeds} seeds per $\sigma$"
    )
    ax.grid(True)
    ax.tick_params(labelsize=11)
    ax.legend()
    fig_path = os.path.join(_SCRIPT_DIR, "Fig_measurement_error.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {fig_path}")

    # --- Text summary ---
    txt_path = os.path.join(_SCRIPT_DIR, "measurement_error_summary.txt")
    with open(txt_path, "w") as fid:
        fid.write("Measurement-error sensitivity (L_t perturbed with N(0, sigma^2))\n")
        fid.write(f"N seeds per sigma: {n_seeds}\n")
        fid.write(f"RCT truth ATE: {ate_truth:+.3f}\n\n")
        fid.write(f"{'sigma':<8} {'mean ATE':<12} {'SD ATE':<12} {'mean bias':<12}\n")
        for is_, sigma in enumerate(sigma_vals):
            fid.write(
                f"{sigma:<8.3f} {ate_mean[is_]:+12.3f} "
                f"{ate_sd[is_]:12.3f} {bias_mean[is_]:+12.3f}\n"
            )
    print(f"Saved text summary to {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
