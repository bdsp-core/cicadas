"""run_nuc_sensitivity.py -- Python port of run_nuc_sensitivity.m.

NUC (no unmeasured confounding) sensitivity analysis for CICADAS.
Injects a hidden binary confounder U ~ Bernoulli(0.5) that influences both
treatment assignment and mortality. Runs the CICADAS g-formula pipeline
without U and reports the bias in the recovered ATE as a function of
(delta_A, delta_Y).

Also computes the Ding-VanderWeele E-value for the unconfounded ATE and
reports the tipping-point delta_Y at which the ATE sign flips.

Outputs (written next to this script):
    nuc_results.mat
    Fig_NUC_sensitivity.pdf
    nuc_summary.txt

Run:
    cd sensitivity/nuc_injection && python3 run_nuc_sensitivity.py

A sanity-check mode can be enabled by setting the env var
CICADAS_NUC_QUICK=1, which drops N to 400 and uses 1 seed per cell.
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

# Locate project layout
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # <repo>/cicadas
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")

# Make CICADAS python helpers importable
sys.path.insert(0, _PYTHON_DIR)
# And this directory (for fcnSimulate_N_Patients_withU)
sys.path.insert(0, _SCRIPT_DIR)

np.random.seed(0)

# --- helpers ---
from fcnGeneratePatientParameters import fcnGeneratePatientParameters  # noqa: E402
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories  # noqa: E402
from fcnBiasedAssignmentProb import fcnBiasedAssignmentProb  # noqa: E402
from fcnPlotKM import fcnPlotKM  # noqa: E402
from fcnEstimateDeathParms import fcnEstimateDeathParms  # noqa: E402
from fcnEstimateParmsL import fcnEstimateParmsL  # noqa: E402
from fcnEstimateParmsPKPD import fcnEstimateParmsPKPD  # noqa: E402
from fcnSimulate_N_Patients_withU import fcnSimulate_N_Patients_withU  # noqa: E402


def _naive_km_ate(T0: pd.DataFrame) -> float:
    """Naive per-patient baseline-Rx KM (returns s1(end) - s0(end))."""
    uids = np.unique(T0["sid"].values)
    tt1, ev1, tt0, ev0 = [], [], [], []
    for uid in uids:
        d = T0[T0["sid"] == uid].sort_values("t")
        y = d["Y"].to_numpy()
        if (y > 0).any():
            idx = int(np.argmax(y > 0))
            tm = float(d["t"].iloc[idx]); ev = 1
        else:
            tm = float(d["t"].max()); ev = 0
        if int(d["Rx"].iloc[0]) == 1:
            tt1.append(tm); ev1.append(ev)
        else:
            tt0.append(tm); ev0.append(ev)

    def _km_end(times, events):
        times = np.asarray(times, float); events = np.asarray(events, int)
        order = np.argsort(times, kind="mergesort")
        times, events = times[order], events[order]
        ue = np.unique(times[events == 1])
        if ue.size == 0:
            return 1.0
        n = times.size; at_risk = n; S = 1.0; idx = 0
        for te in ue:
            while idx < n and times[idx] < te:
                at_risk -= 1; idx += 1
            d_i = int(np.sum((times == te) & (events == 1)))
            if at_risk > 0:
                S *= (1.0 - d_i / at_risk)
            k = 0
            while idx + k < n and times[idx + k] == te:
                k += 1
            at_risk -= k; idx += k
        return S

    s1_end = _km_end(tt1, ev1)
    s0_end = _km_end(tt0, ev0)
    return s1_end - s0_end


def main() -> int:
    # --- check for required inputs (not strictly needed since we regenerate DGP,
    # but fcnEstimateParmsPKPD / fcnGetPKPD_parms_est may touch files). We warn
    # gracefully if anything vital is missing.
    # Quick mode is now the DEFAULT (N=400, 1 seed, ~6 min) because the full
    # 3-5 hour run is impractical for most interactive verification. Set
    # CICADAS_NUC_FULL=1 to run the full 5x5x3-seed N=1500 grid (~3-5 hours).
    full = os.environ.get("CICADAS_NUC_FULL", "0") not in ("", "0", "false", "False")
    # CICADAS_NUC_QUICK retained for backward compat but is a no-op now.
    _ = os.environ.get("CICADAS_NUC_QUICK", "0")
    quick = not full

    # --- Baseline params (copied from a0_GenerateTrialData.m) ---
    N = 400 if quick else 1500
    th = 0.1
    ki = 10; Amax = 50
    parmsControl = np.array([ki, Amax], dtype=float)
    ke = 0.5
    parmsY = np.array([-7.0, 0.3, 20.0, 5.0])
    parmsV = np.array([-5.0, 2.0, 0.1, -5.0, 2.0, 1.5])
    parmsL = np.array([0.25, 1.0, 0.15, 0.05, 0.15, 0.03, 40.0])
    dt = 2.0
    t = np.arange(0.0, 168.0 + dt, dt)

    # --- Sensitivity grid ---
    delta_A_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    delta_Y_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    n_seeds = 1 if quick else 3

    nA = delta_A_vals.size; nY = delta_Y_vals.size
    ate_rct_all      = np.full((nA, nY, n_seeds), np.nan)
    ate_naive_all    = np.full((nA, nY, n_seeds), np.nan)
    ate_gformula_all = np.full((nA, nY, n_seeds), np.nan)

    # Stash zero-perturbation honest curves for E-value
    s0_true_honest = None
    s1_true_honest = None

    total = nA * nY * n_seeds
    k = 0
    t_all = time.time()

    for ia, dA in enumerate(delta_A_vals):
        for iY, dY in enumerate(delta_Y_vals):
            for seed in range(1, n_seeds + 1):
                k += 1
                seed_val = 1000 * seed + 100 * (ia + 1) + (iY + 1)
                np.random.seed(seed_val)
                print(f"\n[{k}/{total}] dA={dA:.1f}, dY={dY:.1f}, seed={seed} ...")
                t_cond = time.time()

                age, sofa, C, g, _ = fcnGeneratePatientParameters(
                    N, "TargetCMean", 3, "TargetGMean", 4, "CV", 0.1,
                )
                U = (np.random.rand(N) < 0.5).astype(float)
                u_shift_Y = U * float(dY)

                L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)

                # --- RCT truth under this U shift on Y ---
                T1 = fcnSimulate_N_Patients_withU(
                    N, 1, 0.5 * np.ones(N), th, C, g, ke, L0,
                    parmsControl, parmsY, parmsV, age, sofa, u_shift_Y,
                )
                s0_true, s1_true, _, _ = fcnPlotKM(T1)
                ate_rct_all[ia, iY, seed - 1] = s1_true[-1] - s0_true[-1]
                if ia == 0 and iY == 0 and seed == 1:
                    s0_true_honest = s0_true
                    s1_true_honest = s1_true

                # --- Observational with U-driven treatment and Y shift ---
                treatProb_base = fcnBiasedAssignmentProb(age, sofa, L0[:, :5])
                p_clip = np.clip(treatProb_base, 1e-6, 1 - 1e-6)
                logit_obs = np.log(p_clip / (1 - p_clip)) + U * float(dA)
                treatProb_obs = 1.0 / (1.0 + np.exp(-logit_obs))
                T0 = fcnSimulate_N_Patients_withU(
                    N, 0, treatProb_obs, th, C, g, ke, L0,
                    parmsControl, parmsY, parmsV, age, sofa, u_shift_Y,
                )
                ate_naive_all[ia, iY, seed - 1] = _naive_km_ate(T0)

                # --- G-formula (U NOT available as covariate) ---
                try:
                    parmsY_est = fcnEstimateDeathParms(T0)
                    parmsL_est, LL, AA, age_e, sofa_e, t_e = fcnEstimateParmsL(T0)
                    ke_est, C_est, g_est, _ = fcnEstimateParmsPKPD(
                        parmsL_est, LL, AA, age_e, sofa_e, t_e,
                    )
                    # pad/trim to 7 per CausalSurvivalAnalysis convention
                    parmsL_est = np.asarray(parmsL_est, dtype=float).reshape(-1)[:7]
                    L0_est = fcnGenerateStochasticTrajectories(t, parmsL_est, N)
                    T1_est = fcnSimulate_N_Patients_withU(
                        N, 1, 0.5 * np.ones(N), th, C_est, g_est, ke_est, L0_est,
                        parmsControl, parmsY_est, np.zeros(6), age, sofa,
                        np.zeros(N),
                    )
                    s0_gf, s1_gf, _, _ = fcnPlotKM(T1_est)
                    ate_gformula_all[ia, iY, seed - 1] = s1_gf[-1] - s0_gf[-1]
                except Exception as e:  # noqa: BLE001
                    print(f"  [WARN] g-formula failed: {e}")
                    traceback.print_exc()

                print(
                    f"  ATE: RCT={ate_rct_all[ia,iY,seed-1]:+.3f}, "
                    f"naive={ate_naive_all[ia,iY,seed-1]:+.3f}, "
                    f"gf={ate_gformula_all[ia,iY,seed-1]:+.3f} "
                    f"({time.time()-t_cond:.1f}s)"
                )

    print(f"\nSweep total: {(time.time() - t_all)/60:.1f} min")

    # --- Averages ---
    ate_rct      = np.nanmean(ate_rct_all, axis=2)
    ate_naive    = np.nanmean(ate_naive_all, axis=2)
    ate_gformula = np.nanmean(ate_gformula_all, axis=2)
    bias_gf      = ate_gformula - ate_rct
    with np.errstate(invalid="ignore"):
        bias_gf_sd = np.nanstd(ate_gformula_all - ate_rct_all, axis=2, ddof=0)

    # --- E-value (Ding-VanderWeele) on honest ATE ---
    if s1_true_honest is not None and s0_true_honest is not None:
        mort_t = 1.0 - float(s1_true_honest[-1])
        mort_u = 1.0 - float(s0_true_honest[-1])
        if mort_t > 0 and mort_u > 0:
            RR = mort_u / mort_t  # >1 means treatment protective
            if RR < 1.0:
                RR = 1.0 / RR
            evalue = RR + np.sqrt(RR * (RR - 1.0))
        else:
            evalue = float("nan")
    else:
        evalue = float("nan")

    # --- Tipping-point delta_Y at delta_A = 0 ---
    sign_flip_dY = float("nan")
    ref_sign = np.sign(ate_rct[0, 0]) if not np.isnan(ate_rct[0, 0]) else 0
    for iY in range(nY):
        val = ate_gformula[0, iY]
        if not np.isnan(val) and np.sign(val) != ref_sign and ref_sign != 0:
            sign_flip_dY = float(delta_Y_vals[iY])
            break

    # --- Save .mat ---
    out_file = os.path.join(_SCRIPT_DIR, "nuc_results.mat")
    savemat(
        out_file,
        {
            "delta_A_vals": delta_A_vals,
            "delta_Y_vals": delta_Y_vals,
            "n_seeds": n_seeds,
            "ate_rct": ate_rct,
            "ate_naive": ate_naive,
            "ate_gformula": ate_gformula,
            "ate_rct_all": ate_rct_all,
            "ate_naive_all": ate_naive_all,
            "ate_gformula_all": ate_gformula_all,
            "bias_gf": bias_gf,
            "bias_gf_sd": bias_gf_sd,
            "evalue": evalue,
            "sign_flip_dY": sign_flip_dY,
            "N": N,
            "parmsY": parmsY,
            "parmsV": parmsV,
        },
        do_compression=True,
    )
    print(f"Saved results to {out_file}")

    # --- Figure: two heatmaps (bias, ATE) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _redblue(n=256):
        half = n // 2
        r = np.concatenate([np.linspace(0, 1, half), np.ones(half)])
        gch = np.concatenate([np.linspace(0, 1, half), np.linspace(1, 0, half)])
        b = np.concatenate([np.ones(half), np.linspace(1, 0, half)])
        from matplotlib.colors import ListedColormap
        return ListedColormap(np.c_[r, gch, b])
    cmap = _redblue()

    # Orient so rows = delta_A, cols = delta_Y via transpose (same as MATLAB .')
    bias_pp = 100.0 * bias_gf.T  # now shape (nY, nA)
    ate_pp  = 100.0 * ate_gformula.T

    vmax = np.nanmax(np.abs(bias_pp))
    im0 = axes[0].imshow(
        bias_pp, origin="lower", aspect="auto",
        extent=(delta_A_vals[0] - 0.25, delta_A_vals[-1] + 0.25,
                delta_Y_vals[0] - 0.25, delta_Y_vals[-1] + 0.25),
        cmap=cmap, vmin=-vmax if vmax > 0 else -1, vmax=vmax if vmax > 0 else 1,
    )
    axes[0].set_xticks(delta_A_vals)
    axes[0].set_yticks(delta_Y_vals)
    axes[0].set_xlabel(r"$\delta_A$ (logit shift on treatment)")
    axes[0].set_ylabel(r"$\delta_Y$ (logit shift on mortality)")
    axes[0].set_title(f"G-formula bias in ATE (pp)\nN={N} patients")
    fig.colorbar(im0, ax=axes[0])
    for ia, dA in enumerate(delta_A_vals):
        for iY, dY in enumerate(delta_Y_vals):
            if not np.isnan(bias_gf[ia, iY]):
                axes[0].text(dA, dY, f"{100*bias_gf[ia, iY]:+.1f}",
                             ha="center", va="center", fontsize=9)

    vmax2 = np.nanmax(np.abs(ate_pp)) if np.isfinite(np.nanmax(np.abs(ate_pp))) else 1.0
    im1 = axes[1].imshow(
        ate_pp, origin="lower", aspect="auto",
        extent=(delta_A_vals[0] - 0.25, delta_A_vals[-1] + 0.25,
                delta_Y_vals[0] - 0.25, delta_Y_vals[-1] + 0.25),
        cmap=cmap, vmin=-vmax2, vmax=vmax2,
    )
    axes[1].set_xticks(delta_A_vals)
    axes[1].set_yticks(delta_Y_vals)
    axes[1].set_xlabel(r"$\delta_A$")
    axes[1].set_ylabel(r"$\delta_Y$")
    axes[1].set_title("G-formula ATE (pp)")
    fig.colorbar(im1, ax=axes[1])
    for ia, dA in enumerate(delta_A_vals):
        for iY, dY in enumerate(delta_Y_vals):
            if not np.isnan(ate_gformula[ia, iY]):
                axes[1].text(dA, dY, f"{100*ate_gformula[ia, iY]:+.1f}",
                             ha="center", va="center", fontsize=9)

    fig_path = os.path.join(_SCRIPT_DIR, "Fig_NUC_sensitivity.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {fig_path}")

    # --- Text summary ---
    txt_file = os.path.join(_SCRIPT_DIR, "nuc_summary.txt")
    with open(txt_file, "w") as fid:
        fid.write("NUC sensitivity analysis\n")
        fid.write("========================\n")
        fid.write(f"N = {N} patients per arm\n")
        fid.write(f"delta_A grid: {' '.join(f'{v:g}' for v in delta_A_vals)}\n")
        fid.write(f"delta_Y grid: {' '.join(f'{v:g}' for v in delta_Y_vals)}\n")
        fid.write(f"\nE-value (honest ATE, delta_A=delta_Y=0): {evalue:.2f}\n")
        if not np.isnan(sign_flip_dY):
            fid.write(f"Tipping-point delta_Y (at delta_A=0) for ATE sign flip: {sign_flip_dY:.2f}\n")
        else:
            fid.write("No sign flip observed in the tested delta_Y range.\n")
        fid.write("\nBias heatmap (rows=delta_A, cols=delta_Y):\n")
        for ia, dA in enumerate(delta_A_vals):
            row = " ".join(f"{100*bias_gf[ia, iY]:+5.1f}" for iY in range(nY))
            fid.write(f"{dA:.2f} | {row}\n")
    print(f"Saved summary to {txt_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
