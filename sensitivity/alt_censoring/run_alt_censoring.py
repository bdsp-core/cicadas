"""run_alt_censoring.py -- Python port of run_alt_censoring.m.

Alternative-censoring sensitivity. Regenerates observational data under
four censoring regimes and reports recovered ATE vs RCT truth.

Outputs (written next to this script):
    alt_censoring_results.mat
    Fig_alt_censoring.pdf
    alt_censoring_summary.txt
"""

import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # <repo>/cicadas
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")

sys.path.insert(0, _PYTHON_DIR)
np.random.seed(0)

from fcnGeneratePatientParameters import fcnGeneratePatientParameters  # noqa: E402
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories  # noqa: E402
from fcnBiasedAssignmentProb import fcnBiasedAssignmentProb  # noqa: E402
from fcnPlotKM import fcnPlotKM  # noqa: E402
from fcnEstimateDeathParms import fcnEstimateDeathParms  # noqa: E402
from fcnEstimateParmsL import fcnEstimateParmsL  # noqa: E402
from fcnEstimateParmsPKPD import fcnEstimateParmsPKPD  # noqa: E402
from fcnSimulate_N_Patients import fcnSimulate_N_Patients  # noqa: E402


def main() -> int:
    # --- Baseline params ---
    N = 1000
    th = 0.1
    parmsControl = np.array([10.0, 50.0])
    np.random.seed(0)
    age, sofa, C, g, _ = fcnGeneratePatientParameters(
        N, "TargetCMean", 3, "TargetGMean", 4, "CV", 0.1,
    )
    ke = 0.5
    parmsY = np.array([-7.0, 0.3, 20.0, 5.0])
    parmsL = np.array([0.25, 1.0, 0.15, 0.05, 0.15, 0.03, 40.0])
    dt = 2.0
    t = np.arange(0.0, 168.0 + dt, dt)

    regimes = {
        "baseline": np.array([-5.0,  2.0, 0.1, -5.0,  2.0, 1.5]),
        "MCAR":     np.array([-6.0,  0.0, 0.0, -6.0,  0.0, 0.0]),
        "strong":   np.array([-3.0,  4.0, 0.5, -3.0,  4.0, 2.5]),
        "none":     np.array([-50.0, 0.0, 0.0, -50.0, 0.0, 0.0]),
    }
    names = list(regimes.keys())

    # --- RCT truth (no censoring in RCT mode) ---
    L0_rct = fcnGenerateStochasticTrajectories(t, parmsL, N)
    T1 = fcnSimulate_N_Patients(
        N, 1, 0.5 * np.ones(N), th, C, g, ke, L0_rct,
        parmsControl, parmsY, regimes["baseline"], age, sofa,
    )
    s0_true, s1_true, _, _ = fcnPlotKM(T1)
    ate_truth = float(s1_true[-1] - s0_true[-1])
    print(f"RCT truth ATE = {ate_truth:.3f}")

    results = {}
    t_all = time.time()
    for i, nm in enumerate(names):
        parmsV_i = regimes[nm]
        np.random.seed(1 + (i + 1))

        L0 = fcnGenerateStochasticTrajectories(t, parmsL, N)
        treatProb = fcnBiasedAssignmentProb(age, sofa, L0[:, :5])
        T0 = fcnSimulate_N_Patients(
            N, 0, treatProb, th, C, g, ke, L0,
            parmsControl, parmsY, parmsV_i, age, sofa,
        )

        n_cens = int((T0["V"] > 0).sum())
        total_rows = len(T0)
        cens_rate = n_cens / max(total_rows, 1)
        print(f"\nRegime \"{nm}\": censoring rate = {100*cens_rate:.1f}%")

        ate = float("nan"); bias = float("nan")
        try:
            parmsY_est = fcnEstimateDeathParms(T0)
            parmsL_est, LL, AA, age_e, sofa_e, t_e = fcnEstimateParmsL(T0)
            ke_est, C_est, g_est, _ = fcnEstimateParmsPKPD(
                parmsL_est, LL, AA, age_e, sofa_e, t_e,
            )
            parmsL_est = np.asarray(parmsL_est, dtype=float).reshape(-1)[:7]
            L0_est = fcnGenerateStochasticTrajectories(t, parmsL_est, N)
            T1_est = fcnSimulate_N_Patients(
                N, 1, 0.5 * np.ones(N), th, C_est, g_est, ke_est, L0_est,
                parmsControl, parmsY_est, np.zeros(6), age, sofa,
            )
            s0_gf, s1_gf, _, _ = fcnPlotKM(T1_est)
            ate = float(s1_gf[-1] - s0_gf[-1])
            bias = ate - ate_truth
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] {e}")
            traceback.print_exc()

        results[nm] = {"ate": ate, "bias": bias, "cens_rate": cens_rate}
        print(f"  ATE = {ate:+.3f} (bias {bias:+.3f})")

    print(f"\nTotal time: {(time.time() - t_all)/60:.1f} min")

    # --- Save .mat (flat keys for SciPy compatibility) ---
    mat_payload = {
        "ate_truth": ate_truth,
        "names": np.array(names, dtype=object),
    }
    for nm in names:
        mat_payload[f"ate_{nm}"]       = results[nm]["ate"]
        mat_payload[f"bias_{nm}"]      = results[nm]["bias"]
        mat_payload[f"cens_rate_{nm}"] = results[nm]["cens_rate"]
        mat_payload[f"parmsV_{nm}"]    = regimes[nm]
    out_file = os.path.join(_SCRIPT_DIR, "alt_censoring_results.mat")
    savemat(out_file, mat_payload, do_compression=True)
    print(f"Saved {out_file}")

    # --- Figure ---
    atess = np.array([100 * results[nm]["ate"]       for nm in names])
    cens_rates = np.array([100 * results[nm]["cens_rate"] for nm in names])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, atess)
    axes[0].axhline(100 * ate_truth, color="red", linestyle="--", linewidth=1.5,
                    label=f"RCT truth = {100*ate_truth:.1f}")
    axes[0].set_ylabel("G-formula ATE (pp)")
    axes[0].set_title("Recovered ATE across censoring regimes")
    axes[0].legend()

    axes[1].bar(names, cens_rates)
    axes[1].set_ylabel("Censoring rate (% of person-time)")
    axes[1].set_title("Censoring rate by regime")

    fig_path = os.path.join(_SCRIPT_DIR, "Fig_alt_censoring.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {fig_path}")

    # --- Text summary ---
    txt_path = os.path.join(_SCRIPT_DIR, "alt_censoring_summary.txt")
    with open(txt_path, "w") as fid:
        fid.write("Alternative-censoring sensitivity\n")
        fid.write(f"RCT truth ATE: {ate_truth:+.3f}\n\n")
        fid.write(f"{'regime':<10} {'cens rate':<14} {'ATE':<14} {'bias':<14}\n")
        for nm in names:
            fid.write(
                f"{nm:<10} {results[nm]['cens_rate']:<14.3f} "
                f"{results[nm]['ate']:<+14.3f} {results[nm]['bias']:<+14.3f}\n"
            )
    print(f"Saved text summary to {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
