"""
Sampling distribution of the CICADAS g-formula estimator (Option A+).

Purpose (Intelligence-Based Medicine revision round 2, 2026-09):
  The manuscript previously claimed "100% coverage of the true RCT values across
  all endpoints". That statement described three non-independent intervals from a
  single cohort, not coverage in the repeated-sampling sense. A full nested
  coverage study (R cohorts x B bootstrap replicates) is infeasible: one g-formula
  fit takes ~64 s, so R=100, B=100 would be ~180 hours.

  This script instead establishes the SAMPLING DISTRIBUTION of the estimator by
  re-running the full generate -> estimate -> emulate pipeline across R independent
  cohorts. Comparing the empirical standard deviation of the estimator against the
  bootstrap standard error from a single cohort is a standard calibration check: if
  the bootstrap SE matches the empirical SD, the uncertainty quantification is
  calibrated, which is the substantive claim the original sentence was reaching for.

Per replication:
  1. generate a randomized cohort  -> ground-truth ATE (no modelling)
  2. generate an observational cohort with confounded assignment + informative censoring
  3. estimate the mortality, natural-history, and PK/PD models FROM THE OBSERVATIONAL
     COHORT ONLY (all estimated -- no true parameters are used anywhere)
  4. emulate a randomized trial by forward Monte Carlo from the fitted models
  5. record the g-formula ATE against the ground truth

Usage:  python coverage_py.py [R] [N]
Output: coverage_results.csv
"""

import sys, os, time, contextlib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from fcnGeneratePatientParameters import fcnGeneratePatientParameters
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnSimulate_N_Patients import fcnSimulate_N_Patients
from fcnBiasedAssignmentProb import fcnBiasedAssignmentProb
from fcnPlotKM import fcnPlotKM
from fcnEstimateDeathParms import fcnEstimateDeathParms
from fcnEstimateParmsL import fcnEstimateParmsL
from scipy.io import loadmat

# Load the a1 PK/PD fit once, by absolute path, and apply the mixed-effects model
# per replication exactly as fcnGetPKPD_parms_est does inside a2.
_m = loadmat(os.path.join(HERE, "..", "PKPD_estimation_results.mat"),
             squeeze_me=True, struct_as_record=False)["results"]
KE_EST = float(_m.twostage_corr.ke_corrected)
PD_EST = np.asarray(_m.twostage_corr.theta, float).ravel()[:6]

def pkpd_for(age, sofa):
    an = (age - np.mean(age)) / np.std(age)
    sn = (sofa - np.mean(sofa)) / np.std(sofa)
    C = PD_EST[0] + PD_EST[1]*an + PD_EST[2]*sn
    g = PD_EST[3] + PD_EST[4]*an + PD_EST[5]*sn
    return C, g, KE_EST

TH = 0.1
PARMS_CONTROL = np.array([10, 50])
PARMS_Y = np.array([-7.0, 0.3, 20.0, 5.0])
PARMS_V = np.array([-5.0, 2.0, 0.1, -5.0, 2.0, 1.5])
PARMS_L = np.array([0.25, 1.0, 0.15, 0.05, 0.15, 0.03, 40.0])
KE, DT = 0.5, 2.0
T_GRID = np.arange(0.0, 168.0 + DT, DT)
QUIET = lambda: contextlib.redirect_stdout(open(os.devnull, "w"))


def one_replication(seed, N):
    np.random.seed(seed)
    age, sofa, C, g, _ = fcnGeneratePatientParameters(
        N, "TargetCMean", 3, "TargetGMean", 4, "CV", 0.1)

    cohorts = {}
    for RCT in [0, 1]:
        L0 = fcnGenerateStochasticTrajectories(T_GRID, PARMS_L, N)
        if RCT == 1:
            treatProb = np.full(N, 0.5)
        else:
            with QUIET():
                treatProb = fcnBiasedAssignmentProb(age, sofa, L0[:, :5])
        cohorts[RCT] = fcnSimulate_N_Patients(
            N, RCT, treatProb, TH, C, g, KE, L0,
            PARMS_CONTROL, PARMS_Y, PARMS_V, age, sofa)

    # --- ground truth: randomized, uncensored, no modelling ---
    s0t, s1t, _, _ = fcnPlotKM(cohorts[1])
    ate_true = float(s1t[-1] - s0t[-1])

    # --- g-formula, mirroring a2_CausalSurvivalAnalysis.m ---
    # The mortality and natural-history models are re-estimated from each new
    # observational cohort. The PK/PD fit is held at its point estimate, exactly as
    # in the paper's pipeline, where a1 is run once on a dedicated dose-switching
    # cohort and a2 consumes its output. The resulting spread is therefore a LOWER
    # BOUND on total sampling variability, since PK/PD uncertainty is not propagated.
    T0 = cohorts[0]
    with QUIET():
        pY = fcnEstimateDeathParms(T0)
        pL, LL, AA, age_e, sofa_e, tt = fcnEstimateParmsL(T0)
        C_e, g_e, ke_e = pkpd_for(np.asarray(age_e,float), np.asarray(sofa_e,float))

        pL = np.asarray(pL, float).ravel()[:7]
        L0_est = fcnGenerateStochasticTrajectories(T_GRID, pL, N)
        T1_est = fcnSimulate_N_Patients(
            N, 1, np.full(N, 0.5), TH, C_e, g_e, ke_e, L0_est,
            PARMS_CONTROL, pY, np.zeros(6), age, sofa)
        s0g, s1g, _, _ = fcnPlotKM(T1_est)
    ate_gf = float(s1g[-1] - s0g[-1])

    return dict(seed=seed, ate_true=ate_true, ate_gformula=ate_gf,
                err_vs_realized=ate_gf - ate_true,
                s0_gf=float(s0g[-1]), s1_gf=float(s1g[-1]),
                s0_true=float(s0t[-1]), s1_true=float(s1t[-1]))


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    out = os.path.join(HERE, "coverage_results.csv")

    rows, t0 = [], time.time()
    for seed in range(R):
        try:
            r = one_replication(seed, N)
            rows.append(r)
            el = time.time() - t0
            print(f"seed {seed:3d}  truth {100*r['ate_true']:+6.2f}%  "
                  f"g-formula {100*r['ate_gformula']:+6.2f}%  "
                  f"err {100*r['err_vs_realized']:+6.2f}pp   "
                  f"[{el/60:.1f} min, ~{el/len(rows)*(R-len(rows))/60:.0f} min left]",
                  flush=True)
        except Exception as e:
            print(f"seed {seed:3d}  FAILED: {type(e).__name__}: {e}", flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(out, index=False)   # checkpoint every seed

    df = pd.DataFrame(rows)
    if df.empty:
        print('no successful replications'); sys.exit(1)
    v = df.ate_gformula * 100
    print("\n" + "=" * 74)
    print(f"G-FORMULA SAMPLING DISTRIBUTION — {len(df)} successful replications, N={N}")
    print("=" * 74)
    print(f"  g-formula ATE   mean {v.mean():+7.3f}  SD {v.std(ddof=1):6.3f}  "
          f"MCSE {v.std(ddof=1)/np.sqrt(len(v)):.3f}")
    t = df.ate_true * 100
    print(f"  ground truth    mean {t.mean():+7.3f}  SD {t.std(ddof=1):6.3f}")
    print(f"  bias vs population estimand: {v.mean()-t.mean():+.3f} pp")
    print(f"\n  ** empirical sampling SD of the estimator = {v.std(ddof=1):.3f} pp **")
    print(f"     compare against the bootstrap SE from a single cohort")
    print(f"saved -> {out}")
