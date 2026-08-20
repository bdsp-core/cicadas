"""
Multi-seed replication of the CICADAS headline result (Python pipeline).

Purpose (revision for Intelligence-Based Medicine, 2026-08):
  The manuscript reports the RCT ground-truth ATE and the naive Kaplan-Meier ATE
  as single draws from one simulated cohort (+14.8% and -4.8%, MATLAB rng(0)).
  The Python port at seed 0 gives +18.9% and +2.2% -- no sign reversal. This
  script establishes the sampling distribution of both quantities so we can say
  whether that gap is Monte Carlo error or a code discrepancy, and so the paper
  can report means with Monte Carlo SEs instead of single draws.

Replicates the RNG consumption order of generatetrialdata.py exactly:
  seed -> fcnGeneratePatientParameters -> for RCT in [0,1]:
          fcnGenerateStochasticTrajectories -> (fcnBiasedAssignmentProb) ->
          fcnSimulate_N_Patients

Usage:  python multiseed_py.py [n_seeds] [N]
Output: multiseed_py_results.csv
"""

import sys
import os
import contextlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from fcnGeneratePatientParameters import fcnGeneratePatientParameters
from fcn_generateStochasticTrajectories import fcnGenerateStochasticTrajectories
from fcnSimulate_N_Patients import fcnSimulate_N_Patients
from fcnBiasedAssignmentProb import fcnBiasedAssignmentProb
from fcnPlotKM import fcnPlotKM

# --- Simulation parameters: identical to generatetrialdata.py -----------------
TH = 0.1
KI, AMAX = 10, 50
KE = 0.5
PARMS_CONTROL = np.array([KI, AMAX])
PARMS_Y = np.array([-7.0, 0.3, 20.0, 5.0])
PARMS_V = np.array([-5.0, 2.0, 0.1, -5.0, 2.0, 1.5])
PARMS_L = np.array([0.25, 1.0, 0.15, 0.05, 0.15, 0.03, 40.0])
DT = 2.0
T_GRID = np.arange(0.0, 168.0 + DT, DT)


def naive_km_ate(T0):
    """Naive KM ATE on the observational cohort, mirroring a0_GenerateTrialData.m.

    Per-patient: death if any Y==1, else censored at that patient's last t.
    Grouped by the patient's baseline Rx.
    """
    times, events, arms = [], [], []
    for _, pdta in T0.groupby("sid", sort=True):
        died = int(pdta["Y"].max() == 1)
        if died:
            times.append(float(pdta.loc[pdta["Y"] == 1, "t"].min()))
        else:
            times.append(float(pdta["t"].max()))
        events.append(died)
        arms.append(int(pdta["Rx"].iloc[0]))
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    arms = np.asarray(arms, int)

    def km_at_end(tt, ee):
        order = np.argsort(tt, kind="mergesort")
        tt, ee = tt[order], ee[order]
        n, S = len(tt), 1.0
        at_risk, i = n, 0
        for te in np.unique(tt[ee == 1]):
            d = int(np.sum((tt == te) & (ee == 1)))
            while i < n and tt[i] < te:
                at_risk -= 1
                i += 1
            if at_risk > 0:
                S *= 1.0 - d / at_risk
            k = 0
            while i + k < n and tt[i + k] == te:
                k += 1
            at_risk -= k
            i += k
        return S

    s1 = km_at_end(times[arms == 1], events[arms == 1])
    s0 = km_at_end(times[arms == 0], events[arms == 0])
    return s1 - s0, s1, s0, int(np.sum(arms == 1)), int(np.sum(arms == 0))


def run_seed(seed, N):
    np.random.seed(seed)
    age, sofa, C, g, _ = fcnGeneratePatientParameters(
        N, "TargetCMean", 3, "TargetGMean", 4, "CV", 0.1
    )
    cohorts = {}
    for RCT in [0, 1]:
        L0 = fcnGenerateStochasticTrajectories(T_GRID, PARMS_L, N)
        if RCT == 1:
            treatProb = np.full(N, 0.5)
        else:
            # this helper prints a debug line per call; silence it
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                treatProb = fcnBiasedAssignmentProb(age, sofa, L0[:, :5])
        cohorts[RCT] = fcnSimulate_N_Patients(
            N, RCT, treatProb, TH, C, g, KE, L0,
            PARMS_CONTROL, PARMS_Y, PARMS_V, age, sofa,
        )

    s0_true, s1_true, _, _ = fcnPlotKM(cohorts[1])
    ate_true = float(s1_true[-1] - s0_true[-1])
    ate_naive, s1_n, s0_n, n_tx, n_ctl = naive_km_ate(cohorts[0])

    return dict(
        seed=seed, ate_true=ate_true, s1_true=float(s1_true[-1]),
        s0_true=float(s0_true[-1]), ate_naive=ate_naive, s1_naive=s1_n,
        s0_naive=s0_n, bias=ate_naive - ate_true, reversal=int(ate_naive < 0),
        n_treated_obs=n_tx, n_control_obs=n_ctl,
    )


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    rows = []
    for seed in range(n_seeds):
        rows.append(run_seed(seed, N))
        r = rows[-1]
        print(f"seed {seed:3d}  truth {100*r['ate_true']:+6.2f}%  "
              f"naive {100*r['ate_naive']:+6.2f}%  "
              f"bias {100*r['bias']:+6.2f}pp  reversal={r['reversal']}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiseed_py_results.csv")
    df.to_csv(out, index=False)

    def summarize(col):
        v = df[col].to_numpy() * 100
        return f"{v.mean():+7.2f}  {v.std(ddof=1):6.2f}  {v.std(ddof=1)/np.sqrt(len(v)):6.3f}  [{np.percentile(v,2.5):+6.2f}, {np.percentile(v,97.5):+6.2f}]"

    print("\n" + "=" * 78)
    print(f"PYTHON PIPELINE — {n_seeds} seeds, N={N} per arm-cohort")
    print("=" * 78)
    print(f"{'quantity':<16}{'mean':>8}{'SD':>8}{'MCSE':>8}   95% range")
    for c in ["ate_true", "ate_naive", "bias"]:
        print(f"{c:<16}{summarize(c)}")
    print(f"\nsign reversal (naive < 0) in {df.reversal.sum()}/{n_seeds} "
          f"= {100*df.reversal.mean():.0f}% of seeds")
    print(f"saved -> {out}")
