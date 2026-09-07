"""
Calibration analysis for the CICADAS g-formula confidence intervals.

Compares the empirical sampling distribution of the estimator (coverage_results.csv,
produced by coverage_py.py across independent cohorts) against the bootstrap standard
error reported by a2_CausalSurvivalAnalysis.m (bootstrap_confidence_bands.mat).

If the bootstrap SE matches the empirical SD, the intervals are calibrated. If the
bootstrap SE is smaller, the intervals are too narrow and their true coverage falls
below the nominal level.
"""

import os, numpy as np, pandas as pd, scipy.io as sio
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "coverage_results.csv"))
b = sio.loadmat(os.path.join(HERE, "..", "bootstrap_confidence_bands.mat"), squeeze_me=True)
boot = (b["S1h"][:, -1] - b["S0h"][:, -1]) * 100

gf = df.ate_gformula * 100
tr = df.ate_true * 100
R = len(df)
emp_sd = gf.std(ddof=1)
boot_se = boot.std(ddof=1)
pop = tr.mean()

print("=" * 78)
print("CICADAS — calibration of the g-formula confidence intervals")
print("=" * 78)
print(f"""
DESIGN
  {R} independent simulated cohorts (N = 2000 each). In every replication the
  mortality and natural-history models are re-estimated from a fresh observational
  cohort and a randomized trial is emulated by forward Monte Carlo. The PK/PD fit
  is held at its point estimate, mirroring the paper's pipeline (a1 is run once on
  a dedicated dose-switching cohort; a2 consumes its output) and mirroring a2's
  bootstrap, which likewise holds it fixed. The spread below is therefore a LOWER
  BOUND on total sampling variability.
""")
print("-" * 78)
print("1. SAMPLING DISTRIBUTION OF THE ESTIMATOR")
print("-" * 78)
print(f"  g-formula ATE      mean {gf.mean():+7.3f} pp   SD {emp_sd:6.3f}   MCSE {emp_sd/np.sqrt(R):.3f}")
print(f"  ground-truth ATE   mean {tr.mean():+7.3f} pp   SD {tr.std(ddof=1):6.3f}")
bias = gf.mean() - pop
t, p = stats.ttest_1samp(gf - tr, 0.0)
print(f"  bias vs population estimand: {bias:+.3f} pp   (paired t = {t:.2f}, p = {p:.3f})")

print("\n" + "-" * 78)
print("2. IS THE BOOTSTRAP SE CALIBRATED?")
print("-" * 78)
print(f"  empirical sampling SD (across {R} cohorts) : {emp_sd:6.3f} pp")
print(f"  bootstrap SE (B = {len(boot)}, single cohort)  : {boot_se:6.3f} pp")
print(f"  ratio bootstrap/empirical                  : {boot_se/emp_sd:6.3f}")

z = 1.959963985
nominal_half = z * boot_se
implied = 2 * stats.norm.cdf(nominal_half / emp_sd) - 1
print(f"""
  A nominal 95% interval built from the bootstrap SE has half-width
  {nominal_half:.2f} pp. Against a true sampling SD of {emp_sd:.2f} pp, its actual
  coverage is 2*Phi({nominal_half:.2f}/{emp_sd:.2f}) - 1 = {100*implied:.0f}%.""")

# direct empirical check: fraction of Wald intervals containing the population estimand
lo = gf - nominal_half
hi = gf + nominal_half
cov = ((lo <= pop) & (pop <= hi)).mean()
se_cov = np.sqrt(cov * (1 - cov) / R)
print(f"""
  Direct check: building a 95% Wald interval around each replication's estimate
  using the bootstrap SE, {100*cov:.0f}% of the {R} intervals contain the population
  estimand ({pop:+.2f} pp), Monte Carlo SE {100*se_cov:.1f} points.""")

print("\n" + "-" * 78)
print("3. WHY")
print("-" * 78)
print("""  a2's bootstrap resamples patients and re-estimates only the mortality model.
  The natural-history and PK/PD fits are held at their point estimates, so two of
  the three sources of estimation uncertainty are not propagated into the interval.
  That is the likely explanation for the intervals being narrower than the true
  sampling variability of the estimator.""")

out = os.path.join(HERE, "coverage_SUMMARY.txt")
print(f"\n(re-run this script to regenerate; results file: coverage_results.csv)")
