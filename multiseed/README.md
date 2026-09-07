# Multi-seed replication of the headline confounding result

Drivers for the 400-replication experiment reported in the CICADAS manuscript
(Section "Simulated cohorts and the confounding problem", Table 1).

## Why this exists

A single simulated cohort gives one draw from the data-generating process. Reporting
the ground-truth and naive treatment effects from one draw understates how much they
vary. These drivers replicate the whole generate-and-analyse procedure across many
seeds so the quantities can be reported as sampling distributions with Monte Carlo
standard errors.

## Contents

| File | |
|---|---|
| `multiseed_matlab.m`          | MATLAB driver, seeds 0..N-1 |
| `multiseed_py.py`             | Python driver, same design |
| `multiseed_matlab_results.csv`| per-seed results, MATLAB (200 seeds) |
| `multiseed_py_results.csv`    | per-seed results, Python (200 seeds) |

Each driver reproduces the RNG consumption order of the corresponding
`a0_GenerateTrialData` pipeline exactly. Seed 0 reproduces that pipeline's
single-draw output in each language, which is what validates the drivers.

## Usage

```bash
matlab -batch "n_seeds=200; N=2000; multiseed_matlab"
python multiseed_py.py 200 2000
```

Each replication regenerates patient parameters, natural-history trajectories,
treatment assignment, and outcomes; then computes the ground-truth ATE from the
randomized cohort and the naive Kaplan-Meier ATE from the observational cohort.
Runtime is roughly 1-2 s per seed.

## Results (400 replications, N = 2000 per cohort, 168-hour endpoint)

| Quantity | Mean | SD | MCSE | 95% range |
|---|---|---|---|---|
| Ground-truth ATE | +14.57 | 2.17 | 0.109 | [+10.81, +18.70] |
| Naive Kaplan-Meier ATE | +1.00 | 3.12 | 0.156 | [-5.33, +6.90] |
| Bias of the naive estimator | -13.57 | 3.79 | 0.189 | [-21.24, -6.52] |

All values are percentage points. Confounding by indication eliminates 93% of the
true treatment effect on average, and reverses its sign in 156/400 (39%) of
replications.

## Note on the two implementations

MATLAB and Python use different random number generators (Mersenne Twister and
PCG64), so the same nominal seed produces different individual cohorts. They are
therefore independent replications, not bit-for-bit reproductions. Across 200 seeds
each, they agree on every quantity to within Monte Carlo error (Welch p >= 0.06;
two-sample Kolmogorov-Smirnov p >= 0.11).

---

## Calibration of the g-formula confidence intervals

`coverage_py.py` and `coverage_analysis.py` establish the sampling distribution of the g-formula
estimator across independently simulated cohorts, and compare it against the bootstrap standard error
reported by `a2_CausalSurvivalAnalysis.m`.

```bash
python coverage_py.py 100 2000     # ~35 min; writes coverage_results.csv
python coverage_analysis.py        # writes the comparison
```

Each replication re-estimates the mortality and natural-history models from a fresh observational
cohort and emulates a randomized trial by forward Monte Carlo. The PK/PD fit is held at its point
estimate, mirroring both the analysis pipeline (`a1` runs once on the dose-switching cohort, `a2`
consumes its output) and `a2`'s bootstrap, which holds it fixed as well. The resulting spread is
therefore a lower bound on total sampling variability.

### Results (100 replications, N = 2000)

| Quantity | Value |
|---|---|
| g-formula ATE | +15.02 pp (SD 2.32, MCSE 0.23) |
| ground-truth ATE | +14.30 pp (SD 2.25) |
| bias | +0.72 pp (95% CI +0.10 to +1.34) |
| empirical sampling SD | 2.32 pp |
| bootstrap SE (B = 1000, one cohort) | 2.16 pp |
| ratio bootstrap/empirical | 0.93 |
| coverage of a nominal 95% interval | 92% (95% CI 87–97%) |

The bootstrap standard error is close to the empirical sampling standard deviation, so the intervals
are approximately calibrated at the nominal level. The estimator carries a small upward bias of about
5% of the effect being estimated.
