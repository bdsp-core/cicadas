# Benchmarks: IPTW and MSM

Head-to-head comparison of the CICADAS g-formula estimator against inverse probability of treatment weighting (IPTW) and marginal structural models (MSM) on the simulated observational data. Added during the iScience revision to address the editor's request (E10) for benchmarking against alternative causal-inference methods.

TMLE is deferred as future work because of the need for an R dependency (`ltmle` / `survtmle`).

Planned contents:

- `iptw.py` — Stabilized IPTW with pooled logistic regression treatment and censoring models, truncation diagnostics.
- `msm.py` — Marginal structural model via weighted pooled logistic regression.
- `compare.py` — Runs IPTW, MSM, and g-formula on the same simulated cohorts; reports ATE estimates, bootstrap CIs, runtime, and weight-stability diagnostics.
- `figures/` — Forest plot comparing methods.

Implementation language: Python (ergonomic libraries: `zEpid`, `lifelines`, `pygformula`).

See `../PLAN_response_to_reviews.md` §5 for scope.
