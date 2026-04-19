# Sensitivity analyses

Experiments added during the iScience revision to address reviewer concerns about unmeasured confounding, measurement error, and alternative censoring mechanisms.

Planned contents:

- `nuc_injection/` — Unobserved binary confounder injected into the DGP and hidden from the g-formula; bias reported as a function of confounder strength.
- `measurement_error/` — Gaussian noise added to `L_t` observations at increasing variance; estimation re-run.
- `alt_censoring/` — Alternative censoring mechanisms (completely-at-random, strongly informative outcome-dependent) replacing the default; estimation re-run.
- `e_values/` — E-value and tipping-point analysis for the recovered ATE.

Implementation language: Python (for easier integration with `zEpid`, `lifelines`, and `pygformula`). MATLAB wrappers for the existing DGP will be invoked via `matlab.engine` or re-implemented.

See `../PLAN_response_to_reviews.md` §5 for scope.
