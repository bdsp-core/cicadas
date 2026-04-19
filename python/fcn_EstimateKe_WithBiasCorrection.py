import numpy as np
from typing import Tuple, Dict, Any

from fcnEstimateKe_Standalone import fcnEstimateKe_Standalone


def fcnEstimateKe_WithBiasCorrection(
    L_obs: np.ndarray,
    A_obs: np.ndarray,
    *args,
    **kwargs,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Standalone ke Estimation with Empirical Bias Correction

    Direct translation of MATLAB:
        [ke_est, ke_raw, results] = fcnEstimateKe_WithBiasCorrection(L_obs, A_obs, varargin)

    Name–value params (same names/defaults as MATLAB):
        'CorrectionFactor' : float (default 1.41)
        'UsePrior'         : bool  (default False)
        'PriorKe'          : float (default 0.5)
        'PriorWeight'      : float (default 0.3)
        'KeRange'          : [low, high] (default [0.3, 0.9])
        'AssumeC'          : float (default 3.0)
        'AssumeG'          : float (default 4.0)
        'Verbose'          : bool  (default True)
    """

    # --------------------------
    # Parse name–value args
    # --------------------------
    opts: Dict[str, Any] = {
        "CorrectionFactor": 1.41,
        "UsePrior": False,
        "PriorKe": 0.5,
        "PriorWeight": 0.3,
        "KeRange": np.array([0.3, 0.9], dtype=float),
        "AssumeC": 3.0,
        "AssumeG": 4.0,
        "Verbose": True,
    }

    if len(args) % 2 != 0:
        raise ValueError("Name–value pairs must be key,value,... (even number of *args).")
    for k, v in zip(args[::2], args[1::2]):
        if not isinstance(k, str):
            raise ValueError("Name–value keys must be strings.")
        opts[k] = v
    opts.update(kwargs)

    correction_factor = float(opts["CorrectionFactor"])
    use_prior = bool(opts["UsePrior"])
    prior_ke = float(opts["PriorKe"])
    prior_weight = float(opts["PriorWeight"])
    ke_range = np.asarray(opts["KeRange"], dtype=float).reshape(-1)
    assumeC = float(opts["AssumeC"])
    assumeG = float(opts["AssumeG"])
    verbose = bool(opts["Verbose"])

    if ke_range.size != 2:
        raise ValueError("KeRange must be length-2.")

    if verbose:
        print("\n========================================")
        print("ke ESTIMATION WITH BIAS CORRECTION")
        print("========================================")
        print(f"Correction factor: {correction_factor:.2f}")
        if use_prior:
            print(f"Using prior: ke = {prior_ke:.3f} (weight = {prior_weight:.2f})")

    # --------------------------
    # Step 1: Get raw ke estimate
    # --------------------------
    ke_raw, results_raw = fcnEstimateKe_Standalone(
        L_obs,
        A_obs,
        "AssumeC",
        assumeC,
        "AssumeG",
        assumeG,
        "KeRange",
        ke_range,
        "WindowSize",
        10,
        "MinDoseChange",
        1.0,
        "EstimationMethod",
        "robust",
        "Verbose",
        False,
    )

    if verbose:
        print(f"\nRaw estimate: {ke_raw:.3f}")

    # --------------------------
    # Step 2: Apply bias correction
    # --------------------------
    ke_corrected = ke_raw * correction_factor
    if verbose:
        print(f"After correction: {ke_corrected:.3f}")

    # --------------------------
    # Step 3: Optional prior incorporation
    # --------------------------
    if use_prior:
        ke_with_prior = (1.0 - prior_weight) * ke_corrected + prior_weight * prior_ke
        if verbose:
            print(f"After prior incorporation: {ke_with_prior:.3f}")
        ke_est = ke_with_prior
    else:
        ke_est = ke_corrected

    # --------------------------
    # Step 4: Ensure within physiological bounds
    # --------------------------
    ke_low, ke_high = float(ke_range[0]), float(ke_range[1])
    ke_est = max(ke_low, min(ke_high, ke_est))
    if verbose and (np.isclose(ke_est, ke_low) or np.isclose(ke_est, ke_high)):
        print("⚠ Estimate bounded to physiological range")

    # --------------------------
    # Compile results
    # --------------------------
    results: Dict[str, Any] = dict(results_raw)  # shallow copy
    results["ke_raw"] = float(ke_raw)
    results["ke_corrected"] = float(ke_corrected)
    results["ke_final"] = float(ke_est)
    results["correction_factor"] = float(correction_factor)
    results["used_prior"] = bool(use_prior)
    if use_prior:
        results["prior_ke"] = float(prior_ke)
        results["prior_weight"] = float(prior_weight)

    # Confidence intervals with correction (if available)
    if "ke_ci95" in results_raw and np.size(results_raw["ke_ci95"]) == 2:
        ci_raw = np.asarray(results_raw["ke_ci95"], dtype=float).reshape(-1)
        ci_corr = ci_raw * correction_factor
        # Clip to range
        ci_final = np.array(
            [max(ke_low, ci_corr[0]), min(ke_high, ci_corr[1])], dtype=float
        )
        results["ci95_raw"] = ci_raw
        results["ci95_corrected"] = ci_corr
        results["ci95_final"] = ci_final

    if verbose:
        print("\n========================================")
        print("FINAL RESULT:")
        print("========================================")
        print(f"ke estimate: {ke_est:.3f}")
        if "ci95_final" in results:
            print(f"95% CI: [{results['ci95_final'][0]:.3f}, {results['ci95_final'][1]:.3f}]")
        print("========================================")

    return float(ke_est), float(ke_raw), results
