import numpy as np
from typing import Tuple, Dict, Any, Iterable


def fcnGeneratePatientParameters(N: int, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct translation of MATLAB:
        [age, sofa, C, g, parmsPD] = fcnGeneratePatientParameters(N, varargin)

    Name–value parameters (same names as MATLAB):
        'TargetCMean' : float (default 3)
        'TargetGMean' : float (default 4)
        'CV'          : float in (0,1) (default 0.1)
        'AgeRange'    : [low, high] (default [18, 90])
        'SofaRange'   : [low, high] (default [0, 24])
        'RandomSeed'  : int or None (default None)
        'MinValue'    : float >= 0 (default 0.1)

    Returns:
        age     : (N,) array
        sofa    : (N,) array
        C       : (N,) array
        g       : (N,) array
        parmsPD : (6,) array -> [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g]
    """

    # ----------------------------
    # Input parsing / validation
    # ----------------------------
    if not (isinstance(N, (int, np.integer)) and N > 0):
        raise ValueError("N must be a positive integer scalar.")

    # Defaults mirroring MATLAB addParameter defaults
    opts: Dict[str, Any] = {
        "TargetCMean": 3.0,
        "TargetGMean": 4.0,
        "CV": 0.1,
        "AgeRange": np.array([18.0, 90.0], dtype=float),
        "SofaRange": np.array([0.0, 24.0], dtype=float),
        "RandomSeed": None,
        "MinValue": 0.1,
    }

    # Support MATLAB-style name–value pairs via *args
    if len(args) % 2 != 0:
        raise ValueError("Name–value pairs must come in key,value pairs (even number of *args).")
    for k, v in zip(args[::2], args[1::2]):
        if not isinstance(k, str):
            raise ValueError("Name–value keys must be strings.")
        opts[k] = v

    # Also allow normal Python kwargs (kwargs override *args if both provided)
    opts.update(kwargs)

    # Extract and validate
    target_C_mean = float(opts["TargetCMean"])
    target_g_mean = float(opts["TargetGMean"])
    cv = float(opts["CV"])
    age_range = np.asarray(opts["AgeRange"], dtype=float)
    sofa_range = np.asarray(opts["SofaRange"], dtype=float)
    random_seed = opts["RandomSeed"]
    min_val = float(opts["MinValue"])

    if not (target_C_mean > 0):
        raise ValueError("TargetCMean must be > 0.")
    if not (target_g_mean > 0):
        raise ValueError("TargetGMean must be > 0.")
    if not (0 < cv < 1):
        raise ValueError("CV must be in (0,1).")
    if not (age_range.size == 2 and age_range[0] < age_range[1]):
        raise ValueError("AgeRange must be length-2 with low < high.")
    if not (sofa_range.size == 2 and sofa_range[0] <= sofa_range[1]):
        raise ValueError("SofaRange must be length-2 with low <= high.")
    if not (min_val >= 0):
        raise ValueError("MinValue must be >= 0.")
    if random_seed is not None and not isinstance(random_seed, (int, np.integer)):
        raise ValueError("RandomSeed must be an integer or None.")

    # Set random seed if provided (mirrors MATLAB rng(RandomSeed))
    if random_seed is not None:
        np.random.seed(int(random_seed))

    # ---------------------------------
    # Generate age and SOFA (uniform)
    # ---------------------------------
    age = age_range[0] + (age_range[1] - age_range[0]) * np.random.rand(N)
    sofa = sofa_range[0] + (sofa_range[1] - sofa_range[0]) * np.random.rand(N)

    # ---------------------------------
    # Coefficients and noise scales
    # ---------------------------------
    sigma_C = cv * target_C_mean
    sigma_g = cv * target_g_mean

    # Normalize predictors (using same constants)
    age_norm = (age - age.mean()) / 20.4
    sofa_norm = (sofa - sofa.mean()) / 6.8468

    # Regression coefficients (intercepts to target means)
    b0_C = target_C_mean
    b1_C = 0.1   # age coefficient for C
    b2_C = 0.15  # sofa coefficient for C

    b0_g = target_g_mean
    b1_g = 0.08  # age coefficient for g
    b2_g = 0.12  # sofa coefficient for g

    # ---------------------------------
    # Generate C and g
    # ---------------------------------
    noise_C = sigma_C * np.random.randn(N)
    C_linear = b0_C + b1_C * age_norm + b2_C * sofa_norm
    C = C_linear + noise_C

    noise_g = sigma_g * np.random.randn(N)
    g_linear = b0_g + b1_g * age_norm + b2_g * sofa_norm
    g = g_linear + noise_g

    # Ensure positivity
    C = np.where(C < min_val, min_val, C)
    g = np.where(g < min_val, min_val, g)

    parmsPD = np.array([b0_C, b1_C, b2_C, b0_g, b1_g, b2_g], dtype=float)

    return age, sofa, C, g, parmsPD
