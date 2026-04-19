# fcnEstimateDeathParms.py

import numpy as np
import pandas as pd

# Prefer statsmodels (closest to MATLAB glmfit). Fall back to scikit-learn if needed.
try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False
    from sklearn.linear_model import LogisticRegression

def fcnEstimateDeathParms(T0: pd.DataFrame) -> np.ndarray:
    """
    Estimate mortality hazard parameters via logistic regression.

    MATLAB model:
      logit_y = a0
                + a1 * (t/170)^2
                + a2 * (sofa) * (cumsum_L/24)^2
                + a3 * (age/90) * (cumsum_A/207)

    Inputs
    ------
    T0 : DataFrame with columns ['sid','t','sofa','age','L','A','Y']

    Returns
    -------
    parmsY_est : np.ndarray shape (4,)
        [a0, a1, a2, a3]
    """
    required = {"sid", "t", "sofa", "age", "L", "A", "Y"}
    missing = required - set(T0.columns)
    if missing:
        raise ValueError(f"T0 is missing required columns: {sorted(missing)}")

    # Build design matrix exactly like MATLAB loop (no per-patient time sorting here)
    x1_list, x2_list, x3_list, y_list = [], [], [], []

    unique_sids = np.unique(T0["sid"].values)
    for sid in unique_sids:
        patient_data = T0[T0["sid"] == sid]

        tt = patient_data["t"].to_numpy(dtype=float)
        sofa = patient_data["sofa"].to_numpy(dtype=float)
        age = patient_data["age"].to_numpy(dtype=float)
        csL = np.cumsum(patient_data["L"].to_numpy(dtype=float))
        csA = np.cumsum(patient_data["A"].to_numpy(dtype=float))
        YY = patient_data["Y"].to_numpy(dtype=int)

        x1_list.append((tt / 170.0) ** 2)
        x2_list.append(sofa * (csL / 24.0) ** 2)
        x3_list.append((age / 90.0) * (csA / 207.0))
        y_list.append(YY)

    x1 = np.concatenate(x1_list, axis=0)
    x2 = np.concatenate(x2_list, axis=0)
    x3 = np.concatenate(x3_list, axis=0)
    Y = np.concatenate(y_list, axis=0).astype(int)

    X = np.column_stack([x1, x2, x3]).astype(float)

    # Drop any rows with NaNs (defensive; MATLAB would error if NaNs present)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y)
    X = X[valid]
    Y = Y[valid]

    if _HAS_SM:
        # statsmodels GLM (Binomial with logit link) — closest to MATLAB glmfit
        Xc = sm.add_constant(X, has_constant="add")  # add intercept term
        model = sm.GLM(Y, Xc, family=sm.families.Binomial(link=sm.families.links.logit()))
        result = model.fit()
        b = result.params  # [a0, a1, a2, a3]
        return np.asarray(b, dtype=float)
    else:
        # Fallback: scikit-learn logistic regression (approximate glmfit)
        # Try no regularization; if not supported, use a very weak L2.
        try:
            lr = LogisticRegression(
                penalty="none",
                solver="lbfgs",
                fit_intercept=True,
                max_iter=1000,
            )
        except ValueError:
            lr = LogisticRegression(
                penalty="l2",
                C=1e6,              # ~unregularized
                solver="lbfgs",
                fit_intercept=True,
                max_iter=1000,
            )
        lr.fit(X, Y)
        a0 = float(lr.intercept_[0])
        a1, a2, a3 = lr.coef_[0].tolist()
        return np.array([a0, a1, a2, a3], dtype=float)
