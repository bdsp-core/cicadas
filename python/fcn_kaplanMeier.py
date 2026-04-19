# fcn_kaplanMeier.py
import numpy as np
from typing import Tuple

def fcn_kaplanMeier(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Kaplan–Meier survival estimates (with Greenwood SE), matching the MATLAB logic.

    Parameters
    ----------
    event_times : array-like
        Times of event or censoring.
    event_indicators : array-like
        1 if event (death), 0 if censored.

    Returns
    -------
    t_km : (m,) ndarray
        Unique event times (where events occurred).
    s_km : (m,) ndarray
        Survival probability at each event time.
    se_km : (m,) ndarray
        Standard errors via Greenwood's formula.
    n_risk : (m,) ndarray
        Number at risk just prior to each event time.
    """
    # -- sanitize & sort --
    times = np.asarray(event_times, dtype=float).ravel()
    events = np.asarray(event_indicators, dtype=float).ravel()  # allow NaN filter; cast to int later

    valid = ~np.isnan(times) & ~np.isnan(events)
    times = times[valid]
    events = events[valid].astype(int)

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]

    # unique times where an event occurred
    unique_times = np.unique(times[events == 1])
    n_times = unique_times.size

    t_km = unique_times.copy()
    s_km = np.ones(n_times, dtype=float)
    se_km = np.zeros(n_times, dtype=float)
    n_risk = np.zeros(n_times, dtype=float)

    survival_prob = 1.0
    variance_sum = 0.0  # Greenwood accumulator

    for i, t in enumerate(unique_times):
        n_at_risk = np.sum(times >= t)
        n_risk[i] = n_at_risk

        n_events = np.sum((times == t) & (events == 1))

        if n_at_risk > 0:
            survival_prob *= (1.0 - n_events / n_at_risk)
            s_km[i] = survival_prob

            if n_events > 0:
                denom = n_at_risk * (n_at_risk - n_events)
                if denom > 0:
                    variance_sum += n_events / denom
                else:
                    variance_sum = np.inf  # all-at-risk fail at once → infinite variance

            if survival_prob > 0 and np.isfinite(variance_sum):
                se_km[i] = survival_prob * np.sqrt(variance_sum)
            elif survival_prob <= 0:
                se_km[i] = 0.0
            else:
                se_km[i] = np.inf

    return t_km, s_km, se_km, n_risk
