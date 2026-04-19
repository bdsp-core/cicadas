# fcnPlotKM.py
# KM curve — empirical with aligned time grid (0:2:168 → 85 points)

import numpy as np
import pandas as pd

def fcnPlotKM(T1: pd.DataFrame):
    """
    Python translation of fcnPlotKM.m

    Parameters
    ----------
    T1 : DataFrame
        Must contain columns: ['sid','Rx','t','Y']

    Returns
    -------
    s0, s1, t0, t1 : np.ndarray
        Each is shape (85,), aligned to t = 0,2,...,168.
        s0: untreated survival; s1: treated survival.
    """
    required = {"sid", "Rx", "t", "Y"}
    missing = required - set(T1.columns)
    if missing:
        raise ValueError(f"T1 is missing required columns: {sorted(missing)}")

    # --- Extract per-patient survival records (time to death or censoring, and baseline Rx)
    # MATLAB uses N = max(T1.sid) and loops 1..N; here we iterate unique IDs (safer, same outcome if sids are 1..N)
    surv_rows = []
    for pid in np.unique(T1["sid"].values):
        pdata = T1[T1["sid"] == pid].sort_values("t")
        initial_rx = int(pdata["Rx"].iloc[0])

        y = pdata["Y"].to_numpy()
        if (y == 1).any():
            # first death time
            idx = int(np.argmax(y == 1))  # first index where Y==1
            event_time = float(pdata["t"].iloc[idx])
            event_ind = 1
        else:
            # censored at last observed time
            event_time = float(pdata["t"].iloc[-1])
            event_ind = 0

        surv_rows.append((event_time, event_ind, initial_rx))

    surv_table = pd.DataFrame(surv_rows, columns=["time", "event", "treatment"])

    treated = surv_table[surv_table["treatment"] == 1]
    control = surv_table[surv_table["treatment"] == 0]

    # --- KM on raw times (without the initial [0,1] point — we'll add it after)
    t_treat_raw, s_treat_raw = _kaplan_meier(treated["time"].to_numpy(), treated["event"].to_numpy())
    t_ctrl_raw,  s_ctrl_raw  = _kaplan_meier(control["time"].to_numpy(),  control["event"].to_numpy())

    # Add initial point like MATLAB:
    s1_raw = np.concatenate([[1.0], s_treat_raw])
    t1_raw = np.concatenate([[0.0], t_treat_raw])
    s0_raw = np.concatenate([[1.0], s_ctrl_raw])
    t0_raw = np.concatenate([[0.0], t_ctrl_raw])

    # --- Interpolate to common grid using "previous/step" interpolation
    t_grid = np.arange(0.0, 168.0 + 2.0, 2.0)  # 85 points
    n_points = t_grid.size

    s0 = _step_interpolate(t0_raw, s0_raw, t_grid)
    s1 = _step_interpolate(t1_raw, s1_raw, t_grid)

    # Extend flat tail if last observed time < 168
    if np.max(t0_raw) < 168.0:
        last_idx = np.searchsorted(t_grid, np.max(t0_raw), side="right")
        s0[last_idx:] = s0_raw[-1]
    if np.max(t1_raw) < 168.0:
        last_idx = np.searchsorted(t_grid, np.max(t1_raw), side="right")
        s1[last_idx:] = s1_raw[-1]

    # Outputs as 1D arrays of length 85 (to mirror column vectors)
    assert s0.shape[0] == 85 and s1.shape[0] == 85 and n_points == 85
    t0 = t_grid.copy()
    t1 = t_grid.copy()
    return s0, s1, t0, t1


# ---------------------------------------------------------------------
# Helpers (equivalent to MATLAB fcn_kaplanMeier + step interpolation)
# ---------------------------------------------------------------------
def _kaplan_meier(times: np.ndarray, events: np.ndarray):
    """
    Kaplan–Meier survival at unique event times (no initial [0,1] point).
    Returns (t_event, S_at_event), both 1D arrays (possibly empty if no events).
    """
    times = np.asarray(times, dtype=float).copy()
    events = np.asarray(events, dtype=int).copy()
    if times.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Sort by time (stable) to match MATLAB behavior
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]

    event_times = np.unique(times[events == 1])
    if event_times.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n = times.size
    at_risk = n
    S = 1.0
    t_out = []
    s_out = []
    idx = 0  # pointer to first record at current event time

    for te in event_times:
        # advance idx to first record with time == te (decrease at_risk for all with time < te)
        while idx < n and times[idx] < te:
            at_risk -= 1
            idx += 1

        # counts at this event time
        d_i = int(np.sum((times == te) & (events == 1)))
        # KM multiplicative step
        if at_risk > 0:
            S *= (1.0 - d_i / at_risk)
        t_out.append(float(te))
        s_out.append(float(S))

        # remove *all* observations at te (events and censors) from risk set
        k = 0
        while idx + k < n and times[idx + k] == te:
            k += 1
        at_risk -= k
        idx += k

    return np.asarray(t_out, dtype=float), np.asarray(s_out, dtype=float)


def _step_interpolate(t_raw: np.ndarray, s_raw: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Step-function ("previous") interpolation:
    For each grid time, take the last s_raw at t_raw <= grid time; else 1.0.
    """
    t_raw = np.asarray(t_raw, dtype=float)
    s_raw = np.asarray(s_raw, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    out = np.empty_like(t_grid, dtype=float)
    # For each grid time, find rightmost index in t_raw <= grid time
    idxs = np.searchsorted(t_raw, t_grid, side="right") - 1
    # If idx < 0 → no events yet → survival = 1
    out[idxs < 0] = 1.0
    valid = idxs >= 0
    out[valid] = s_raw[idxs[valid]]
    return out
