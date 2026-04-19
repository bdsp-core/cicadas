# fcnSimulate_StructuredDoses.py
import math
import numpy as np
import pandas as pd

def fcnSimulate_StructuredDoses(N, th, C, g, ke, L0, parmsY, age, sofa, dt=2.0, seed=None):
    """
    Generate data with STRUCTURED dose changes that improve identifiability.

    Args:
        N (int): number of patients
        th (float): (unused; kept for signature compatibility)
        C (array-like): length-N PD parameter C per patient
        g (array-like): length-N PD parameter g per patient
        ke (float): elimination constant
        L0 (ndarray): shape (N, T) baseline disease trajectories
        parmsY (array-like): (unused; kept for signature compatibility)
        age (array-like): length-N ages
        sofa (array-like): length-N SOFA scores
        dt (float): time step (hours). Default 2.0
        seed (int|None): RNG seed (not used here but kept for symmetry with other simulators)

    Returns:
        pandas.DataFrame with columns:
        ['age','sofa','sid','t','L','A','V','Y','Rx']
    """
    C = np.asarray(C, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(-1)
    age = np.asarray(age, dtype=float).reshape(-1)
    sofa = np.asarray(sofa, dtype=float).reshape(-1)
    L0 = np.asarray(L0, dtype=float)

    if L0.ndim != 2 or L0.shape[0] != N:
        raise ValueError("L0 must be an array of shape (N, T).")

    T = L0.shape[1]
    t_vec = np.arange(0.0, T * dt, dt)  # 0:dt:168 (assumes T*dt≈168+dt)

    frames = []
    for i in range(N):
        protocol = (i % 4) + 1  # 1..4 cycling
        df_i = _run_simulation_structured(
            sid=i + 1,
            C_i=C[i],
            g_i=g[i],
            ke=ke,
            L0_i=L0[i, :],
            dt=dt,
            t_vec=t_vec,
            protocol=protocol,
            age_i=age[i],
            sofa_i=sofa[i],
        )
        frames.append(df_i)

    T_df = pd.concat(frames, ignore_index=True)
    return T_df


def _run_simulation_structured(sid, C_i, g_i, ke, L0_i, dt, t_vec, protocol, age_i, sofa_i):
    """
    Per-patient simulator with structured dose protocols.
    Mirrors MATLAB fcnRunSimulation_Structured() behavior.
    """
    T = len(t_vec)

    # Initialize
    Rx = np.ones(T, dtype=int)  # all treated
    L = np.zeros(T, dtype=float)
    A = np.zeros(T, dtype=float)
    V = np.zeros(T, dtype=int)  # no censoring in this version
    Y = np.zeros(T, dtype=int)  # no mortality in this version
    X = np.zeros(T, dtype=float)

    L[0] = L0_i[0]

    # Define dose schedules
    if protocol == 1:       # Escalation
        dose_schedule = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    elif protocol == 2:     # De-escalation
        dose_schedule = [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    elif protocol == 3:     # On-off
        dose_schedule = [3, 3, 0, 0, 3, 3, 0, 0, 3, 3]
    else:                   # protocol == 4: Step changes
        dose_schedule = [2, 2, 4, 4, 2, 2, 4, 4, 2, 2]

    schedule_len = len(dose_schedule)

    # MATLAB used: dose_idx = @(j) dose_schedule(min(ceil(j/8.5), schedule_length));
    # Here j is 1-based in MATLAB loop (j=2..Nt). We loop j=1..T-1 in Python.
    def dose_for_step(j_one_based: int) -> float:
        # Change bin roughly every ~17 hours -> ~8.5 steps (dt=2h)
        idx_1based = min(int(math.ceil(j_one_based / 8.5)), schedule_len)
        return dose_schedule[idx_1based - 1]

    # Simulation loop
    for j in range(1, T):
        # Structured dose assignment (use 1-based j for binning)
        A[j] = dose_for_step(j)

        # PK update
        X[j] = ke * X[j - 1] + A[j]

        # PD effect
        if X[j - 1] > 0:
            sX = 1.0 - 1.0 / (((C_i / X[j - 1]) ** g_i) + 1.0)
        else:
            sX = 1.0

        # Disease dynamics (scaled baseline)
        L[j] = L0_i[j] * sX

        # Hazards remain zero in this structured-dose simulator
        V[j] = 0
        Y[j] = 0

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "age": age_i * np.ones(T),
            "sofa": sofa_i * np.ones(T),
            "sid": sid * np.ones(T, dtype=int),
            "t": t_vec,
            "L": L,
            "A": A,
            "V": V,
            "Y": Y,
            "Rx": Rx,
        }
    )
    return df
