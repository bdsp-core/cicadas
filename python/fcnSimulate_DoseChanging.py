# fcnSimulate_DoseChanging.py
import numpy as np
import pandas as pd

def fcnSimulate_DoseChanging(N, th, C, g, ke, L0, parmsY, age, sofa, dt=2.0, seed=None):
    """
    Simulate dose-changing trajectories for N patients (Rx fixed ON) and return a long-format DataFrame.

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
        seed (int|None): RNG seed for reproducibility

    Returns:
        pandas.DataFrame with columns:
        ['age','sofa','sid','t','L','A','V','Y','Rx']
    """
    rng = np.random.default_rng(seed)

    C = np.asarray(C, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(-1)
    age = np.asarray(age, dtype=float).reshape(-1)
    sofa = np.asarray(sofa, dtype=float).reshape(-1)
    L0 = np.asarray(L0, dtype=float)

    if L0.ndim != 2 or L0.shape[0] != N:
        raise ValueError("L0 must be an array of shape (N, T).")
    T = L0.shape[1]
    t_vec = np.arange(0.0, (T - 1) * dt + dt, dt)  # 0:dt:168 (inclusive)

    frames = []
    for i in range(N):
        df_i = _run_simulation_dose_change(
            sid=i + 1,
            C_i=C[i],
            g_i=g[i],
            ke=ke,
            L0_i=L0[i, :],
            dt=dt,
            t_vec=t_vec,
            rng=rng,
            age_i=age[i],
            sofa_i=sofa[i],
        )
        frames.append(df_i)

    T_df = pd.concat(frames, ignore_index=True)
    return T_df


def _run_simulation_dose_change(sid, C_i, g_i, ke, L0_i, dt, t_vec, rng, age_i, sofa_i):
    """
    Per-patient simulator with random dose changes.
    Mirrors MATLAB fcnRunSimulationDoseChange() behavior.
    """
    T = len(t_vec)
    # State/outputs
    Rx = np.ones(T, dtype=int)  # Rx stays 1 (always treated)
    L = np.zeros(T, dtype=float)
    A = np.zeros(T, dtype=float)
    V = np.zeros(T, dtype=int)  # no censoring events here
    Y = np.zeros(T, dtype=int)  # no deaths here
    X = np.zeros(T, dtype=float)  # drug concentration

    # Initial conditions (match MATLAB)
    A[0] = Rx[0] * 2.0  # initial pump value
    L[0] = 0.0
    V[0] = 0
    Y[0] = 0
    X[0] = 0.0

    # Dose-changing vars
    current_dose = Rx[0] * 0.0  # start at 0 (even though A[0]=2 as in MATLAB)
    # time_since_last_change is tracked in MATLAB but not used downstream

    for j in range(1, T):
        # Treatment status persists
        Rx[j] = Rx[j - 1]

        # PD effect
        if Rx[j] == 1 and X[j - 1] > 0:
            sX = 1.0 - 1.0 / (((C_i / X[j - 1]) ** g_i) + 1.0)
        else:
            # Note: in MATLAB, with X=0 the fraction tends to 0, so sX≈1
            sX = 1.0

        # Disease with treatment scaling
        L[j] = L0_i[j] * sX

        # Random dose change with probability 1/5 each step (as in MATLAB code)
        if rng.random() < (1.0 / 5.0):
            current_dose = rng.random() * 5.0  # new random dose in [0,5]

        A[j] = current_dose

        # PK update
        X[j] = ke * X[j - 1] + A[j]

        # Events (kept zero here, matching provided MATLAB)
        V[j] = 0
        Y[j] = 0

    # Build DataFrame
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
