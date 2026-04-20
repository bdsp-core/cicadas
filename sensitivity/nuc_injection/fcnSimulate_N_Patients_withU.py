"""fcnSimulate_N_Patients_withU.py

Python port of sensitivity/nuc_injection/fcnSimulate_N_Patients_withU.m.

Simulates N patients with an optional per-patient additive shift to the
mortality hazard logit. Identical to python/fcnSimulate_N_Patients.py
except for the extra u_effect_on_Y term. U is never written to the output
table (it is an unobserved confounder by construction; downstream
g-formula estimators cannot adjust for it).
"""

from typing import Tuple

import numpy as np
import pandas as pd


def fcnSimulate_N_Patients_withU(
    N: int,
    RCT: int,
    treatProb: np.ndarray,
    th: float,
    C: np.ndarray,
    g: np.ndarray,
    ke: float,
    L0: np.ndarray,
    parmsControl: np.ndarray,
    parmsY: np.ndarray,
    parmsV: np.ndarray,
    age: np.ndarray,
    sofa: np.ndarray,
    u_effect_on_Y: np.ndarray = None,
) -> pd.DataFrame:
    """Direct port of MATLAB fcnSimulate_N_Patients_withU.

    Parameters
    ----------
    u_effect_on_Y : (N,) array, optional
        Per-patient additive shift to the mortality logit. In the NUC
        experiment this is U(i) * delta_Y where U ~ Bernoulli(0.5). Pass
        None (or zeros) to recover the baseline DGP.
    """
    if u_effect_on_Y is None:
        u_effect_on_Y = np.zeros(N, dtype=float)
    u_effect_on_Y = np.asarray(u_effect_on_Y, dtype=float).reshape(-1)

    tt, Lt, At, Vt, Yt, Rxt = [], [], [], [], [], []
    sid, Age, Sofa = [], [], []

    for i in range(N):
        p_i = treatProb[i] if np.ndim(treatProb) else float(treatProb)
        Rx0 = 1 if np.random.rand() < p_i else 0

        t_i, L_i, A_i, V_i, Y_i, Rx_i = _fcnRunSimulation(
            RCT, Rx0, th, float(C[i]), float(g[i]), float(age[i]), float(sofa[i]),
            float(ke), np.asarray(L0[i, :], dtype=float),
            parmsControl, parmsY, parmsV, float(u_effect_on_Y[i]),
        )

        sid.extend([i + 1] * len(t_i))
        tt.extend(t_i.tolist())
        Lt.extend(L_i.tolist())
        At.extend(A_i.tolist())
        Vt.extend(V_i.tolist())
        Yt.extend(Y_i.tolist())
        Rxt.extend(Rx_i.tolist())
        Age.extend([float(age[i])] * len(t_i))
        Sofa.extend([float(sofa[i])] * len(t_i))

    T = pd.DataFrame(
        {
            "age": np.asarray(Age, dtype=float),
            "sofa": np.asarray(Sofa, dtype=float),
            "sid": np.asarray(sid, dtype=int),
            "t": np.asarray(tt, dtype=float),
            "L": np.asarray(Lt, dtype=float),
            "A": np.asarray(At, dtype=float),
            "V": np.asarray(Vt, dtype=int),
            "Y": np.asarray(Yt, dtype=int),
            "Rx": np.asarray(Rxt, dtype=int),
        }
    )
    return T


def _fcnRunSimulation(
    RCT: int,
    Rx0: int,
    th: float,
    C: float,
    g: float,
    age: float,
    sofa: float,
    ke: float,
    L0: np.ndarray,
    parmsControl: np.ndarray,
    parmsY: np.ndarray,
    parmsV: np.ndarray,
    u_shift_y: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ki, Amax = float(parmsControl[0]), float(parmsControl[1])
    a0, a1, a2, a3 = (float(parmsY[0]), float(parmsY[1]),
                      float(parmsY[2]), float(parmsY[3]))
    # a4 kept for parity; unused
    _ = float(parmsY[4]) if len(parmsY) >= 5 else 0.0
    b0, b1, b2, b3, b4, b5 = (float(parmsV[0]), float(parmsV[1]), float(parmsV[2]),
                              float(parmsV[3]), float(parmsV[4]), float(parmsV[5]))

    dt = 2.0
    t = np.arange(0.0, 168.0 + dt, dt)
    Nt = t.size

    def expit(x):
        return 1.0 / (1.0 + np.exp(-x))

    eInt = 0.0

    A = np.zeros(Nt, dtype=float)
    L = np.zeros(Nt, dtype=float)
    V = np.zeros(Nt, dtype=int)
    Y = np.zeros(Nt, dtype=int)
    Rx = np.zeros(Nt, dtype=int)
    X = np.zeros(Nt, dtype=float)

    Rx[0] = int(Rx0)
    A[0] = Rx0 * 2.0
    L[0] = 0.0
    V[0] = 0
    Y[0] = 0
    X[0] = 0.0

    cumsum_L = 0.0
    cumsum_A = 0.0

    last_index = 0
    for j in range(1, Nt):
        last_index = j
        Rx[j] = Rx[j - 1]

        if Rx[j] == 1:
            if X[j - 1] == 0.0:
                sX = 1.0
            else:
                sX = 1.0 - 1.0 / (((C / X[j - 1]) ** g) + 1.0)
        else:
            sX = 1.0

        L[j] = L0[j] * sX

        if (Rx[j] == 1) and (j > 5):
            e = L[j] - th
            eInt = eInt + e * dt
            Aunsat = ki * eInt
            A[j] = np.clip(Aunsat, 0.0, Amax)
            if A[j] != Aunsat:
                eInt = eInt - (Aunsat - A[j]) / ki
        else:
            A[j] = 0.0

        X[j] = ke * X[j - 1] + A[j]
        cumsum_L += L[j]
        cumsum_A += A[j]

        V[j] = 0
        Y[j] = 0

        if RCT == 0:
            logit_v = (Rx[j] * (b0 + b1 * (cumsum_A / 207.0) + b2 * (t[j] / 170.0) ** 2)
                       + (1 - Rx[j]) * (b3 + b4 * cumsum_L / 24.0 + b5 * (t[j] / 170.0) ** 2))
            p_v = expit(logit_v)
            if np.random.rand() < p_v:
                V[j] = 1
                break

        logit_y = (a0
                   + a1 * (t[j] / 170.0) ** 2
                   + (a2 * sofa) * (cumsum_L / 24.0) ** 2
                   + (a3 * (age / 90.0)) * (cumsum_A / 207.0)
                   + u_shift_y)
        p_y = expit(logit_y)
        if np.random.rand() < p_y:
            Y[j] = 1
            break

    end = last_index + 1
    return (t[:end].copy(), L[:end].copy(), A[:end].copy(),
            V[:end].copy(), Y[:end].copy(), Rx[:end].copy())
