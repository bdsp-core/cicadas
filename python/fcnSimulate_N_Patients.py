import numpy as np
import pandas as pd
from typing import Tuple


def fcnSimulate_N_Patients(
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
) -> pd.DataFrame:
    """
    Direct translation of MATLAB:
      T = fcnSimulate_N_Patients(N,RCT,treatProb,th, C, g, ke, L0, parmsControl, parmsY, parmsV, age, sofa)

    Returns:
      pandas.DataFrame with columns: ['age','sofa','sid','t','L','A','V','Y','Rx']
    """

    # ----------------------------
    # Initialize storage arrays
    # ----------------------------
    tt = []
    Lt = []
    At = []
    Vt = []
    Yt = []
    Rxt = []
    sid = []
    Age = []
    Sofa = []

    # ----------------------------
    # 3. PATIENT SIMULATION LOOP
    # ----------------------------
    for i in range(N):
        # Assign treatment based on trial design
        p_i = treatProb[i] if np.ndim(treatProb) else float(treatProb)
        Rx0 = 1 if np.random.rand() < p_i else 0

        # Run individual patient simulation with logit-based hazards
        t_i, L_i, A_i, V_i, Y_i, Rx_i = _fcnRunSimulation(
            RCT, Rx0, th, float(C[i]), float(g[i]), float(age[i]), float(sofa[i]),
            float(ke), np.asarray(L0[i, :], dtype=float), parmsControl, parmsY, parmsV
        )

        sid.extend([i + 1] * len(t_i))  # MATLAB is 1-based for subject IDs
        tt.extend(t_i.tolist())
        Lt.extend(L_i.tolist())
        At.extend(A_i.tolist())
        Vt.extend(V_i.tolist())
        Yt.extend(Y_i.tolist())
        Rxt.extend(Rx_i.tolist())
        Age.extend([float(age[i])] * len(t_i))
        Sofa.extend([float(sofa[i])] * len(t_i))

    # ----------------------------
    # 4. DATA EXPORT AND STORAGE
    # ----------------------------
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inner function translating MATLAB nested function:
      [t, L, A, V, Y, Rx] = fcnRunSimulation(...)
    """

    # --- unpack parameters ---
    ki, Amax = float(parmsControl[0]), float(parmsControl[1])
    a0, a1, a2, a3 = float(parmsY[0]), float(parmsY[1]), float(parmsY[2]), float(parmsY[3])
    a4 = float(parmsY[4]) if len(parmsY) >= 5 else 0.0  # kept for parity; unused below
    b0, b1, b2, b3, b4, b5 = (float(parmsV[0]), float(parmsV[1]), float(parmsV[2]),
                              float(parmsV[3]), float(parmsV[4]), float(parmsV[5]))

    # --- Pre-allocate arrays ---
    dt = 2.0  # [h]
    t = np.arange(0.0, 168.0 + dt, dt)  # 0:dt:168 inclusive
    Nt = t.size

    # Expit function (inverse logit)
    def expit(x):
        return 1.0 / (1.0 + np.exp(-x))

    # PI controller state
    eInt = 0.0

    # Initialize variables (match MATLAB indexing/initialization)
    A = np.zeros(Nt, dtype=float)
    L = np.zeros(Nt, dtype=float)
    V = np.zeros(Nt, dtype=int)
    Y = np.zeros(Nt, dtype=int)
    Rx = np.zeros(Nt, dtype=int)
    X = np.zeros(Nt, dtype=float)  # drug concentration

    Rx[0] = int(Rx0)
    A[0] = Rx0 * 2.0  # Initial pump value
    L[0] = 0.0
    V[0] = 0
    Y[0] = 0
    X[0] = 0.0

    # Cumulative sums for hazard calculations
    cumsum_L = 0.0
    cumsum_A = 0.0

    # --- Simulation loop ---
    last_index = 0
    for j in range(1, Nt):
        last_index = j

        # ---- Disease dynamics with previous pump rate -------------------
        Rx[j] = Rx[j - 1]  # continue treatment status

        if Rx[j] == 1:
            # sX = 1 - 1./((C./X(j-1)).^g + 1);
            # Handle X[j-1] == 0 the same way MATLAB effectively does (C/0 -> inf)
            if X[j - 1] == 0.0:
                sX = 1.0
            else:
                sX = 1.0 - 1.0 / (((C / X[j - 1]) ** g) + 1.0)
        else:
            sX = 1.0

        L[j] = L0[j] * sX

        # ---- PI control (if on treatment) ----------------------------
        if (Rx[j] == 1) and (j > 5):
            e = L[j] - th
            eInt = eInt + e * dt
            Aunsat = ki * eInt

            # Saturation & anti-wind-up
            A[j] = np.clip(Aunsat, 0.0, Amax)
            if A[j] != Aunsat:
                eInt = eInt - (Aunsat - A[j]) / ki
        else:
            A[j] = 0.0

        # X(j) = ke*X(j-1) + A(j);
        X[j] = ke * X[j - 1] + A[j]

        # Update cumulative sums
        cumsum_L += L[j]
        cumsum_A += A[j]

        # ---- Event probabilities (logit scale) --------------------------
        V[j] = 0
        Y[j] = 0  # default values -- can be changed by events below

        # Only happens in observational mode
        if RCT == 0:
            # V: Censoring (due to switching treatment or dropping out)
            logit_v = Rx[j] * (b0 + b1 * (cumsum_A / 207.0) + b2 * (t[j] / 170.0) ** 2) + \
                      (1 - Rx[j]) * (b3 + b4 * cumsum_L / 24.0 + b5 * (t[j] / 170.0) ** 2)
            p_v = expit(logit_v)
            if np.random.rand() < p_v:
                V[j] = 1
                break  # Stop simulation

        # Death hazard - simple model where treatment is harmful
        logit_y = a0 + a1 * (t[j] / 170.0) ** 2 + (a2 * sofa) * (cumsum_L / 24.0) ** 2 + (a3 * (age / 90.0)) * (cumsum_A / 207.0)
        p_y = expit(logit_y)
        if np.random.rand() < p_y:
            Y[j] = 1
            break  # Stop simulation

    # Truncate outputs to the simulated length (up to and including last_index)
    end = last_index + 1
    t_out = t[:end].copy()
    L_out = L[:end].copy()
    A_out = A[:end].copy()
    V_out = V[:end].copy()
    Y_out = Y[:end].copy()
    Rx_out = Rx[:end].copy()

    return t_out, L_out, A_out, V_out, Y_out, Rx_out
