# fcn_bootstrapBySID.py

from __future__ import annotations
import numpy as np
import pandas as pd

def fcn_bootstrapBySID(T0: pd.DataFrame, N: int | None = None) -> pd.DataFrame:
    """
    Bootstrap-by-subject-ID (SID), mirroring the MATLAB function.

    Parameters
    ----------
    T0 : pd.DataFrame
        Original data with a 'sid' column (subject ID).
    N : int, optional
        Number of subjects to sample (with replacement). If None, uses the
        number of unique SIDs in T0.

    Returns
    -------
    pd.DataFrame
        Bootstrap sample with SIDs reassigned to 1..N.
    """
    # Unique subject IDs (sorted for stable reporting)
    unique_sids = np.sort(T0["sid"].unique())
    n_subjects = unique_sids.size

    # Default N
    if N is None:
        N = n_subjects

    # Sample original SIDs with replacement (uses current np.random state)
    sampled_sids = np.random.choice(unique_sids, size=N, replace=True)

    # Build bootstrap table; reassign new SIDs 1..N
    frames = []
    for i in range(N):
        subject_rows = T0[T0["sid"] == sampled_sids[i]].copy()
        subject_rows.loc[:, "sid"] = i + 1  # new sequential SID
        frames.append(subject_rows)

    T0_boot = pd.concat(frames, ignore_index=True)

    # ---- Console summary (mirrors MATLAB prints) ----
    print("Bootstrap sample created:")
    print(f"  Original subjects: {n_subjects}")
    print(f"  Sampled subjects:  {N}")
    print(f"  Total rows in original:  {len(T0)}")
    print(f"  Total rows in bootstrap: {len(T0_boot)}")
    print(f"  New subject IDs: 1 to {N}")

    # Sampling frequencies per original SID
    counts = pd.Series(sampled_sids).value_counts()
    not_sampled = n_subjects - counts.size
    sampled_once = int((counts == 1).sum())
    sampled_2plus = int((counts >= 2).sum())

    print("\nSubject sampling frequencies:")
    print(f"  Not sampled:    {not_sampled} subjects")
    print(f"  Sampled once:   {sampled_once} subjects")
    print(f"  Sampled 2+ times: {sampled_2plus} subjects")

    # ID mapping preview
    if N <= 10:
        k = N
        print(f"\nID mapping (first {k} subjects):")
        print("  New ID -> Original ID")
        for i in range(k):
            print(f"  {i+1:6d} -> {int(sampled_sids[i])}")
    else:
        print("\nID mapping (first 5 subjects):")
        print("  New ID -> Original ID")
        for i in range(5):
            print(f"  {i+1:6d} -> {int(sampled_sids[i])}")
        print(f"  ... (showing first 5 of {N})")

    return T0_boot
