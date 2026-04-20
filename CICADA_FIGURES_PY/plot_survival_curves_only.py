"""
Python port of CICADA_FIGURES/plot_survival_curves_only.m

Diagnostic utility: plots survival curves with 95% bootstrap CIs, overlaid
with RCT reference KM curves. Output: Fig_survival_curves_only.pdf.
"""

import matplotlib

matplotlib.use("Agg")

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

np.random.seed(0)

from fcnPlotKM import fcnPlotKM  # type: ignore


def _load_mat(path):
    try:
        return loadmat(path)
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as h:
            return {k: np.array(h[k]) for k in h.keys()}


def main():
    boot_path = os.path.join(REPO_ROOT, "bootstrap_confidence_bands.mat")
    trial_path = os.path.join(REPO_ROOT, "trialData1.csv")

    if not os.path.exists(boot_path):
        print(f"[plot_survival_curves_only] Missing: {boot_path}. Exiting.")
        return 0
    if not os.path.exists(trial_path):
        print(f"[plot_survival_curves_only] Missing: {trial_path}. Exiting.")
        return 0

    d = _load_mat(boot_path)
    s0_lower = np.asarray(d["s0_lower"], dtype=float).ravel()
    s0_upper = np.asarray(d["s0_upper"], dtype=float).ravel()
    s1_lower = np.asarray(d["s1_lower"], dtype=float).ravel()
    s1_upper = np.asarray(d["s1_upper"], dtype=float).ravel()
    s0_median = np.asarray(d["s0_median"], dtype=float).ravel()
    s1_median = np.asarray(d["s1_median"], dtype=float).ravel()
    Nboot = int(np.asarray(d["Nboot"]).ravel()[0])

    T1 = pd.read_csv(trial_path)
    s0_true, s1_true, t0_true, t1_true = fcnPlotKM(T1)

    t_grid = np.arange(0, 168 + 2, 2)

    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)

    ax.fill_between(
        t_grid,
        s0_lower,
        s0_upper,
        facecolor=(0.2, 0.4, 0.8),
        alpha=0.3,
        edgecolor="none",
        label="Untreated 95% CI",
    )
    ax.fill_between(
        t_grid,
        s1_lower,
        s1_upper,
        facecolor=(0.8, 0.2, 0.2),
        alpha=0.3,
        edgecolor="none",
        label="Treated 95% CI",
    )
    ax.plot(t0_true, s0_true, "b--", linewidth=2.5, label="Untreated (RCT)")
    ax.plot(t1_true, s1_true, "r--", linewidth=2.5, label="Treated (RCT)")
    ax.plot(t_grid, s0_median, "b-", linewidth=2.5, label="Untreated (Estimated)")
    ax.plot(t_grid, s1_median, "r-", linewidth=2.5, label="Treated (Estimated)")

    ax.set_xlabel("Time (hours)", fontsize=14)
    ax.set_ylabel("Survival Probability", fontsize=14)
    ax.set_title(
        "Survival Curves with 95% Bootstrap Confidence Intervals", fontsize=16
    )
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 169, 24))

    # Summary
    print("\n=== Survival Analysis Summary ===")
    print(f"Bootstrap samples: {Nboot}")
    print("\nSurvival Probabilities:")
    print("-------------------------------------------")
    print("Time (h) | Untreated         | Treated")
    print("         | Median [95% CI]   | Median [95% CI]")
    print("-------------------------------------------")
    for kt in [24, 48, 72, 96, 120, 144, 168]:
        idx = int(np.argmin(np.abs(t_grid - kt)))
        print(
            f"{kt:8d} | {s0_median[idx]*100:.1f}% [{s0_lower[idx]*100:.1f}-{s0_upper[idx]*100:.1f}] | "
            f"{s1_median[idx]*100:.1f}% [{s1_lower[idx]*100:.1f}-{s1_upper[idx]*100:.1f}]"
        )

    te = s1_median[-1] - s0_median[-1]
    print("\n-------------------------------------------")
    print("Treatment Effect at 168 hours:")
    print(f"  Absolute Risk Reduction: {te*100:.1f}%")
    if te != 0:
        print(f"  Number Needed to Treat: {1/te:.1f}")

    out_path = os.path.join(HERE, "Fig_survival_curves_only.pdf")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nFigure saved as: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
