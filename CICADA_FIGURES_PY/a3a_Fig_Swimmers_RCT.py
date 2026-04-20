"""a3a_Fig_Swimmers_RCT.py

Port of CICADA_FIGURES/a3a_Fig_Swimmers_RCT.m.

Produces Fig3_swimmer_survival_plot_RCT.pdf: two swimmer plots (L and A)
plus a Kaplan-Meier survival panel for the RCT (trialData1.csv).
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY_HELPERS = os.path.join(REPO_ROOT, "python")
sys.path.insert(0, PY_HELPERS)

np.random.seed(0)

try:
    from fcnSingleSwimmerPlot_v4 import fcnSingleSwimmerPlot_v4  # type: ignore
    from fcnPlotKM import fcnPlotKM  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[a3a] WARNING: could not import helpers: {exc}")
    sys.exit(0)


def _input_csv(name: str) -> str:
    return os.path.join(REPO_ROOT, name)


def main() -> int:
    trial1 = _input_csv("trialData1.csv")
    if not os.path.exists(trial1):
        print(f"[a3a] WARNING: missing input {trial1}; skipping.")
        return 0

    T1 = pd.read_csv(trial1)

    # Build swimmer figure (two panels, L and A).
    fig = fcnSingleSwimmerPlot_v4(T1)
    fig.set_size_inches(6.0, 9.0)

    # Reposition existing swimmer axes into the top 2/3 of the figure and add a
    # survival axis below, matching MATLAB's layout in a3a.
    axes = [a for a in fig.axes if not a.get_label().startswith("<colorbar>")]
    # Filter out colorbars by checking if they have a regular artist pattern.
    swimmer_axes = [a for a in fig.axes if a.get_ylabel() == "" and len(a.images) > 0]

    if len(swimmer_axes) < 2:
        # Fallback: just use first two non-colorbar axes.
        swimmer_axes = [a for a in fig.axes if a not in getattr(fig, "_colorbars", [])][:2]

    # Figure-wide layout
    left_margin = 0.06
    right_margin = 0.10
    top_margin = 0.02
    bottom_margin = 0.05
    vertical_gap = 0.02

    width = 1.0 - left_margin - right_margin
    # Reserve roughly half the figure for the survival plot.
    surv_height = 0.28
    remaining = 1.0 - top_margin - bottom_margin - surv_height - 2 * vertical_gap
    swim_height = remaining / 2.0

    # Re-position the two swimmer axes (top = L, bottom = A) at the top.
    if len(swimmer_axes) >= 2:
        swimmer_axes[0].set_position(
            [left_margin, 1.0 - top_margin - swim_height, width, swim_height]
        )
        swimmer_axes[1].set_position(
            [
                left_margin,
                1.0 - top_margin - 2 * swim_height - vertical_gap,
                width,
                swim_height,
            ]
        )

    # Survival panel
    ax_surv = fig.add_axes(
        [left_margin, bottom_margin, width, surv_height]
    )

    # KM curves from the RCT data
    s0_true, s1_true, t0_true, t1_true = fcnPlotKM(T1)

    ax_surv.plot(
        t0_true, s0_true, "-", color=(0.4, 0.4, 0.8), linewidth=2.5,
        label="Untreated (RCT)",
    )
    ax_surv.plot(
        t1_true, s1_true, "-", color=(0.0, 0.0, 0.5), linewidth=2.5,
        label="Treated (RCT)",
    )

    # On-curve labels (approx; MATLAB places them 0.10 below the curve at t=130)
    label_time = 130.0
    idx0 = int(np.argmin(np.abs(t0_true - label_time)))
    idx1 = int(np.argmin(np.abs(t1_true - label_time)))
    ax_surv.text(
        label_time,
        max(0.0, float(s0_true[idx0]) - 0.10),
        "Untreated (RCT)",
        color=(0.4, 0.4, 0.8),
        fontsize=12,
        fontweight="bold",
        ha="center",
    )
    ax_surv.text(
        label_time,
        max(0.0, float(s1_true[idx1]) - 0.10),
        "Treated (RCT)",
        color=(0.0, 0.0, 0.5),
        fontsize=12,
        fontweight="bold",
        ha="center",
    )

    ax_surv.set_xlabel("Time (hours)", fontsize=12)
    ax_surv.text(
        5.0, 0.3, "Survival Probability",
        fontsize=12, fontweight="bold", rotation=90,
    )
    ax_surv.grid(True)
    ax_surv.tick_params(labelsize=11)
    ax_surv.set_xlim(0, 168)
    ax_surv.set_ylim(0, 1)
    ax_surv.set_xticks(np.arange(0, 169, 24))

    out_pdf = os.path.join(HERE, "Fig3_swimmer_survival_plot_RCT.pdf")
    fig.savefig(out_pdf, format="pdf", dpi=300)
    plt.close(fig)
    print(f"[a3a] Wrote {out_pdf}")

    # ---- text summary ---------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(HERE, f"rct_swimmer_analysis_results_{stamp}.txt")

    try:
        unique_patients = np.unique(T1["sid"].values)
        N = int(len(unique_patients))

        treated = 0
        untreated = 0
        treated_deaths = 0
        untreated_deaths = 0
        tot_hours_trt = 0.0
        tot_hours_unt = 0.0
        for pid in unique_patients:
            pdata = T1[T1["sid"] == pid].sort_values("t")
            is_trt = int(pdata["Rx"].iloc[0]) == 1
            died = bool((pdata["Y"] > 0).any())
            mt = float(pdata["t"].max())
            if is_trt:
                treated += 1
                if died:
                    treated_deaths += 1
                tot_hours_trt += mt
            else:
                untreated += 1
                if died:
                    untreated_deaths += 1
                tot_hours_unt += mt

        final_unt = float(s0_true[-1]) * 100.0
        final_trt = float(s1_true[-1]) * 100.0
        ate = final_trt - final_unt

        with open(txt, "w") as f:
            f.write("=" * 58 + "\n")
            f.write("RCT SWIMMER PLOT ANALYSIS RESULTS FOR PAPER\n")
            f.write(f"Generated on: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write("=" * 58 + "\n\n")
            f.write("RCT DATA CHARACTERISTICS:\n")
            f.write(f"- Sample size: {N} patients\n")
            f.write("- Study type: Randomized Controlled Trial (RCT=1)\n")
            f.write("- Study period: 168 hours\n")
            f.write("- Data file: trialData1.csv\n\n")
            f.write("TREATMENT ASSIGNMENT:\n")
            f.write(
                f"- Treated patients: {treated} ({treated/N*100:.1f}%)\n"
            )
            f.write(
                f"- Untreated patients: {untreated} ({untreated/N*100:.1f}%)\n"
            )
            f.write("- Assignment method: Randomized (50% probability)\n\n")
            f.write("PRIMARY ENDPOINT (168 hours):\n")
            f.write(f"- Untreated survival: {final_unt:.1f}%\n")
            f.write(f"- Treated survival: {final_trt:.1f}%\n")
            f.write(f"- Average Treatment Effect (ATE): {ate:.1f}% points\n\n")
            f.write("MORTALITY:\n")
            if untreated > 0:
                f.write(
                    f"- Untreated deaths: {untreated_deaths}/{untreated} "
                    f"({untreated_deaths/untreated*100:.1f}%)\n"
                )
            if treated > 0:
                f.write(
                    f"- Treated deaths: {treated_deaths}/{treated} "
                    f"({treated_deaths/treated*100:.1f}%)\n"
                )
            f.write("\nFIGURE GENERATION:\n")
            f.write("- Output file: Fig3_swimmer_survival_plot_RCT.pdf\n")
            f.write("- Format: PDF vector graphics (300 DPI)\n")
            f.write("- Dimensions: 6 x 9 inches\n")
        print(f"[a3a] Wrote {txt}")
    except Exception as exc:
        warnings.warn(f"[a3a] text summary failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
