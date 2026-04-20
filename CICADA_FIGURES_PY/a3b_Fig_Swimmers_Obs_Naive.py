"""a3b_Fig_Swimmers_Obs_Naive.py

Port of CICADA_FIGURES/a3b_Fig_Swimmers_Obs_Naive.m.

Produces Fig4_swimmer_survival_plot_Obs_Naive.pdf: two swimmer plots (L and A)
for observational data plus a KM panel comparing naive KM (from trialData0.csv)
vs RCT ground truth (from trialData1.csv).
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
    print(f"[a3b] WARNING: could not import helpers: {exc}")
    sys.exit(0)


def _naive_km(times: np.ndarray, events: np.ndarray):
    """Kaplan-Meier estimator, returns step (t, S) including initial (0, 1).

    times: observed time per subject
    events: 1 if death, 0 if censored
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    if times.size == 0:
        return np.array([0.0]), np.array([1.0])

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]

    event_times = np.unique(times[events == 1])
    n = times.size
    at_risk = n
    S = 1.0
    t_out = [0.0]
    s_out = [1.0]
    idx = 0
    for te in event_times:
        while idx < n and times[idx] < te:
            at_risk -= 1
            idx += 1
        d_i = int(np.sum((times == te) & (events == 1)))
        if at_risk > 0:
            S *= 1.0 - d_i / at_risk
        t_out.append(float(te))
        s_out.append(float(S))
        k = 0
        while idx + k < n and times[idx + k] == te:
            k += 1
        at_risk -= k
        idx += k
    return np.asarray(t_out), np.asarray(s_out)


def main() -> int:
    trial0 = os.path.join(REPO_ROOT, "trialData0.csv")
    trial1 = os.path.join(REPO_ROOT, "trialData1.csv")
    if not (os.path.exists(trial0) and os.path.exists(trial1)):
        print(f"[a3b] WARNING: missing input CSVs; skipping.")
        return 0

    T0 = pd.read_csv(trial0)
    T1 = pd.read_csv(trial1)

    # Swimmer panel
    fig = fcnSingleSwimmerPlot_v4(T0)
    fig.set_size_inches(6.0, 9.0)

    swimmer_axes = [a for a in fig.axes if len(a.images) > 0]
    left_margin = 0.06
    right_margin = 0.10
    top_margin = 0.02
    bottom_margin = 0.05
    vertical_gap = 0.02
    width = 1.0 - left_margin - right_margin
    surv_height = 0.28
    remaining = 1.0 - top_margin - bottom_margin - surv_height - 2 * vertical_gap
    swim_height = remaining / 2.0

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

    ax_surv = fig.add_axes([left_margin, bottom_margin, width, surv_height])

    # Compute naive KM by *initial* treatment assignment on T0.
    unique_patients = np.unique(T0["sid"].values)
    trt_times, trt_events = [], []
    unt_times, unt_events = [], []
    for pid in unique_patients:
        pdata = T0[T0["sid"] == pid].sort_values("t")
        is_trt = int(pdata["Rx"].iloc[0]) == 1
        y = pdata["Y"].to_numpy()
        tvals = pdata["t"].to_numpy()
        if (y > 0).any():
            idx = int(np.argmax(y > 0))
            event_time = float(tvals[idx])
            event = 1
        else:
            event_time = float(tvals[-1])
            event = 0
        if is_trt:
            trt_times.append(event_time)
            trt_events.append(event)
        else:
            unt_times.append(event_time)
            unt_events.append(event)

    t0_naive, s0_naive = _naive_km(np.array(unt_times), np.array(unt_events))
    t1_naive, s1_naive = _naive_km(np.array(trt_times), np.array(trt_events))

    # RCT ground-truth KM
    s0_true, s1_true, t0_true, t1_true = fcnPlotKM(T1)

    # Dashed = RCT truth
    ax_surv.plot(t0_true, s0_true, "--", color=(0.4, 0.4, 0.8), linewidth=2.5)
    ax_surv.plot(t1_true, s1_true, "--", color=(0.0, 0.0, 0.5), linewidth=2.5)

    # Solid step = naive observational
    ax_surv.step(t0_naive, s0_naive, where="post",
                 color=(0.4, 0.4, 0.8), linewidth=2.5)
    ax_surv.step(t1_naive, s1_naive, where="post",
                 color=(0.0, 0.0, 0.5), linewidth=2.5)

    # Labels at MATLAB-specified hand-tuned positions
    ax_surv.text(131.21, 0.4020, "Treated (Naive)",
                 color=(0.0, 0.0, 0.5), fontsize=10, fontweight="bold",
                 ha="left")
    ax_surv.text(91.87, 0.7253, "Treated (RCT)",
                 color=(0.0, 0.0, 0.5), fontsize=10, fontweight="bold",
                 ha="left")
    ax_surv.text(80.66, 0.6789, "Untreated (Naive)",
                 color=(0.4, 0.4, 0.8), fontsize=10, fontweight="bold",
                 ha="right")
    ax_surv.text(61.69, 0.5247, "Untreated (RCT)",
                 color=(0.4, 0.4, 0.8), fontsize=10, fontweight="bold",
                 ha="right")

    ax_surv.set_xlabel("Time (hours)", fontsize=12)
    ax_surv.text(5.0, 0.3, "Survival Probability",
                 fontsize=12, fontweight="bold", rotation=90)
    ax_surv.grid(True)
    ax_surv.tick_params(labelsize=11)
    ax_surv.set_xlim(0, 168)
    ax_surv.set_ylim(0, 1)
    ax_surv.set_xticks(np.arange(0, 169, 24))

    out_pdf = os.path.join(HERE, "Fig4_swimmer_survival_plot_Obs_Naive.pdf")
    fig.savefig(out_pdf, format="pdf", dpi=300)
    plt.close(fig)
    print(f"[a3b] Wrote {out_pdf}")

    # ---- text summary ---------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(HERE, f"observational_naive_analysis_results_{stamp}.txt")
    try:
        naive_unt = float(s0_naive[-1]) * 100.0
        naive_trt = float(s1_naive[-1]) * 100.0
        naive_ate = naive_trt - naive_unt
        rct_unt = float(s0_true[-1]) * 100.0
        rct_trt = float(s1_true[-1]) * 100.0
        rct_ate = rct_trt - rct_unt

        with open(txt, "w") as f:
            f.write("=" * 58 + "\n")
            f.write("OBSERVATIONAL NAIVE ANALYSIS RESULTS FOR PAPER\n")
            f.write(f"Generated on: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write("=" * 58 + "\n\n")
            f.write("STUDY CHARACTERISTICS:\n")
            f.write("- Study type: Observational (RCT=0)\n")
            f.write("- Analysis method: Naive Kaplan-Meier\n")
            f.write(f"- Sample size: {len(unique_patients)} patients\n")
            f.write("- Study period: 168 hours\n")
            f.write("- Data file: trialData0.csv\n\n")
            f.write("TREATMENT ASSIGNMENT (Biased):\n")
            f.write(
                f"- Treated patients: {len(trt_times)} "
                f"({len(trt_times)/len(unique_patients)*100:.1f}%)\n"
            )
            f.write(
                f"- Untreated patients: {len(unt_times)} "
                f"({len(unt_times)/len(unique_patients)*100:.1f}%)\n\n"
            )
            f.write("SURVIVAL OUTCOMES COMPARISON (168 hours):\n")
            f.write(f"- RCT (Ground Truth): Untreated {rct_unt:.1f}%, "
                    f"Treated {rct_trt:.1f}%, ATE {rct_ate:.1f}%\n")
            f.write(f"- Naive Observational: Untreated {naive_unt:.1f}%, "
                    f"Treated {naive_trt:.1f}%, ATE {naive_ate:.1f}%\n\n")
            f.write("BIAS QUANTIFICATION:\n")
            f.write(f"- Untreated survival bias: {naive_unt - rct_unt:+.1f}%\n")
            f.write(f"- Treated survival bias:   {naive_trt - rct_trt:+.1f}%\n")
            f.write(f"- ATE bias:                {naive_ate - rct_ate:+.1f}%\n\n")
            f.write("FIGURE GENERATION:\n")
            f.write("- Output file: Fig4_swimmer_survival_plot_Obs_Naive.pdf\n")
            f.write("- Format: PDF vector graphics (300 DPI)\n")
            f.write("- Dimensions: 6 x 9 inches\n")
        print(f"[a3b] Wrote {txt}")
    except Exception as exc:
        warnings.warn(f"[a3b] text summary failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
