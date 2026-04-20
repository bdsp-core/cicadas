"""
Python port of CICADA_FIGURES/a5_OptimizationCurve.m

Produces: Fig_optimization_curves_with_survival.pdf

Main optimization curve (ATE vs treatment threshold theta) with four
survival-curve insets for theta in {0, 0.02, 0.1, 0.8}. Also writes a
text summary file (optimization_curve_analysis_results_<stamp>.txt).
"""

import matplotlib

matplotlib.use("Agg")

import os
import sys
import datetime as _dt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

np.random.seed(0)


def _load_mat(path):
    try:
        return loadmat(path)
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as h:
            return {k: np.array(h[k]) for k in h.keys()}


def main():
    a01_path = os.path.join(REPO_ROOT, "A01Data.mat")
    three_path = os.path.join(REPO_ROOT, "ThreeCurves.mat")
    for p in (a01_path, three_path):
        if not os.path.exists(p):
            print(f"[a5_OptimizationCurve] Missing required input: {p}. Exiting.")
            return 0

    a01 = _load_mat(a01_path)
    three = _load_mat(three_path)

    A0 = np.asarray(a01["A0"], dtype=float)  # (Nboot, nth)
    A1 = np.asarray(a01["A1"], dtype=float)
    th = np.asarray(a01["th"], dtype=float).ravel()

    S0ref = np.asarray(three["S0ref"], dtype=float)
    S1ref = np.asarray(three["S1ref"], dtype=float)
    S0est = np.asarray(three["S0est"], dtype=float)
    S1est = np.asarray(three["S1est"], dtype=float)

    alpha = 0.05
    if A0.shape[0] > 1:
        A0_lower = np.percentile(A0, 100 * alpha / 2, axis=0)
        A0_upper = np.percentile(A0, 100 * (1 - alpha / 2), axis=0)
        A0_median = np.percentile(A0, 50, axis=0)
        A1_median = np.percentile(A1, 50, axis=0)
    else:
        A0_lower = A0.ravel()
        A0_upper = A0.ravel()
        A0_median = A0.ravel()
        A1_median = A1.ravel()

    fig_width, fig_height = 10.0, 8.0
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    subplot_height = 0.23
    subplot_width = 0.17
    subplot_y = 0.50
    subplot_spacing = 0.055
    main_left = 0.1

    t = np.arange(0, 168 + 2, 2)  # 0:2:168 -> 85 points

    # Row indices in S*est/S*ref correspond to th = 0, 0.02, 0.1, 0.8 (four rows)
    theta_vals = [0.0, 0.02, 0.1, 0.8]

    inset_axes = []
    for i, theta in enumerate(theta_vals):
        x = main_left + i * (subplot_width + subplot_spacing)
        ax = fig.add_axes([x, subplot_y, subplot_width, subplot_height])
        inset_axes.append(ax)

        # Observational (solid)
        ax.plot(t, S0est[i, :], "-", color=(0.4, 0.4, 0.8), linewidth=1.5)
        ax.plot(t, S1est[i, :], "-", color=(0.0, 0.0, 0.5), linewidth=1.5)
        # RCT (dashed)
        ax.plot(t, S0ref[i, :], "--", color=(0.4, 0.4, 0.8), linewidth=1.0)
        ax.plot(t, S1ref[i, :], "--", color=(0.0, 0.0, 0.5), linewidth=1.0)

        ax.set_xlabel("Time (h)", fontsize=9)
        if i == 0:
            ax.set_ylabel("Survival", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(rf"$\theta$ = {theta:g}", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 168)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.tick_params(labelsize=8)
        ax.set_xticks([0, 84, 168])
        # Only keep left+bottom spines for "inset" look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(1.5)

    # Main ATE optimization plot
    main_height = 0.35
    main_width = 0.85
    main_y = 0.1
    ax_main = fig.add_axes([main_left, main_y, main_width, main_height])

    th = th.ravel()
    A0_lower = A0_lower.ravel()
    A0_upper = A0_upper.ravel()
    A0_median = A0_median.ravel()
    A1_median = A1_median.ravel()

    print("\n=== Debug Information ===")
    print(f"th: {th.size} elements, range [{th.min():.3f}, {th.max():.3f}]")
    print(f"A0_median: {A0_median.size} elements, range [{A0_median.min():.3f}, {A0_median.max():.3f}]")
    print(f"A1_median: {A1_median.size} elements, range [{A1_median.min():.3f}, {A1_median.max():.3f}]")
    print(f"A0 matrix dimensions: {A0.shape[0]} x {A0.shape[1]}")
    print("========================\n")

    # Confidence band (observational)
    ax_main.fill_between(
        th, A0_lower, A0_upper, facecolor=(0.4, 0.4, 0.8), alpha=0.3, edgecolor="none"
    )
    ax_main.plot(th, A0_median, "-", color=(0.4, 0.4, 0.8), linewidth=2.5)
    ax_main.plot(th, A1_median, "-", color=(0.0, 0.0, 0.5), linewidth=2.5)

    idx_A0 = int(np.argmax(A0_median))
    max_A0 = float(A0_median[idx_A0])
    th_opt_A0 = float(th[idx_A0])
    ax_main.plot(
        th_opt_A0,
        max_A0,
        "o",
        markerfacecolor=(0.4, 0.4, 0.8),
        markeredgecolor=(0.4, 0.4, 0.8),
        markersize=10,
    )

    # Vertical lines + markers at four anchor thresholds
    def _find_near(value):
        m = np.where(np.abs(th - value) < 0.01)[0]
        return int(m[0]) if m.size else None

    anchor_idxs = {v: _find_near(v) for v in theta_vals}

    y_min_pre = min(A0_lower.min(), A1_median.min(), A0_median.min()) - 0.02
    y_max_pre = max(A0_upper.max(), A1_median.max(), A0_median.max()) + 0.02

    for v in theta_vals:
        ax_main.plot([v, v], [y_min_pre, y_max_pre], "-", color=(0.7, 0.7, 0.7), linewidth=1)
        idx = anchor_idxs[v]
        if idx is not None:
            ax_main.plot(
                v,
                A0_median[idx],
                "o",
                markerfacecolor="white",
                markeredgecolor=(0.4, 0.4, 0.8),
                markersize=8,
                markeredgewidth=2,
            )

    # Dashed vertical at the optimum
    ax_main.plot(
        [th_opt_A0, th_opt_A0],
        [y_min_pre, y_max_pre],
        "--",
        color=(0.4, 0.4, 0.8),
        linewidth=1.5,
    )

    ax_main.set_xlabel(r"Treatment Threshold ($\theta$)", fontsize=12)
    ax_main.set_ylabel("Average Treatment Effect (ATE)", fontsize=12)
    ax_main.grid(True)
    ax_main.tick_params(labelsize=11)
    ax_main.set_xlim(-0.05, float(th.max()))
    ax_main.set_ylim(y_min_pre, y_max_pre)

    # Figure-level annotation lines connecting inset theta=30h point to main-plot theta anchor
    x_range = th.max() - (-0.05)

    def theta_x(v):
        return main_left + main_width * (v - (-0.05)) / x_range

    inset_theta30_offset = subplot_width * (30 / 168)
    inset_xs = [
        main_left + i * (subplot_width + subplot_spacing) + inset_theta30_offset
        for i in range(4)
    ]
    main_anchor_xs = [theta_x(v) for v in theta_vals]

    mid_ys = [0.485, 0.462, 0.455, 0.470]
    main_top = 0.45

    for ix, ax_x, my in zip(inset_xs, main_anchor_xs, mid_ys):
        # vertical down from inset
        fig.add_artist(
            plt.Line2D([ix, ix], [subplot_y, my], color=(0.5, 0.5, 0.5), linewidth=1)
        )
        # horizontal
        fig.add_artist(
            plt.Line2D([ix, ax_x], [my, my], color=(0.5, 0.5, 0.5), linewidth=1)
        )
        # vertical down to main plot
        fig.add_artist(
            plt.Line2D([ax_x, ax_x], [my, main_top], color=(0.5, 0.5, 0.5), linewidth=1)
        )

    out_path = os.path.join(HERE, "Fig_optimization_curves_with_survival.pdf")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nFigure saved as: {out_path}")

    # --- Summary (stdout + text file) ---
    def _ci(idx):
        if idx is None:
            return (float("nan"),) * 3
        return float(A0_median[idx]), float(A0_lower[idx]), float(A0_upper[idx])

    ate_0, ate_0_lo, ate_0_hi = _ci(anchor_idxs[0.0])
    ate_002, ate_002_lo, ate_002_hi = _ci(anchor_idxs[0.02])
    ate_01, ate_01_lo, ate_01_hi = _ci(anchor_idxs[0.1])
    ate_08, ate_08_lo, ate_08_hi = _ci(anchor_idxs[0.8])

    print("\n=== Optimization Results ===")
    print("Observational (G-formula):")
    print(f"  Optimal threshold: {th_opt_A0:.3f}")
    print(
        f"  Maximum ATE: {max_A0:.3f} [{float(A0_lower[idx_A0]):.3f}, "
        f"{float(A0_upper[idx_A0]):.3f}]"
    )

    print("\n=== ATE Values at Displayed Thresholds (for Figure Caption) ===")
    print(f"At theta = 0.00 (treat all):     ATE = {ate_0:.3f} [95% CI: {ate_0_lo:.3f}, {ate_0_hi:.3f}]")
    print(f"At theta = 0.02 (near optimal):  ATE = {ate_002:.3f} [95% CI: {ate_002_lo:.3f}, {ate_002_hi:.3f}]")
    print(f"At theta = 0.10:                 ATE = {ate_01:.3f} [95% CI: {ate_01_lo:.3f}, {ate_01_hi:.3f}]")
    print(f"At theta = 0.80 (restrictive):   ATE = {ate_08:.3f} [95% CI: {ate_08_lo:.3f}, {ate_08_hi:.3f}]")

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(HERE, f"optimization_curve_analysis_results_{stamp}.txt")
    Nboot = int(A0.shape[0])
    with open(txt_path, "w") as fid:
        fid.write("==========================================================\n")
        fid.write("OPTIMIZATION CURVE ANALYSIS RESULTS FOR PAPER\n")
        fid.write(f"Generated on: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fid.write("==========================================================\n\n")
        fid.write("STUDY CHARACTERISTICS:\n")
        fid.write("- Analysis: Treatment threshold optimization with bootstrap\n")
        fid.write("- Data sources: A01Data.mat, ThreeCurves.mat\n")
        fid.write(f"- Bootstrap samples: {Nboot}\n")
        fid.write(
            f"- Threshold range: {th.min():.3f} to {th.max():.3f} ({th.size} levels)\n"
        )
        fid.write("- Key thresholds analyzed: 0.00, 0.02, 0.10, 0.80\n\n")
        fid.write("OPTIMAL THRESHOLD ANALYSIS:\n")
        fid.write(f"- Optimal threshold (G-formula): {th_opt_A0:.3f}\n")
        fid.write(
            f"- Maximum ATE: {max_A0:.3f} [95% CI: {float(A0_lower[idx_A0]):.3f}, {float(A0_upper[idx_A0]):.3f}]\n"
        )
        fid.write("- Optimization method: Bootstrap-based grid search\n\n")

        fid.write("ATE RESULTS AT KEY THRESHOLDS:\n")
        fid.write(f"{'Threshold':<12}{'ATE':<12}{'95% CI':<20}{'Interpretation':<15}\n")
        fid.write(f"{'---------':<12}{'---':<12}{'-----':<20}{'--------------':<15}\n")
        for thr, v, lo, hi in [
            (0.00, ate_0, ate_0_lo, ate_0_hi),
            (0.02, ate_002, ate_002_lo, ate_002_hi),
            (0.10, ate_01, ate_01_lo, ate_01_hi),
            (0.80, ate_08, ate_08_lo, ate_08_hi),
        ]:
            if np.isnan(v):
                continue
            interp = "Beneficial" if v > 0 else "Harmful"
            fid.write(
                f"{thr:<12.2f}{v:<12.3f}[{lo:<8.3f},{hi:<8.3f}] {interp:<15}\n"
            )

        fid.write("\nSURVIVAL OUTCOMES AT KEY THRESHOLDS (168 hours):\n")
        fid.write(
            f"{'Threshold':<12}{'Obs Untrt':<12}{'Obs Treat':<12}{'RCT Untrt':<12}{'RCT Treat':<12}\n"
        )
        for i, thr in enumerate([0.00, 0.02, 0.10, 0.80]):
            fid.write(
                f"{thr:<12.2f}{S0est[i,-1]:<12.3f}{S1est[i,-1]:<12.3f}"
                f"{S0ref[i,-1]:<12.3f}{S1ref[i,-1]:<12.3f}\n"
            )

        ate_range = float(A0_median.max() - A0_median.min())
        ate_variation = float(A0_median.std(ddof=1)) if A0_median.size > 1 else 0.0
        fid.write("\nOPTIMIZATION CURVE CHARACTERISTICS:\n")
        fid.write(
            f"- ATE range across thresholds: {ate_range:.3f} ({ate_range*100:.1f}% points)\n"
        )
        fid.write(f"- ATE standard deviation: {ate_variation:.3f}\n")
        if ate_range > 0.05:
            fid.write("- Curve shape: Strong threshold dependence\n")
        else:
            fid.write("- Curve shape: Relatively flat\n")

        if th.size > 10:
            ate_gradient = np.gradient(A0_median)
            low = np.sum(np.abs(ate_gradient) < 0.01)
            pct = low / th.size * 100
            fid.write(f"- Plateau regions (low gradient): {pct:.0f}% of threshold range\n")

        cv_optimal = float(A0[:, idx_A0].std(ddof=1) / max(A0[:, idx_A0].mean(), 1e-12) * 100)
        ci_width_optimal = float(A0_upper[idx_A0] - A0_lower[idx_A0])
        fid.write("\nBOOTSTRAP UNCERTAINTY ANALYSIS:\n")
        fid.write("Uncertainty at optimal threshold:\n")
        fid.write(f"- Coefficient of variation: {cv_optimal:.1f}%\n")
        fid.write(
            f"- 95% CI width: {ci_width_optimal:.3f} ({ci_width_optimal*100:.1f}% points)\n"
        )

        safe_thresholds = int(np.sum(A0_lower > 0))
        fid.write(
            f"\nSafety assessment:\n- Thresholds with CI lower bound > 0: "
            f"{safe_thresholds}/{th.size} ({safe_thresholds/th.size*100:.1f}%)\n"
        )

        fid.write("\nTECHNICAL DETAILS:\n")
        fid.write("- Figure output: Fig_optimization_curves_with_survival.pdf\n")
        fid.write("- Format: PDF vector graphics (300 DPI)\n")
        fid.write("- Dimensions: 10 x 8 inches\n")
        fid.write("- Layout: 4 survival insets above main optimization curve\n")
        fid.write("- Statistical method: Bootstrap percentiles for confidence intervals\n")
    print(f"Optimization curve analysis results exported to: {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
