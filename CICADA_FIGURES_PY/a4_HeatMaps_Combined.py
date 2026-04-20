"""
Python port of CICADA_FIGURES/a4_HeatMaps_Combined.m

Produces: Fig_heatmap_figure.pdf

Ultra-tight 2x3 heatmap grid comparing aggressive (top row) and normal
(bottom row) treatment strata for s1, s0, and (s1 - s0).
"""

import matplotlib

matplotlib.use("Agg")

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

np.random.seed(0)


def _load_mat(path):
    try:
        return loadmat(path)
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as h:
            return {k: np.array(h[k]) for k in h.keys()}


def main():
    agg_path = os.path.join(REPO_ROOT, "HeatMapAggressive.mat")
    norm_path = os.path.join(REPO_ROOT, "HeatMapData.mat")
    for p in (agg_path, norm_path):
        if not os.path.exists(p):
            print(f"[a4_HeatMaps_Combined] Missing required input: {p}. Exiting.")
            return 0

    sigma = 1.5

    agg = _load_mat(agg_path)
    s0_agg = np.asarray(agg["s0"], dtype=float)
    s1_agg = np.asarray(agg["s1"], dtype=float)
    A2 = np.asarray(agg["A2"], dtype=float).ravel()
    A3 = np.asarray(agg["A3"], dtype=float).ravel()

    s0_smooth_agg = gaussian_filter(s0_agg, sigma=sigma)
    s1_smooth_agg = gaussian_filter(s1_agg, sigma=sigma)
    diff_smooth_agg = s1_smooth_agg - s0_smooth_agg

    nrm = _load_mat(norm_path)
    s0 = np.asarray(nrm["s0"], dtype=float)
    s1 = np.asarray(nrm["s1"], dtype=float)

    s0_smooth = gaussian_filter(s0, sigma=sigma)
    s1_smooth = gaussian_filter(s1, sigma=sigma)
    diff_smooth = s1_smooth - s0_smooth

    all_data = np.concatenate(
        [
            s0_smooth_agg.ravel(),
            s1_smooth_agg.ravel(),
            diff_smooth_agg.ravel(),
            s0_smooth.ravel(),
            s1_smooth.ravel(),
            diff_smooth.ravel(),
        ]
    )
    global_min = float(np.min(all_data))
    global_max = float(np.max(all_data))
    n_contours = 10

    # Margins & half gaps (matching MATLAB)
    left_margin = 0.02
    right_margin = 0.15
    bottom_margin = 0.08
    top_margin = 0.05
    gap_h = 0.005
    gap_v = 0.02

    plot_w = (1 - left_margin - right_margin - 2 * gap_h) / 3
    plot_h = (1 - bottom_margin - top_margin - gap_v) / 2

    x1 = left_margin
    x2 = x1 + plot_w + gap_h
    x3 = x2 + plot_w + gap_h
    y_bottom = bottom_margin
    y_top = y_bottom + plot_h + gap_v

    positions = [
        (x1, y_top, plot_w, plot_h),
        (x2, y_top, plot_w, plot_h),
        (x3, y_top, plot_w, plot_h),
        (x1, y_bottom, plot_w, plot_h),
        (x2, y_bottom, plot_w, plot_h),
        (x3, y_bottom, plot_w, plot_h),
    ]

    # MATLAB transposes with ' so we do .T here for imshow (data[row=A3, col=A2])
    titles = ["S_1", "S_0", "S_1 - S_0", "S_1^*", "S_0", "S_1^* - S_0"]
    data_arr = [
        s1_smooth_agg.T,
        s0_smooth_agg.T,
        diff_smooth_agg.T,
        s1_smooth.T,
        s0_smooth.T,
        diff_smooth.T,
    ]

    fig_w, fig_h = 7.0, 4.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    last_im = None
    extent = [float(A2.min()), float(A2.max()), float(A3.min()), float(A3.max())]

    for idx, (pos, title, d) in enumerate(zip(positions, titles, data_arr)):
        ax = fig.add_axes(pos)
        im = ax.imshow(
            d,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="hot",
            vmin=global_min,
            vmax=global_max,
            interpolation="nearest",
        )
        last_im = im

        # Contours on smoothed data
        X, Y = np.meshgrid(A2, A3)
        ax.contour(X, Y, d, levels=n_contours, colors="k", linewidths=0.5)

        # Title text placed at (25, 45) in data coords
        ax.text(
            25,
            45,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor=(1, 1, 1, 0.7), edgecolor="none", pad=1.5),
        )

        ax.set_xlim(1, 50)
        ax.set_ylim(1, 50)

        if idx != 3:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel("Harm from L", fontsize=11)
            ax.set_ylabel("Harm from A", fontsize=11)
            ax.tick_params(labelsize=9)

    # Right-side colorbar
    cb_width = 0.02
    cb_height = 2 * plot_h + gap_v
    cb_x = 1 - right_margin + 0.01
    cb_y = bottom_margin
    cax = fig.add_axes([cb_x, cb_y, cb_width, cb_height])
    fig.colorbar(last_im, cax=cax)

    out_path = os.path.join(HERE, "Fig_heatmap_figure.pdf")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved as {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
