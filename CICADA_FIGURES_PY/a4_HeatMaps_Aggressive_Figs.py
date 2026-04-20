"""
Python port of CICADA_FIGURES/a4_HeatMaps_Aggressive_Figs.m

Produces: Fig_heatmaps.pdf  (matches the MATLAB script's implicit figure
output; MATLAB only drew to figure(1) without an explicit print call, so
we save an analogous 3-panel aggressive heatmap under a descriptive name).
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
    if not os.path.exists(agg_path):
        print(f"[a4_HeatMaps_Aggressive] Missing required input: {agg_path}. Exiting.")
        return 0

    d = _load_mat(agg_path)
    s0 = np.asarray(d["s0"], dtype=float)
    s1 = np.asarray(d["s1"], dtype=float)
    A2 = np.asarray(d["A2"], dtype=float).ravel()
    A3 = np.asarray(d["A3"], dtype=float).ravel()

    sigma = 1.5
    s0_smooth = gaussian_filter(s0, sigma=sigma)
    s1_smooth = gaussian_filter(s1, sigma=sigma)
    diff_smooth = s1_smooth - s0_smooth

    all_data = np.concatenate([s0_smooth.ravel(), s1_smooth.ravel(), diff_smooth.ravel()])
    global_min = float(all_data.min())
    global_max = float(all_data.max())

    gap = 0.02
    plot_height = 0.35
    plot_width = 0.25
    y_pos = 0.3
    pos1 = (0.05, y_pos, plot_width, plot_height)
    pos2 = (0.05 + plot_width + gap, y_pos, plot_width, plot_height)
    pos3 = (0.05 + 2 * (plot_width + gap), y_pos, plot_width, plot_height)
    n_contours = 10

    extent = [float(A2.min()), float(A2.max()), float(A3.min()), float(A3.max())]
    X, Y = np.meshgrid(A2, A3)

    fig = plt.figure(figsize=(10, 5), facecolor="white")

    panels = [
        (pos1, s1_smooth.T, "S1*", True, False),
        (pos2, s0_smooth.T, "S0", False, False),
        (pos3, diff_smooth.T, "S1* - S0", False, True),
    ]

    last_im = None
    for pos, data, title, show_y, show_xlabel in panels:
        ax = fig.add_axes(pos)
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="hot",
            vmin=global_min,
            vmax=global_max,
            interpolation="nearest",
        )
        ax.contour(X, Y, data, levels=n_contours, colors="k", linewidths=0.5)
        ax.set_xlim(1, 50)
        ax.set_ylim(1, 50)
        ax.set_xticklabels([])
        if not show_y:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Harm from A")
        if show_xlabel:
            ax.set_xlabel("Harm from L")
        ax.set_title(title)
        last_im = im

    cb_x = pos3[0] + pos3[2] + gap
    fig.add_axes([cb_x, y_pos, 0.02, plot_height])
    cax = fig.axes[-1]
    fig.colorbar(last_im, cax=cax)

    # "x-label for all plots" centered near bottom — mirrors MATLAB text() call
    fig.text(0.4, 0.2, "Harm from L", ha="center", fontsize=12)

    out_path = os.path.join(HERE, "Fig_heatmaps.pdf")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved as {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
