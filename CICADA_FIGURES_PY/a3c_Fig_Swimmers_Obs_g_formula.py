"""a3c_Fig_Swimmers_Obs_g_formula.py

Port of CICADA_FIGURES/a3c_Fig_Swimmers_Obs_g_formula.m.

Produces Fig_gformula_corrected_survival_curves.pdf: g-formula corrected survival
curves with bootstrap 95% CIs, compared to RCT ground truth. No swimmer panels.
Uses bootstrap_confidence_bands.mat from the repo root.
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
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY_HELPERS = os.path.join(REPO_ROOT, "python")
sys.path.insert(0, PY_HELPERS)

np.random.seed(0)

try:
    from fcnPlotKM import fcnPlotKM  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[a3c] WARNING: could not import fcnPlotKM: {exc}")
    sys.exit(0)


def _load_bootstrap(path: str) -> dict | None:
    """Load bootstrap_confidence_bands.mat with h5py fallback."""
    try:
        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        keys = ["s0_lower", "s0_median", "s0_upper",
                "s1_lower", "s1_median", "s1_upper", "t_grid"]
        return {k: np.asarray(d[k]).ravel() for k in keys if k in d}
    except NotImplementedError:
        pass
    except Exception as exc:
        print(f"[a3c] loadmat failed: {exc}; trying h5py fallback.")

    try:
        import h5py  # type: ignore

        out: dict = {}
        with h5py.File(path, "r") as f:
            for k in ["s0_lower", "s0_median", "s0_upper",
                      "s1_lower", "s1_median", "s1_upper", "t_grid"]:
                if k in f:
                    out[k] = np.asarray(f[k]).ravel()
        return out
    except Exception as exc:
        print(f"[a3c] h5py fallback failed: {exc}")
        return None


def main() -> int:
    boot_mat = os.path.join(REPO_ROOT, "bootstrap_confidence_bands.mat")
    trial1 = os.path.join(REPO_ROOT, "trialData1.csv")
    if not os.path.exists(boot_mat):
        print(f"[a3c] WARNING: missing {boot_mat}; skipping.")
        return 0
    if not os.path.exists(trial1):
        print(f"[a3c] WARNING: missing {trial1}; skipping.")
        return 0

    boot = _load_bootstrap(boot_mat)
    if not boot or not all(
        k in boot for k in [
            "s0_lower", "s0_median", "s0_upper",
            "s1_lower", "s1_median", "s1_upper",
        ]
    ):
        print(f"[a3c] WARNING: bootstrap arrays not found in {boot_mat}; skipping.")
        return 0

    s0_lower = boot["s0_lower"]
    s0_median = boot["s0_median"]
    s0_upper = boot["s0_upper"]
    s1_lower = boot["s1_lower"]
    s1_median = boot["s1_median"]
    s1_upper = boot["s1_upper"]

    # MATLAB uses t_grid = 0:2:168 (85 points) — reconstruct locally.
    t_grid = np.arange(0.0, 169.0, 2.0)
    if t_grid.size != s0_median.size:
        # Fall back to whatever was saved in the mat file.
        t_grid = np.asarray(boot.get("t_grid", t_grid)).ravel()

    # RCT ground truth
    T1 = pd.read_csv(trial1)
    s0_true, s1_true, t0_true, t1_true = fcnPlotKM(T1)

    fig = plt.figure(figsize=(7.0, 5.0))
    ax = fig.add_axes([0.15, 0.12, 0.75, 0.75])

    # Confidence bands (gray, behind)
    ax.fill_between(
        t_grid, s0_lower, s0_upper,
        color=(0.7, 0.7, 0.7), alpha=0.3, linewidth=0,
        label="Untreated 95% CI",
    )
    ax.fill_between(
        t_grid, s1_lower, s1_upper,
        color=(0.7, 0.7, 0.7), alpha=0.3, linewidth=0,
        label="Treated 95% CI",
    )

    # RCT reference (dashed)
    ax.plot(t0_true, s0_true, "--", color=(0.4, 0.4, 0.8), linewidth=2.5)
    ax.plot(t1_true, s1_true, "--", color=(0.0, 0.0, 0.5), linewidth=2.5)

    # G-formula medians (solid)
    ax.plot(t_grid, s0_median, "-", color=(0.4, 0.4, 0.8), linewidth=2.5)
    ax.plot(t_grid, s1_median, "-", color=(0.0, 0.0, 0.5), linewidth=2.5)

    # Manual label positions copied from MATLAB
    ax.text(148.22, 0.2435, "Untreated",
            color=(0.4, 0.4, 0.8), fontsize=12, fontweight="bold",
            ha="center")
    ax.text(152.67, 0.5965, "Treated",
            color=(0.0, 0.0, 0.5), fontsize=12, fontweight="bold",
            ha="center")

    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.grid(True)
    ax.tick_params(labelsize=11)
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 169, 24))

    out_pdf = os.path.join(HERE, "Fig_gformula_corrected_survival_curves.pdf")
    fig.savefig(out_pdf, format="pdf", dpi=300)
    plt.close(fig)
    print(f"[a3c] Wrote {out_pdf}")

    # ---- text summary ---------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = os.path.join(HERE, f"gformula_corrected_analysis_results_{stamp}.txt")
    try:
        gf_unt = float(s0_median[-1]) * 100.0
        gf_trt = float(s1_median[-1]) * 100.0
        gf_ate = gf_trt - gf_unt
        rct_unt = float(s0_true[-1]) * 100.0
        rct_trt = float(s1_true[-1]) * 100.0
        rct_ate = rct_trt - rct_unt
        gf_unt_lo = float(s0_lower[-1]) * 100.0
        gf_unt_hi = float(s0_upper[-1]) * 100.0
        gf_trt_lo = float(s1_lower[-1]) * 100.0
        gf_trt_hi = float(s1_upper[-1]) * 100.0
        ate_lo = (float(s1_lower[-1]) - float(s0_upper[-1])) * 100.0
        ate_hi = (float(s1_upper[-1]) - float(s0_lower[-1])) * 100.0

        with open(txt, "w") as f:
            f.write("=" * 58 + "\n")
            f.write("G-FORMULA CORRECTED ANALYSIS RESULTS FOR PAPER\n")
            f.write(f"Generated on: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write("=" * 58 + "\n\n")
            f.write("STUDY CHARACTERISTICS:\n")
            f.write("- Analysis method: G-formula (parametric g-computation)\n")
            f.write("- Data source: Observational data with estimated parameters\n")
            f.write("- Uncertainty: Bootstrap 95% confidence intervals\n")
            f.write("- Comparison: RCT ground truth\n")
            f.write("- Study period: 168 hours\n\n")
            f.write("PRIMARY OUTCOMES (168 hours):\n")
            f.write(
                f"- RCT (Ground Truth): Untreated {rct_unt:.1f}%, "
                f"Treated {rct_trt:.1f}%, ATE {rct_ate:.1f}%\n"
            )
            f.write(
                f"- G-formula:          Untreated {gf_unt:.1f}% "
                f"[{gf_unt_lo:.1f}, {gf_unt_hi:.1f}], "
                f"Treated {gf_trt:.1f}% [{gf_trt_lo:.1f}, {gf_trt_hi:.1f}], "
                f"ATE {gf_ate:.1f}% [{ate_lo:.1f}, {ate_hi:.1f}]\n\n"
            )
            f.write("BIAS ASSESSMENT:\n")
            f.write(f"- Untreated survival bias: {gf_unt - rct_unt:+.1f}%\n")
            f.write(f"- Treated survival bias:   {gf_trt - rct_trt:+.1f}%\n")
            f.write(f"- ATE bias:                {gf_ate - rct_ate:+.1f}%\n\n")
            f.write("FIGURE GENERATION:\n")
            f.write("- Output file: Fig_gformula_corrected_survival_curves.pdf\n")
            f.write("- Format: PDF vector graphics (300 DPI)\n")
            f.write("- Dimensions: 7 x 5 inches\n")
        print(f"[a3c] Wrote {txt}")
    except Exception as exc:
        warnings.warn(f"[a3c] text summary failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
