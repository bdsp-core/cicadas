"""
Port of CICADA_FIGURES/a1_SingleTraces.m

Produces Fig1_singleTrajectories_3panels.pdf: 3-panel trajectory plot showing
an untreated patient, a dose-changing patient, and a closed-loop-control patient.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make existing Python helpers importable (mirror of MATLAB addpath('..'))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'python'))

np.random.seed(0)

# -------------------- inputs --------------------
csv1 = os.path.join(REPO_ROOT, 'trialData1.csv')
csv_dose = os.path.join(REPO_ROOT, 'trialDataDoseChanging.csv')

missing = [p for p in (csv1, csv_dose) if not os.path.exists(p)]
if missing:
    print(f"WARNING: required input files not found: {missing}. Exiting gracefully.")
    sys.exit(0)

T = pd.read_csv(csv1)
T_dose = pd.read_csv(csv_dose)

# --- Patient selection (same exemplar IDs as MATLAB) ----------------------
sid1 = np.sort(T.loc[T.Rx == 1, 'sid'].unique())
sid0 = np.sort(T.loc[T.Rx == 0, 'sid'].unique())
sid_dose_all = np.sort(T_dose['sid'].unique())

# MATLAB uses 1-based indices idx1=1, idx0=9, idx_dose=58
idx1 = 1 - 1
idx0 = 9 - 1
idx_dose = 58 - 1

# Guard against out-of-range (simulations with fewer patients)
idx1 = min(idx1, len(sid1) - 1) if len(sid1) else None
idx0 = min(idx0, len(sid0) - 1) if len(sid0) else None
idx_dose = min(idx_dose, len(sid_dose_all) - 1) if len(sid_dose_all) else None

# -------------------- styling --------------------
axesFS = 11
lblFS = 11
ttlFS = 12
lgdFS = 10
LW_L = 3.0
LW_A = 1.5
YL_LIM = (0.0, 1.25)
YR_LIM = (-0.1, 5.0)
XLIM = (0.0, 168.0)

fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True)

panel_specs = [
    ("Untreated", (0.2, 0.2, 0.8), T, sid0, idx0, True),
    ("Random Dose Switching", (0.2, 0.6, 0.2), T_dose, sid_dose_all, idx_dose, False),
    ("Closed-Loop Control", (0.8, 0.2, 0.2), T, sid1, idx1, False),
]

for p, (ttl, C, df, sid_list, idx, mark_death) in enumerate(panel_specs):
    ax = axes[p]
    if idx is None:
        ax.text(0.5, 0.5, f"(no data for panel: {ttl})",
                transform=ax.transAxes, ha='center', va='center')
        continue

    pd_data = df[df['sid'] == sid_list[idx]].sort_values('t')
    t_pat = pd_data['t'].to_numpy()
    L_pat = pd_data['L'].to_numpy()
    A_pat = pd_data['A'].to_numpy()

    # Left axis: L(t)
    line_L, = ax.plot(t_pat, L_pat, '-', color=C, linewidth=LW_L, label=r'$L_t$')
    ax.set_ylim(*YL_LIM)
    ax.tick_params(axis='y', colors=C)
    for spine in ('left',):
        ax.spines[spine].set_color(C)
    ax.set_xlim(*XLIM)

    # Black "death" star on untreated panel
    if mark_death and len(t_pat) > 0:
        ax.plot(t_pat[-1], L_pat[-1], marker='*', color='k',
                markersize=14, markerfacecolor='k', linestyle='None')

    # Right axis: A(t) extended across 0-168h with "previous" interpolation
    ax_r = ax.twinx()
    t_full = np.arange(XLIM[0], XLIM[1] + 1, 1.0)
    # "previous" interpolation: at time t_full[i], take A at largest t_pat<=t_full[i], else 0
    if len(t_pat) > 0:
        order = np.argsort(t_pat)
        tp = t_pat[order]
        Ap = A_pat[order]
        idxs = np.searchsorted(tp, t_full, side='right') - 1
        A_full = np.where(idxs >= 0, Ap[np.clip(idxs, 0, len(Ap) - 1)], 0.0)
    else:
        A_full = np.zeros_like(t_full)
    line_A, = ax_r.plot(t_full, A_full, 'k--', linewidth=LW_A, label=r'$A_t$')
    ax_r.set_ylim(*YR_LIM)
    ax_r.tick_params(axis='y', colors='k')

    # Titles inside the plot (upper-left)
    ax.text(0.02, 0.95, ttl, transform=ax.transAxes,
            fontsize=ttlFS, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))

    # Axis labels only on bottom panel
    if p == 2:
        ax.set_xlabel('Time [hours]', fontsize=lblFS)
        ax.set_ylabel(r'$L_t$', fontsize=lblFS, color=C)
        ax_r.set_ylabel(r'$A_t$', fontsize=lblFS, color='k')

    ax.tick_params(axis='both', labelsize=axesFS)
    ax_r.tick_params(axis='both', labelsize=axesFS)

    # Combined legend
    ax.legend([line_L, line_A], [r'$L_t$', r'$A_t$'],
              fontsize=lgdFS, frameon=False, loc='best')

    ax.grid(True, alpha=0.3)
    for spine_name in ('top', 'right'):
        ax.spines[spine_name].set_visible(False)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__),
                        'Fig1_singleTrajectories_3panels.pdf')
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")
