"""
Port of CICADA_FIGURES/a2_EvaluatePKPD_estimates_figures.m

Produces Fig_Combined_PKPD_Analysis.pdf: a 4-panel (A-D) PKPD parameter-
recovery and L-prediction figure.

Input: PKPD_estimation_results.mat (v7.3 HDF5) at repo root.
"""
import os
import sys
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'python'))

np.random.seed(0)

MAT_PATH = os.path.join(REPO_ROOT, 'PKPD_estimation_results.mat')
OUT_DIR = os.path.dirname(__file__)
FIG_PDF = os.path.join(OUT_DIR, 'Fig_Combined_PKPD_Analysis.pdf')
SUMMARY_TXT = os.path.join(OUT_DIR, 'PKPD_combined_figure_summary.txt')


def _load_results():
    """Load results struct from PKPD_estimation_results.mat.

    Tries scipy.io first; falls back to h5py for MATLAB v7.3 files.
    Returns a plain dict with the fields used downstream.
    """
    try:
        from scipy.io import loadmat
        raw = loadmat(MAT_PATH, struct_as_record=False, squeeze_me=True)
        r = raw['results']

        def f(obj, name):
            return getattr(obj, name)

        out = {
            'N': int(f(r, 'N')),
            't': np.asarray(f(r, 't')).flatten(),
            'true_params': np.asarray(f(r, 'true_params')).flatten(),
            'param_names': [str(x) for x in np.atleast_1d(f(r, 'param_names'))],
            'ke_true': float(f(r, 'ke_true')),
            'L_obs': np.asarray(f(r, 'L_obs')),
            'A_obs': np.asarray(f(r, 'A_obs')),
            'L0_true': np.asarray(f(r, 'L0_true')).flatten(),
            'L_pred_trueL0': np.asarray(f(r, 'L_prediction').L_pred_trueL0),
            'mape_L_trueL0': np.asarray(f(r, 'L_prediction').mape_L_trueL0).flatten(),
            'joint_errors': np.asarray(f(r, 'joint').errors).flatten(),
            'joint_mape': float(f(r, 'joint').mape),
            'joint_theta': np.asarray(f(r, 'joint').theta).flatten(),
            'joint_time': float(f(r, 'joint').time),
            'joint_L0_corr': float(f(r, 'joint').L0_corr),
            'oracle_errors': np.asarray(f(r, 'oracle').errors).flatten(),
            'oracle_mape': float(f(r, 'oracle').mape),
            'oracle_L0_corr': float(f(r, 'oracle').L0_corr),
            'oracle_time': float(f(r, 'oracle').time),
            'twostage_raw_errors': np.asarray(f(r, 'twostage_raw').errors).flatten(),
            'twostage_raw_mape': float(f(r, 'twostage_raw').mape),
            'twostage_raw_L0_corr': float(f(r, 'twostage_raw').L0_corr),
            'twostage_raw_time': float(f(r, 'twostage_raw').time),
            'twostage_raw_ke_raw': float(f(r, 'twostage_raw').ke_raw),
            'twostage_corr_errors': np.asarray(f(r, 'twostage_corr').errors).flatten(),
            'twostage_corr_mape': float(f(r, 'twostage_corr').mape),
            'twostage_corr_L0_corr': float(f(r, 'twostage_corr').L0_corr),
            'twostage_corr_time': float(f(r, 'twostage_corr').time),
            'twostage_corr_ke_corrected': float(f(r, 'twostage_corr').ke_corrected),
            'twostage_corr_theta': np.asarray(f(r, 'twostage_corr').theta).flatten(),
            'twostage_corr_patient_params_C': np.asarray(
                f(r, 'twostage_corr').patient_params.C_indiv).flatten(),
            'twostage_corr_patient_params_g': np.asarray(
                f(r, 'twostage_corr').patient_params.g_indiv).flatten(),
            'C_true': np.asarray(f(r, 'C_true')).flatten(),
            'g_true': np.asarray(f(r, 'g_true')).flatten(),
            'patient_age': np.asarray(f(r, 'patient_age')).flatten(),
            'patient_sofa': np.asarray(f(r, 'patient_sofa')).flatten(),
        }
        return out
    except NotImplementedError:
        pass  # fall through to h5py

    try:
        import h5py
    except ImportError:
        print("WARNING: h5py not installed; cannot read v7.3 MAT file. Exiting.")
        sys.exit(0)

    with h5py.File(MAT_PATH, 'r') as fh:
        R = fh['results']

        def g(path):
            return np.asarray(R[path][()])

        param_names = []
        for ref in R['param_names'][:, 0]:
            obj = fh[ref][()]
            s = ''.join(chr(int(c)) for c in np.asarray(obj).flatten())
            param_names.append(s)

        # HDF5 stores 2-D MATLAB arrays transposed relative to scipy.io; undo it
        # to match (N_patients x T) orientation.
        def gT(path):
            a = np.asarray(R[path][()])
            return a.T if a.ndim == 2 else a

        out = {
            'N': int(g('N').flatten()[0]),
            't': g('t').flatten(),
            'true_params': g('true_params').flatten(),
            'param_names': param_names,
            'ke_true': float(g('ke_true').flatten()[0]),
            'L_obs': gT('L_obs'),
            'A_obs': gT('A_obs'),
            'L0_true': gT('L0_true').flatten(),
            'L_pred_trueL0': gT('L_prediction/L_pred_trueL0'),
            'mape_L_trueL0': g('L_prediction/mape_L_trueL0').flatten(),
            'joint_errors': g('joint/errors').flatten(),
            'joint_mape': float(g('joint/mape').flatten()[0]),
            'joint_theta': g('joint/theta').flatten(),
            'joint_time': float(g('joint/time').flatten()[0]),
            'joint_L0_corr': float(g('joint/L0_corr').flatten()[0]),
            'oracle_errors': g('oracle/errors').flatten(),
            'oracle_mape': float(g('oracle/mape').flatten()[0]),
            'oracle_L0_corr': float(g('oracle/L0_corr').flatten()[0]),
            'oracle_time': float(g('oracle/time').flatten()[0]),
            'twostage_raw_errors': g('twostage_raw/errors').flatten(),
            'twostage_raw_mape': float(g('twostage_raw/mape').flatten()[0]),
            'twostage_raw_L0_corr': float(g('twostage_raw/L0_corr').flatten()[0]),
            'twostage_raw_time': float(g('twostage_raw/time').flatten()[0]),
            'twostage_raw_ke_raw': float(g('twostage_raw/ke_raw').flatten()[0]),
            'twostage_corr_errors': g('twostage_corr/errors').flatten(),
            'twostage_corr_mape': float(g('twostage_corr/mape').flatten()[0]),
            'twostage_corr_L0_corr': float(g('twostage_corr/L0_corr').flatten()[0]),
            'twostage_corr_time': float(g('twostage_corr/time').flatten()[0]),
            'twostage_corr_ke_corrected': float(
                g('twostage_corr/ke_corrected').flatten()[0]),
            'twostage_corr_theta': g('twostage_corr/theta').flatten(),
            'twostage_corr_patient_params_C': g(
                'twostage_corr/patient_params/C_indiv').flatten(),
            'twostage_corr_patient_params_g': g(
                'twostage_corr/patient_params/g_indiv').flatten(),
            'C_true': g('C_true').flatten(),
            'g_true': g('g_true').flatten(),
            'patient_age': g('patient_age').flatten(),
            'patient_sofa': g('patient_sofa').flatten(),
        }
        return out


# ------------------- load -------------------
if not os.path.exists(MAT_PATH):
    print(f"WARNING: {MAT_PATH} not found. Run the PKPD estimation first. Exiting.")
    sys.exit(0)

print("Loading PKPD_estimation_results.mat ...")
res = _load_results()
print("Loaded.")

N = res['N']
t = res['t']
L_obs = res['L_obs']
L_pred_trueL0 = res['L_pred_trueL0']

# ------------------- per-patient MAE for L ---------------
mae_L_trueL0 = np.full(N, np.nan)
for p_idx in range(N):
    if p_idx >= L_obs.shape[0] or p_idx >= L_pred_trueL0.shape[0]:
        continue
    obs = L_obs[p_idx, :]
    pred = L_pred_trueL0[p_idx, :]
    valid = ~np.isnan(obs)
    if np.sum(valid) > 0:
        mae_L_trueL0[p_idx] = np.mean(np.abs(obs[valid] - pred[valid]))

# ------------------- figure -------------------
fig = plt.figure(figsize=(12, 9), facecolor='white')

# Panel positions (mirror the MATLAB layout)
sw, sh = 0.42, 0.38
lm, rm = 0.08, 0.05
bm, tm = 0.08, 0.12
hs, vs = 0.08, 0.08
pos = {
    'A': [lm, bm + sh + vs, sw, sh],
    'B': [lm + sw + hs, bm + sh + vs, sw, sh],
    'C': [lm, bm, sw, sh],
    'D': [lm + sw + hs, bm, sw, sh],
}

# =================== PANEL A: parameter errors =====================
ax_A = fig.add_axes(pos['A'])
param_display = [r'$\beta_0^C$', r'$\beta_1^C$', r'$\beta_2^C$',
                 r'$\beta_0^\gamma$', r'$\beta_1^\gamma$', r'$\beta_2^\gamma$',
                 r'$k_e$']

errs_raw = np.column_stack([
    res['joint_errors'][:7],
    res['oracle_errors'][:7],
    res['twostage_raw_errors'][:7],
    res['twostage_corr_errors'][:7],
])
errs_raw[6, 1] = 0.0  # oracle has 0 error for ke by definition
y_max = 5.0
errs = np.clip(errs_raw, None, y_max)

method_colors = [(0.8, 0.2, 0.2), (0.2, 0.6, 0.2),
                 (0.4, 0.4, 0.8), (0.2, 0.4, 0.8)]
method_labels = ['Joint', 'Oracle (fixed ke)', '2-Stage (Raw)', '2-Stage (Corrected)']

n_params = 7
n_methods = 4
bar_w = 0.2
x_idx = np.arange(n_params)
for m in range(n_methods):
    offset = (m - (n_methods - 1) / 2.0) * bar_w
    ax_A.bar(x_idx + offset, errs[:, m], bar_w,
             color=method_colors[m], edgecolor='k', linewidth=0.8,
             label=method_labels[m])
    # truncation markers for capped bars
    for p_idx in range(n_params):
        if errs_raw[p_idx, m] > y_max:
            x_pos = p_idx + offset
            ax_A.plot([x_pos - 0.04, x_pos + 0.04],
                      [y_max - 0.15, y_max - 0.15], 'k-', linewidth=1.5)
            ax_A.text(x_pos, y_max + 0.2,
                      f'{errs_raw[p_idx, m]:.0f}%',
                      ha='center', fontsize=9, fontweight='bold',
                      color=method_colors[m])

ax_A.set_xticks(x_idx)
ax_A.set_xticklabels(param_display, fontsize=11, fontweight='bold')
ax_A.set_ylabel('Absolute Error (%)', fontsize=11, fontweight='bold')
ax_A.set_xlabel('Parameter', fontsize=11, fontweight='bold')
ax_A.set_ylim(0, 5.9)
ax_A.spines['top'].set_visible(False)
ax_A.spines['right'].set_visible(False)
ax_A.tick_params(direction='out', width=1.2)
ax_A.yaxis.grid(True, alpha=0.3)
ax_A.set_axisbelow(True)
ax_A.text(0.02, 0.98, 'A. PKPD Parameter Estimation Errors',
          transform=ax_A.transAxes, fontsize=12, fontweight='bold', va='top',
          bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
leg = ax_A.legend(loc='upper center', bbox_to_anchor=(0.5, 0.82),
                  fontsize=9, frameon=True)
leg.get_frame().set_edgecolor('none')
leg.get_frame().set_facecolor('white')

# =================== PANEL B: MAE distribution =====================
ax_B = fig.add_axes(pos['B'])
valid_mae = mae_L_trueL0[~np.isnan(mae_L_trueL0)]
ax_B.hist(valid_mae, bins=30, color=(0.2, 0.4, 0.8),
          edgecolor='k', linewidth=0.5, alpha=0.8)
ax_B.set_xlabel('MAE', fontsize=11, fontweight='bold')
ax_B.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
ax_B.set_ylim(0, 225)
ax_B.spines['top'].set_visible(False)
ax_B.spines['right'].set_visible(False)
ax_B.tick_params(direction='out', width=1.2)
ax_B.grid(True, alpha=0.3)
ax_B.set_axisbelow(True)
ax_B.text(0.02, 0.95, 'B. Prediction Error Distribution',
          transform=ax_B.transAxes, fontsize=12, fontweight='bold', va='top',
          bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
ax_B.text(0.6, 0.85,
          f'Mean: {np.mean(valid_mae):.3f}\nMedian: {np.median(valid_mae):.3f}',
          transform=ax_B.transAxes, fontsize=10, fontweight='bold',
          bbox=dict(facecolor='white', edgecolor='none'))

# =================== PANELS C & D: example patients =====================
def _plot_patient(ax, patient_idx, letter):
    if patient_idx >= L_obs.shape[0]:
        ax.text(0.5, 0.5, f"(patient {patient_idx} out of range)",
                transform=ax.transAxes, ha='center')
        return
    obs = L_obs[patient_idx, :]
    pred = L_pred_trueL0[patient_idx, :]
    valid = ~np.isnan(obs)
    ax.plot(t[valid], obs[valid], 'o', markersize=5,
            markerfacecolor=(0.3, 0.3, 0.3), markeredgecolor='k',
            linewidth=0.5, label='Observed')
    ax.plot(t, pred, '-', color=(0.8, 0.2, 0.2), linewidth=2.5, label='Predicted')
    ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'$L_t$', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.32)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', width=1.2)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    pmae = mae_L_trueL0[patient_idx]
    ax.text(0.02, 0.95,
            f'{letter}. Patient {patient_idx + 1} (MAE: {pmae:.3f})',
            transform=ax.transAxes, fontsize=12, fontweight='bold', va='top',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax.legend(loc='best', fontsize=10, frameon=False)


ax_C = fig.add_axes(pos['C'])
_plot_patient(ax_C, 99, 'C')  # MATLAB patient_idx=100 -> 0-based 99

ax_D = fig.add_axes(pos['D'])
_plot_patient(ax_D, 19, 'D')  # MATLAB patient_idx=20 -> 0-based 19

# Overall title
fig.text(0.5, 0.955, 'PKPD Parameter Estimation Evaluation',
         ha='center', va='center', fontsize=16, fontweight='bold')

fig.savefig(FIG_PDF, bbox_inches='tight', dpi=300)
# Also save a PNG alongside (MATLAB script does too)
fig.savefig(FIG_PDF.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved: {FIG_PDF}")
print(f"Saved: {FIG_PDF.replace('.pdf', '.png')}")

# =================== summary txt =====================
lines = []
lines.append("==========================================================")
lines.append("PKPD PUBLICATION FIGURES: SUMMARY OUTPUT")
lines.append("==========================================================\n")
lines.append(f"Generated: {datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')}")
lines.append("Source script: Combined PKPD Analysis Figure (Python port)")
lines.append("Source data: PKPD_estimation_results.mat\n")
lines.append("FIGURES GENERATED:")
lines.append("  - Fig_Combined_PKPD_Analysis.pdf")
lines.append("  - Fig_Combined_PKPD_Analysis.png\n")
lines.append("KEY RESULTS:")
lines.append("  Best Method: Two-Stage with Bias Correction")
lines.append(f"  - MAE: {res['twostage_corr_mape']:.1f}% (vs {res['joint_mape']:.1f}% for joint estimation)")
fold = res['joint_mape'] / res['twostage_corr_mape'] if res['twostage_corr_mape'] else np.nan
lines.append(f"  - Improvement: {fold:.0f}-fold")
lines.append(f"  - ke estimation error: {res['twostage_corr_errors'][6]:.1f}% (after correction)")
lines.append("\nL PREDICTION ACCURACY:")
lines.append(f"  - With true L0: {np.nanmean(mae_L_trueL0):.3f} MAE")
lines.append("\nL0 RECOVERY:")
lines.append(f"  - Correlation: {res['twostage_corr_L0_corr']:.3f} (moderate due to high suppression)")
lines.append("  - This is expected and does not affect L prediction")
lines.append("\nFigures saved as PDF and PNG files.")
lines.append("==========================================================")

summary = "\n".join(lines) + "\n"
with open(SUMMARY_TXT, 'w') as f:
    f.write(summary)
print(f"Summary saved to: {SUMMARY_TXT}")
