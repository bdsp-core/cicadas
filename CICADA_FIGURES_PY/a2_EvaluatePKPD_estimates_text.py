"""
Port of CICADA_FIGURES/a2_EvaluatePKPD_estimates_text.m

Produces PKPD_estimation_summary.txt (no figures) -- statistical summary of
PKPD parameter estimation results.
"""
import os
import sys
import datetime
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'python'))

np.random.seed(0)

MAT_PATH = os.path.join(REPO_ROOT, 'PKPD_estimation_results.mat')
OUT_FILE = os.path.join(os.path.dirname(__file__), 'PKPD_estimation_summary.txt')


def _load_results():
    """Load results struct, supporting both classic and v7.3 MAT files."""
    try:
        from scipy.io import loadmat
        raw = loadmat(MAT_PATH, struct_as_record=False, squeeze_me=True)
        r = raw['results']

        def f(obj, name):
            return getattr(obj, name)

        return {
            'N': int(f(r, 'N')),
            't': np.asarray(f(r, 't')).flatten(),
            'true_params': np.asarray(f(r, 'true_params')).flatten(),
            'param_names': [str(x) for x in np.atleast_1d(f(r, 'param_names'))],
            'ke_true': float(f(r, 'ke_true')),
            'L_obs': np.asarray(f(r, 'L_obs')),
            'L_pred_trueL0': np.asarray(f(r, 'L_prediction').L_pred_trueL0),
            'mape_L_trueL0': np.asarray(f(r, 'L_prediction').mape_L_trueL0).flatten(),
            'mape_L_estL0': np.asarray(f(r, 'L_prediction').mape_L_estL0).flatten(),
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
            'twostage_corr_C_indiv': np.asarray(
                f(r, 'twostage_corr').patient_params.C_indiv).flatten(),
            'twostage_corr_g_indiv': np.asarray(
                f(r, 'twostage_corr').patient_params.g_indiv).flatten(),
            'C_true': np.asarray(f(r, 'C_true')).flatten(),
            'g_true': np.asarray(f(r, 'g_true')).flatten(),
            'patient_age': np.asarray(f(r, 'patient_age')).flatten(),
            'patient_sofa': np.asarray(f(r, 'patient_sofa')).flatten(),
        }
    except NotImplementedError:
        pass

    try:
        import h5py
    except ImportError:
        print("WARNING: h5py not installed; cannot read v7.3 MAT file. Exiting.")
        sys.exit(0)

    with h5py.File(MAT_PATH, 'r') as fh:
        R = fh['results']

        def g(path):
            return np.asarray(R[path][()])

        def gT(path):
            a = np.asarray(R[path][()])
            return a.T if a.ndim == 2 else a

        param_names = []
        for ref in R['param_names'][:, 0]:
            obj = fh[ref][()]
            s = ''.join(chr(int(c)) for c in np.asarray(obj).flatten())
            param_names.append(s)

        return {
            'N': int(g('N').flatten()[0]),
            't': g('t').flatten(),
            'true_params': g('true_params').flatten(),
            'param_names': param_names,
            'ke_true': float(g('ke_true').flatten()[0]),
            'L_obs': gT('L_obs'),
            'L_pred_trueL0': gT('L_prediction/L_pred_trueL0'),
            'mape_L_trueL0': g('L_prediction/mape_L_trueL0').flatten(),
            'mape_L_estL0': g('L_prediction/mape_L_estL0').flatten(),
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
            'twostage_corr_C_indiv': g('twostage_corr/patient_params/C_indiv').flatten(),
            'twostage_corr_g_indiv': g('twostage_corr/patient_params/g_indiv').flatten(),
            'C_true': g('C_true').flatten(),
            'g_true': g('g_true').flatten(),
            'patient_age': g('patient_age').flatten(),
            'patient_sofa': g('patient_sofa').flatten(),
        }


# ------------------- main -------------------
print("==========================================================")
print("a1_EvaluatePKPD_estimates: STATISTICAL SUMMARY (Python port)")
print("==========================================================")

if not os.path.exists(MAT_PATH):
    print(f"WARNING: {MAT_PATH} not found. Run the PKPD estimation first. Exiting.")
    sys.exit(0)

print("Loading PKPD_estimation_results.mat ...")
res = _load_results()
print("Loaded.\n")

N = res['N']
t = res['t']
L_obs = res['L_obs']
L_pred_trueL0 = res['L_pred_trueL0']
mape_L_trueL0 = res['mape_L_trueL0']
param_names = res['param_names']

# --- L prediction accuracy (stack every 5th observed point) ---
all_obs = []
all_pred = []
for p_idx in range(N):
    if p_idx >= L_obs.shape[0]:
        continue
    obs_row = L_obs[p_idx, :]
    pred_row = L_pred_trueL0[p_idx, :]
    valid = ~np.isnan(obs_row)
    idxs = np.where(valid)[0][::5]
    if idxs.size > 0:
        all_obs.append(obs_row[idxs])
        all_pred.append(pred_row[idxs])

if all_obs:
    all_obs = np.concatenate(all_obs)
    all_pred = np.concatenate(all_pred)
    # matching MATLAB corr(all_obs', all_pred')^2
    R2_L_pred = np.corrcoef(all_obs, all_pred)[0, 1] ** 2
else:
    R2_L_pred = np.nan

valid_mape = mape_L_trueL0[mape_L_trueL0 < 200]
pctiles = np.percentile(valid_mape, [25, 50, 75]) if valid_mape.size else (np.nan,) * 3

# --- Population-model R^2 for C, g ---
C_twostage = res['twostage_corr_C_indiv']
g_twostage = res['twostage_corr_g_indiv']
age = res['patient_age']
sofa = res['patient_sofa']
age_norm = (age - age.mean()) / age.std(ddof=1)
sofa_norm = (sofa - sofa.mean()) / sofa.std(ddof=1)
theta = res['twostage_corr_theta']
C_pred = theta[0] + theta[1] * age_norm + theta[2] * sofa_norm
g_pred = theta[3] + theta[4] * age_norm + theta[5] * sofa_norm
SS_tot_C = np.sum((C_twostage - C_twostage.mean()) ** 2)
SS_res_C = np.sum((C_twostage - C_pred) ** 2)
R2_C = 1 - SS_res_C / SS_tot_C if SS_tot_C > 0 else np.nan
SS_tot_g = np.sum((g_twostage - g_twostage.mean()) ** 2)
SS_res_g = np.sum((g_twostage - g_pred) ** 2)
R2_g = 1 - SS_res_g / SS_tot_g if SS_tot_g > 0 else np.nan

improvement_fold = res['joint_mape'] / res['twostage_corr_mape'] if res['twostage_corr_mape'] else np.nan

# --- Text ---
L = []
A = L.append
A("==========================================================")
A("PKPD PARAMETER ESTIMATION: STATISTICAL SUMMARY")
A("==========================================================\n")
A(f"Generated: {datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')}")
A("Source data: PKPD_estimation_results.mat")
A("Analysis script: a2_EvaluatePKPD_estimates_text.py (Python port)\n")
A("DATA CHARACTERISTICS:")
A(f"  Number of patients: {N}")
A(f"  Time points per patient: {len(t)}")
A(f"  Time range: {int(t[0])} to {int(t[-1])} hours")
A(f"  Time step: {int(t[1] - t[0])} hours\n")

A("METHOD COMPARISON:")
A("----------------------------------------------------------")
A("                     Joint    Oracle   2-Stage  2-Stage")
A("                                       (Raw)    (Corr)")
A("----------------------------------------------------------")
A(f"MAPE (%):           {res['joint_mape']:6.1f}   {res['oracle_mape']:6.1f}   "
  f"{res['twostage_raw_mape']:6.1f}   {res['twostage_corr_mape']:6.1f}")
A(f"ke Error (%):       {res['joint_errors'][6]:6.1f}   {0.0:6.1f}   "
  f"{res['twostage_raw_errors'][6]:6.1f}   {res['twostage_corr_errors'][6]:6.1f}")
A(f"L0 Correlation:     {res['joint_L0_corr']:6.3f}   {res['oracle_L0_corr']:6.3f}   "
  f"{res['twostage_raw_L0_corr']:6.3f}   {res['twostage_corr_L0_corr']:6.3f}")
A(f"Time (seconds):     {res['joint_time']:6.1f}   {res['oracle_time']:6.1f}   "
  f"{res['twostage_raw_time']:6.1f}   {res['twostage_corr_time']:6.1f}")
A("----------------------------------------------------------\n")

A("BEST METHOD: Two-Stage with Bias Correction")
A(f"  - MAPE: {res['twostage_corr_mape']:.1f}% (vs {res['joint_mape']:.1f}% for joint estimation)")
A(f"  - Improvement: {improvement_fold:.0f}-fold")
A(f"  - ke estimation error: {res['twostage_corr_errors'][6]:.1f}% (after correction)\n")

A("INDIVIDUAL PARAMETER ERRORS (Two-Stage Corrected):")
for i in range(6):
    A(f"  {param_names[i]}: {res['twostage_corr_errors'][i]:.2f}%")
A("")

A("L PREDICTION ACCURACY:")
A(f"  Mean MAPE:        {np.mean(valid_mape):.1f}%")
A(f"  Median MAPE:      {np.median(valid_mape):.1f}%")
A(f"  25th percentile:  {pctiles[0]:.1f}%")
A(f"  75th percentile:  {pctiles[2]:.1f}%")
A(f"  R\u00b2 (obs vs pred): {R2_L_pred:.3f}\n")

mape_true = res['mape_L_trueL0']
mape_est = res['mape_L_estL0']
A("L PREDICTION WITH DIFFERENT L0 SOURCES:")
A(f"  With true L0:      {np.mean(mape_true[mape_true < 100]):.1f}% MAPE")
A(f"  With estimated L0: {np.mean(mape_est[mape_est < 100]):.1f}% MAPE\n")

A("POPULATION MODEL QUALITY (Two-Stage Corrected):")
A(f"  R\u00b2 for C model: {R2_C:.3f}")
A(f"  R\u00b2 for g model: {R2_g:.3f}\n")

A("INDIVIDUAL PARAMETER CORRELATIONS (True vs Estimated):")
A(f"  C correlation: {np.corrcoef(res['C_true'], C_twostage)[0, 1]:.3f}")
A(f"  g correlation: {np.corrcoef(res['g_true'], g_twostage)[0, 1]:.3f}\n")

A("ke ESTIMATION DETAILS:")
A(f"  True value:           {res['ke_true']:.3f}")
A(f"  Joint estimate:       {res['joint_theta'][6]:.3f} (error: {res['joint_errors'][6]:.1f}%)")
A(f"  Two-stage raw:        {res['twostage_raw_ke_raw']:.3f} (error: {res['twostage_raw_errors'][6]:.1f}%)")
A(f"  Two-stage corrected:  {res['twostage_corr_ke_corrected']:.3f} (error: {res['twostage_corr_errors'][6]:.1f}%)")
A("  Correction factor:    1.41\n")

A("L0 RECOVERY:")
A(f"  Correlation (Two-Stage): {res['twostage_corr_L0_corr']:.3f}")
A("  Note: Moderate correlation is expected due to high")
A("        disease suppression by treatment. This does not")
A("        significantly affect L prediction accuracy.\n")

A("KEY FINDINGS:")
A(f"  1. Two-stage approach with bias correction achieves {improvement_fold:.0f}-fold")
A("     improvement over joint estimation")
A("  2. Bias correction factor of 1.41 effectively compensates")
A("     for systematic underestimation in ke")
A("  3. Population models (C and g) achieve excellent fit")
A(f"     with R\u00b2 > {min(R2_C, R2_g):.2f} for both parameters")
A("  4. L(t) prediction accuracy is clinically acceptable")
A(f"     with median MAPE of {np.median(valid_mape):.1f}%")
A("  5. Method is computationally efficient, requiring only")
A(f"     {res['twostage_corr_time']:.1f} seconds for {N} patients\n")

A("CLINICAL IMPLICATIONS:")
A("  - Parameter estimation is sufficiently accurate for")
A("    clinical decision support")
A("  - Two-stage approach recommended when ke can be")
A("    estimated separately or is known a priori")
A("  - Robust to moderate uncertainty in ke (\u00b110%)\n")

A("==========================================================")
A("END OF SUMMARY")
A("==========================================================")

summary_text = "\n".join(L) + "\n"

with open(OUT_FILE, 'w') as f:
    f.write(summary_text)

print(summary_text)
print(f"\nSummary saved to: {OUT_FILE}")
print("==========================================================")
