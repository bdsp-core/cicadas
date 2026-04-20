function [C_est, g_est, ke_est, parmsPD_est] = fcnGetPKPD_parms_est(patient_age, patient_sofa)

% This applies the mixed effects model:
% - C_i = b0_C + b1_C × age_norm_i + b2_C × sofa_norm_i
% - g_i = b0_g + b1_g × age_norm_i + b2_g × sofa_norm_i

load('PKPD_estimation_results.mat', 'results');
ke_est = results.twostage_corr.ke_corrected;

% C_est = results.twostage_corr.patient_params.C_indiv;
% g_est = results.twostage_corr.patient_params.g_indiv;

parmsPD_est = results.twostage_corr.theta;  % [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g];

% Make sure parmsPD_est only has 6 elements (no ke)
if length(parmsPD_est) > 6
    parmsPD_est = parmsPD_est(1:6);
end

% Normalize age and sofa (important - must match how it was done during estimation)
age_norm = (patient_age - mean(patient_age)) / std(patient_age);
sofa_norm = (patient_sofa - mean(patient_sofa)) / std(patient_sofa);

% Calculate individual C and g values using the population model
C_est = parmsPD_est(1) + parmsPD_est(2)*age_norm + parmsPD_est(3)*sofa_norm;
g_est = parmsPD_est(4) + parmsPD_est(5)*age_norm + parmsPD_est(6)*sofa_norm;

% Create theta_est for fcnDiseaseModelDiagnostics
% theta_est contains: [parmsL_est(:); b0_C; b1_C; b2_C; b0_g; b1_g; b2_g; ke]
% Note: parmsL_est might have 8 parameters while parmsL has 7, so we only take first 7

