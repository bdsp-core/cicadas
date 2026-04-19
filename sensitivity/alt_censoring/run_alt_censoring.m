%% Alternative-censoring sensitivity
%
% Reviewer 1 asked whether CICADAS results are robust to alternative
% censoring mechanisms. The baseline DGP has treatment-dependent informative
% censoring (baseline V parameters in a0_GenerateTrialData.m). Here we
% regenerate observational data under three alternative censoring models:
%
%   (1) baseline: parmsV as in a0_GenerateTrialData.m
%   (2) MCAR:     censoring completely at random (uniform censoring rate)
%   (3) strong:   strongly outcome-dependent censoring
%   (4) none:     no censoring at all (a stress-test)
%
% For each, we run the CICADAS g-formula pipeline and report recovered
% ATE vs the RCT truth.

clear; clc; format compact;
script_dir = fileparts(mfilename('fullpath'));
repo_root  = fileparts(fileparts(script_dir));
addpath(repo_root);
addpath(genpath(fullfile(repo_root, 'CICADA_FIGURES')));

%% Shared baseline params
rng(0);
N = 1000;
th = 0.1;
parmsControl = [10, 50];
[age, sofa, C, g, parmsPD] = fcnGeneratePatientParameters(N, ...
    'TargetCMean', 3, 'TargetGMean', 4, 'CV', 0.1);
ke = 0.5;
parmsY = [-7, .3, 20, 5];
parmsL = [0.25, 1, 0.15, 0.05, 0.15, 0.03, 40];
dt = 2; t = 0:dt:168;

% Censoring regimes: parmsV = [b0 b1 b2 b3 b4 b5]
%   logit_v = Rx*(b0 + b1*cumA + b2*t^2) + (1-Rx)*(b3 + b4*cumL + b5*t^2)
regimes = struct();
regimes.baseline = [-5,  2.0, 0.1, -5,  2.0, 1.5];
regimes.MCAR     = [-6,  0.0, 0.0, -6,  0.0, 0.0];
regimes.strong   = [-3,  4.0, 0.5, -3,  4.0, 2.5];
regimes.none     = [-50, 0.0, 0.0, -50, 0.0, 0.0];
names = fieldnames(regimes);

% RCT truth (no censoring active in RCT mode)
L0_rct = fcnGenerateStochasticTrajectories(t, parmsL, N);
T1 = fcnSimulate_N_Patients(N, 1, 0.5*ones(1,N), th, C, g, ke, L0_rct, ...
                             parmsControl, parmsY, regimes.baseline, age, sofa);
[s0_true, s1_true] = fcnPlotKM(T1); close all;
ate_truth = s1_true(end) - s0_true(end);
fprintf('RCT truth ATE = %.3f\n', ate_truth);

results = struct();
for i = 1:length(names)
    nm = names{i};
    parmsV_i = regimes.(nm);
    rng(1 + i);

    L0 = fcnGenerateStochasticTrajectories(t, parmsL, N);
    treatProb = fcnBiasedAssignmentProb(age, sofa, L0(:, 1:5));
    T0 = fcnSimulate_N_Patients(N, 0, treatProb, th, C, g, ke, L0, ...
                                parmsControl, parmsY, parmsV_i, age, sofa);

    n_cens = sum(T0.V > 0);
    total_rows = height(T0);
    fprintf('\nRegime "%s": censoring rate = %.1f%%\n', nm, 100*n_cens/total_rows);

    try
        parmsY_est = fcnEstimateDeathParms(T0);
        [parmsL_est, LL, AA, age_e, sofa_e, t_e] = fcnEstimateParmsL(T0);
        [ke_est, C_est, g_est] = fcnEstimateParmsPKPD(parmsL_est, LL, AA, age_e, sofa_e, t_e);
        L0_est = fcnGenerateStochasticTrajectories(t, parmsL_est, N);
        T1_est = fcnSimulate_N_Patients(N, 1, 0.5*ones(1,N), th, C_est, g_est, ke_est, ...
                                         L0_est, parmsControl, parmsY_est, zeros(1,6), age, sofa);
        [s0_gf, s1_gf] = fcnPlotKM(T1_est); close all;
        ate = s1_gf(end) - s0_gf(end);
        bias = ate - ate_truth;
    catch ME
        fprintf('  [WARN] %s\n', ME.message);
        ate = NaN; bias = NaN;
    end
    results.(nm) = struct('ate', ate, 'bias', bias, 'cens_rate', n_cens/total_rows);
    fprintf('  ATE = %+.3f (bias %+.3f)\n', ate, bias);
end

save(fullfile(script_dir, 'alt_censoring_results.mat'), 'results','ate_truth','names','regimes');
fprintf('\nSaved results.\n');

%% Figure
figure(1); clf;
atess = zeros(1, length(names));
biases = zeros(1, length(names));
cens_rates = zeros(1, length(names));
for i = 1:length(names)
    atess(i) = 100 * results.(names{i}).ate;
    biases(i) = 100 * results.(names{i}).bias;
    cens_rates(i) = 100 * results.(names{i}).cens_rate;
end
subplot(1,2,1);
bar(atess); set(gca, 'XTickLabel', names); ylabel('G-formula ATE (pp)');
yline(100*ate_truth, 'r--', 'LineWidth', 1.5, 'Label', sprintf('RCT truth = %.1f', 100*ate_truth));
title('Recovered ATE across censoring regimes');
subplot(1,2,2);
bar(cens_rates); set(gca, 'XTickLabel', names); ylabel('Censoring rate (% of person-time)');
title('Censoring rate by regime');
set(gcf, 'Units','inches', 'Position',[1 1 12 4.5]);
set(gcf, 'PaperUnits','inches', 'PaperSize',[12 4.5], 'PaperPosition',[0 0 12 4.5]);
print(gcf, fullfile(script_dir, 'Fig_alt_censoring.pdf'), '-dpdf', '-r300');

%% Text summary
fid = fopen(fullfile(script_dir, 'alt_censoring_summary.txt'), 'w');
fprintf(fid, 'Alternative-censoring sensitivity\n');
fprintf(fid, 'RCT truth ATE: %+.3f\n\n', ate_truth);
fprintf(fid, '%-10s %-14s %-14s %-14s\n', 'regime', 'cens rate', 'ATE', 'bias');
for i = 1:length(names)
    nm = names{i};
    fprintf(fid, '%-10s %-14.3f %-+14.3f %-+14.3f\n', nm, ...
            results.(nm).cens_rate, results.(nm).ate, results.(nm).bias);
end
fclose(fid);
