%% NUC-injection sensitivity analysis for CICADAS
%
% Injects a hidden (unmeasured) binary confounder U ~ Bernoulli(0.5) that
% influences both treatment assignment and mortality. Runs the CICADAS
% g-formula pipeline without U (i.e., U is unmeasured by construction) and
% reports the resulting bias in the recovered ATE as a function of
% (delta_A, delta_Y) -- the additive logit-shifts produced by U on
% treatment assignment and on mortality hazard respectively.
%
% Also computes the Ding & VanderWeele E-value for the unconfounded ATE and
% reports the tipping-point delta_Y at which the ATE sign flips.
%
% Run from the CICADAS repo root so that addpath(genpath(pwd)) resolves the
% existing fcn*.m helpers.
%
% Outputs: sensitivity/nuc_injection/nuc_results.mat
%          sensitivity/nuc_injection/Fig_NUC_sensitivity.pdf

clear; clc; format compact;
script_dir = fileparts(mfilename('fullpath'));
repo_root  = fileparts(fileparts(script_dir));
addpath(repo_root);
addpath(genpath(fullfile(repo_root, 'CICADA_FIGURES')));

%% 1. Baseline params (copied from a0_GenerateTrialData.m)
rng(42);
N = 1000;
th = 0.1;
ki = 10; Amax = 50;
parmsControl = [ki Amax];
[age, sofa, C, g, parmsPD] = fcnGeneratePatientParameters(N, ...
    'TargetCMean', 3, 'TargetGMean', 4, 'CV', 0.1);
ke = 0.5;
parmsY = [-7, .3, 20, 5];
parmsV = [-5, 2.0, .1, -5, 2, 1.5];
parmsL = [0.25, 1, 0.15, 0.05, 0.15, 0.03, 40];
dt = 2; t = 0:dt:168;

%% 2. Sensitivity grid
delta_A_vals = [0.0, 0.5, 1.0, 1.5, 2.0];   % logit shift for treatment assignment
delta_Y_vals = [0.0, 0.5, 1.0, 1.5, 2.0];   % logit shift for mortality hazard

nA = numel(delta_A_vals); nY = numel(delta_Y_vals);
ate_rct      = nan(nA, nY);   % oracle RCT ATE (U still active in DGP)
ate_naive    = nan(nA, nY);
ate_gformula = nan(nA, nY);
bias_gf      = nan(nA, nY);

%% 3. Sweep
total = nA * nY;
k = 0;
t_all = tic;

for ia = 1:nA
    dA = delta_A_vals(ia);
    for iY = 1:nY
        dY = delta_Y_vals(iY);
        k = k + 1;
        fprintf('\n[%d/%d] delta_A=%.2f, delta_Y=%.2f ...\n', k, total, dA, dY);
        t_cond = tic;

        % per-patient hidden confounder (same across RCT and Obs arms for this condition)
        U = rand(1, N) < 0.5;
        u_shift_Y = U * dY;

        %% ---- RCT arm (U still modifies mortality; no selection) ----
        L0 = fcnGenerateStochasticTrajectories(t, parmsL, N);
        treatProb_rct = 0.5*ones(1, N);
        T1 = fcnSimulate_N_Patients_withU(N, 1, treatProb_rct, th, C, g, ke, L0, ...
                                          parmsControl, parmsY, parmsV, age, sofa, ...
                                          u_shift_Y);
        [s0_true, s1_true] = fcnPlotKM(T1);
        close all;
        ate_rct(ia, iY) = s1_true(end) - s0_true(end);

        %% ---- Observational arm (U affects treatment assignment via dA) ----
        treatProb_obs_base = fcnBiasedAssignmentProb(age, sofa, L0(:, 1:5));
        logit_base = log(max(min(treatProb_obs_base, 1-1e-6), 1e-6) ./ ...
                         (1 - max(min(treatProb_obs_base, 1-1e-6), 1e-6)));
        logit_obs  = logit_base + U * dA;
        treatProb_obs = 1 ./ (1 + exp(-logit_obs));
        T0 = fcnSimulate_N_Patients_withU(N, 0, treatProb_obs, th, C, g, ke, L0, ...
                                          parmsControl, parmsY, parmsV, age, sofa, ...
                                          u_shift_Y);

        %% ---- Naive KM from observational data ----
        [s0_naive, s1_naive] = naiveKM(T0);
        ate_naive(ia, iY) = s1_naive(end) - s0_naive(end);

        %% ---- G-formula (U is NOT in T0; hidden by construction) ----
        try
            parmsY_est = fcnEstimateDeathParms(T0);
            [parmsL_est, LL, AA, age_e, sofa_e, t_e] = fcnEstimateParmsL(T0);
            [ke_est, C_est, g_est] = fcnEstimateParmsPKPD(parmsL_est, LL, AA, age_e, sofa_e, t_e);
            L0_est = fcnGenerateStochasticTrajectories(t, parmsL_est, N);
            % simulate counterfactual RCT using the estimated DGP (no U, baseline parmsV=0)
            T1_est = fcnSimulate_N_Patients_withU(N, 1, 0.5*ones(1,N), th, C_est, g_est, ...
                                                   ke_est, L0_est, parmsControl, ...
                                                   parmsY_est, zeros(1,6), age, sofa, ...
                                                   zeros(1,N));
            [s0_gf, s1_gf] = fcnPlotKM(T1_est);
            close all;
            ate_gformula(ia, iY) = s1_gf(end) - s0_gf(end);
            bias_gf(ia, iY) = ate_gformula(ia, iY) - ate_rct(ia, iY);
        catch ME
            fprintf('  [WARN] g-formula failed: %s\n', ME.message);
        end

        fprintf('  ATE_rct=%.3f, ATE_naive=%.3f, ATE_gf=%.3f, bias=%.3f (%.1fs)\n', ...
                ate_rct(ia, iY), ate_naive(ia, iY), ate_gformula(ia, iY), ...
                bias_gf(ia, iY), toc(t_cond));
    end
end

fprintf('\nSweep total: %.1f min\n', toc(t_all)/60);

%% 4. E-value on the zero-perturbation (honest) ATE
% Dragon/VanderWeele E-value for a risk ratio (RR).
% Convert ATE on survival difference to RR on mortality at 168h for honest condition:
% zeros are at (ia=1, iY=1) = (delta_A=0, delta_Y=0)
mort_t = 1 - s1_true(end);                % treated mortality at baseline
mort_u = 1 - s0_true(end);                % untreated mortality at baseline
if mort_t > 0 && mort_u > 0
    RR = mort_u / mort_t;                 % protective RR > 1 means treatment is beneficial
    if RR < 1, RR = 1/RR; end             % E-value formula uses RR >= 1
    evalue = RR + sqrt(RR * (RR - 1));
else
    evalue = NaN;
end

%% 5. Tipping-point: smallest delta_Y at delta_A=0 that flips ATE sign
sign_flip_dY = NaN;
for iY = 1:nY
    if sign(ate_gformula(1, iY)) ~= sign(ate_rct(1, 1))
        sign_flip_dY = delta_Y_vals(iY);
        break;
    end
end

%% 6. Save results
out_file = fullfile(script_dir, 'nuc_results.mat');
save(out_file, 'delta_A_vals','delta_Y_vals','ate_rct','ate_naive','ate_gformula', ...
     'bias_gf','evalue','sign_flip_dY','N','parmsY','parmsV');
fprintf('Saved results to %s\n', out_file);

%% 7. Heatmap figure
figure(1); clf;
tiledlayout(1, 2, 'Padding','compact','TileSpacing','compact');
nexttile;
imagesc(delta_A_vals, delta_Y_vals, 100*bias_gf.', 'AlphaData', ~isnan(bias_gf.'));
set(gca, 'YDir','normal');
colorbar; colormap(redblue_map());
xlabel('\delta_A (logit shift on treatment)');
ylabel('\delta_Y (logit shift on mortality)');
title(sprintf('G-formula bias in ATE (pp)\nN=%d patients', N));
caxis([-max(abs(bias_gf(:)*100), [], 'omitnan'), max(abs(bias_gf(:)*100), [], 'omitnan')]);
for ia = 1:nA
    for iY = 1:nY
        if ~isnan(bias_gf(ia, iY))
            text(delta_A_vals(ia), delta_Y_vals(iY), sprintf('%+.1f', 100*bias_gf(ia, iY)), ...
                 'HorizontalAlignment','center', 'FontSize',9);
        end
    end
end

nexttile;
imagesc(delta_A_vals, delta_Y_vals, 100*ate_gformula.');
set(gca, 'YDir','normal');
colorbar; colormap(redblue_map());
xlabel('\delta_A'); ylabel('\delta_Y');
title('G-formula ATE (pp)');
for ia = 1:nA
    for iY = 1:nY
        if ~isnan(ate_gformula(ia, iY))
            text(delta_A_vals(ia), delta_Y_vals(iY), sprintf('%+.1f', 100*ate_gformula(ia, iY)), ...
                 'HorizontalAlignment','center', 'FontSize',9);
        end
    end
end

set(gcf, 'Units','inches', 'Position',[1 1 12 5]);
set(gcf, 'PaperUnits','inches', 'PaperSize',[12 5], 'PaperPosition',[0 0 12 5]);
print(gcf, fullfile(script_dir, 'Fig_NUC_sensitivity.pdf'), '-dpdf', '-r300');
fprintf('Saved figure to %s\n', fullfile(script_dir, 'Fig_NUC_sensitivity.pdf'));

%% 8. Text summary
txt_file = fullfile(script_dir, 'nuc_summary.txt');
fid = fopen(txt_file, 'w');
fprintf(fid, 'NUC sensitivity analysis\n');
fprintf(fid, '========================\n');
fprintf(fid, 'N = %d patients per arm\n', N);
fprintf(fid, 'delta_A grid: %s\n', num2str(delta_A_vals));
fprintf(fid, 'delta_Y grid: %s\n', num2str(delta_Y_vals));
fprintf(fid, '\nE-value (honest ATE, delta_A=delta_Y=0): %.2f\n', evalue);
if ~isnan(sign_flip_dY)
    fprintf(fid, 'Tipping-point delta_Y (at delta_A=0) for ATE sign flip: %.2f\n', sign_flip_dY);
else
    fprintf(fid, 'No sign flip observed in the tested delta_Y range.\n');
end
fprintf(fid, '\nBias heatmap (rows=delta_A, cols=delta_Y):\n');
for ia = 1:nA
    fprintf(fid, '%.2f |', delta_A_vals(ia));
    for iY = 1:nY
        fprintf(fid, ' %+5.1f', 100*bias_gf(ia, iY));
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('Saved summary to %s\n', txt_file);

%% Helpers
function [s0, s1] = naiveKM(T0)
uids = unique(T0.sid);
n = length(uids);
tt1 = []; ev1 = []; tt0_ = []; ev0_ = [];
for i = 1:n
    d = T0(T0.sid == uids(i), :);
    d = sortrows(d, 't');
    dth = find(d.Y > 0, 1, 'first');
    if ~isempty(dth)
        tm = d.t(dth); ev = 1;
    else
        tm = max(d.t); ev = 0;
    end
    if d.Rx(1) == 1
        tt1(end+1,1) = tm; ev1(end+1,1) = ev;
    else
        tt0_(end+1,1) = tm; ev0_(end+1,1) = ev;
    end
end
[f1, x1] = ecdf(tt1, 'Censoring', ~ev1);
s1 = [1; 1-f1];
[f0, x0] = ecdf(tt0_, 'Censoring', ~ev0_);
s0 = [1; 1-f0];
end

function cm = redblue_map()
% Simple red-to-white-to-blue diverging map.
n = 128;
r = [linspace(0, 1, n), ones(1, n)];
g = [linspace(0, 1, n), linspace(1, 0, n)];
b = [ones(1, n), linspace(1, 0, n)];
cm = [r(:), g(:), b(:)];
end
