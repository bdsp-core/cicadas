function run_all_figures()
% Run all 7 manuscript figure-generation scripts in one MATLAB session.
% Execute from the parent directory (Simulations_Causal_PKPD) so that
% readtable/load calls find trialData*.csv and *.mat in the CWD.

scripts = {
    'a1_SingleTraces'                     % -> Fig1_singleTrajectories_3panels.pdf
    'a2_EvaluatePKPD_estimates_figures'   % -> Fig_Combined_PKPD_Analysis.pdf
    'a3a_Fig_Swimmers_RCT'                % -> Fig3_swimmer_survival_plot_RCT.pdf
    'a3b_Fig_Swimmers_Obs_Naive'          % -> Fig4_swimmer_survival_plot_Obs_Naive.pdf
    'a3c_Fig_Swimmers_Obs_g_formula'      % -> Fig_gformula_corrected_survival_curves.pdf
    'a4_HeatMaps_Combined'                % -> Fig_heatmap_figure.pdf
    'a5_OptimizationCurve'                % -> Fig_optimization_curves_with_survival.pdf
};

addpath(fullfile(pwd, 'CICADA_FIGURES'));

fprintf('\n================================================\n');
fprintf('RUNNING %d FIGURE SCRIPTS\n', numel(scripts));
fprintf('================================================\n');

for i = 1:numel(scripts)
    s = scripts{i};
    fprintf('\n[%d/%d] %s ...\n', i, numel(scripts), s);
    t0 = tic;
    try
        close all; clearvars -except scripts i s t0;
        run(fullfile('CICADA_FIGURES', [s '.m']));
        fprintf('  [OK] %s (%.1fs)\n', s, toc(t0));
    catch ME
        fprintf('  [FAIL] %s: %s\n', s, ME.message);
        for k = 1:min(3, length(ME.stack))
            fprintf('      at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
        end
    end
end

fprintf('\n================================================\n');
fprintf('DONE\n');
fprintf('================================================\n');
end
