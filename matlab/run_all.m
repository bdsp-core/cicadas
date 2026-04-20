function run_all()
% RUN_ALL  Full CICADAS pipeline: generate data, estimate, analyze, plot.
%
% Runs scripts a0 -> a1 -> a2 -> a3 -> a4_* -> figure scripts in order.
% Designed for reproducibility: each script uses its own RNG seed or
% inherits from the prior script. See README.md for expected runtime.
%
% Invoke from the repo root (the directory that contains matlab/,
% CICADA_FIGURES/, sensitivity/, etc.):
%   >> run('matlab/run_all')
%
% Individual scripts can be run interactively in the same order by
% first adding the matlab/ directory to the path.

% Add all subdirectories of the repo root to the path so that the a*
% and fcn* functions (now under matlab/) and the figure scripts
% (under CICADA_FIGURES/) are all findable regardless of current dir.
repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(repo_root));
cd(repo_root);  % analyses read trialData*.csv from the repo root.

stages = {
    'a0_GenerateTrialData'              % ~25 s
    'a0_GenerateDoseSwitchingData'      % ~90 s
    'a1_EstimatePKPD'                   % ~85 s
    'a2_CausalSurvivalAnalysis'         % ~40 min (1000 bootstraps)
    'a3_ThreeTreatmentTargets'          % ~15 s
    'a4_HeatMap_Agressive'              % ~3 min
    'a4_OptimalTreatmentTarget'         % ~1 h (parallel bootstrap)
    'a4_Optimize_Heatmap'               % ~45 min
};

figure_scripts = {
    'a1_SingleTraces'                       % -> Fig1_singleTrajectories_3panels.pdf
    'a2_EvaluatePKPD_estimates_figures'     % -> Fig_Combined_PKPD_Analysis.pdf
    'a3a_Fig_Swimmers_RCT'                  % -> Fig3_swimmer_survival_plot_RCT.pdf
    'a3b_Fig_Swimmers_Obs_Naive'            % -> Fig4_swimmer_survival_plot_Obs_Naive.pdf
    'a3c_Fig_Swimmers_Obs_g_formula'        % -> Fig_gformula_corrected_survival_curves.pdf
    'a4_HeatMaps_Combined'                  % -> Fig_heatmap_figure.pdf
    'a5_OptimizationCurve'                  % -> Fig_optimization_curves_with_survival.pdf
};

fprintf('\n==========================================================\n');
fprintf(' CICADAS full pipeline\n');
fprintf('==========================================================\n');

t0_all = tic;

% --- Analysis stages ---
for i = 1:numel(stages)
    s = stages{i};
    fprintf('\n[STAGE %d/%d] %s ...\n', i, numel(stages), s);
    t0 = tic;
    run(s);
    fprintf('[STAGE %d/%d] %s done (%.1f s)\n', i, numel(stages), s, toc(t0));
end

% --- Figure generation ---
fprintf('\n==========================================================\n');
fprintf(' Figure generation\n');
fprintf('==========================================================\n');
orig_dir = pwd;
cleanup = onCleanup(@() cd(orig_dir));
cd('CICADA_FIGURES');
for i = 1:numel(figure_scripts)
    s = figure_scripts{i};
    fprintf('\n[FIG %d/%d] %s ...\n', i, numel(figure_scripts), s);
    t0 = tic;
    try
        run(s);
        fprintf('[FIG %d/%d] %s done (%.1f s)\n', i, numel(figure_scripts), s, toc(t0));
    catch ME
        fprintf('[FIG %d/%d] %s FAILED: %s\n', i, numel(figure_scripts), s, ME.message);
    end
end

fprintf('\n==========================================================\n');
fprintf(' CICADAS full pipeline finished in %.1f min\n', toc(t0_all)/60);
fprintf('==========================================================\n');
end
