%% Multi-seed replication of the CICADAS headline result (MATLAB pipeline).
%
% Companion to multiseed_py.py. Establishes the sampling distribution of the
% RCT ground-truth ATE and the naive Kaplan-Meier ATE, so that (a) the paper can
% report means with Monte Carlo SEs rather than single draws, and (b) we can test
% whether the MATLAB/Python divergence at seed 0 is Monte Carlo error or a code
% discrepancy.
%
% Replicates the RNG consumption order of a0_GenerateTrialData.m exactly:
%   rng(seed) -> fcnGeneratePatientParameters -> for RCT = 0:1:
%                fcnGenerateStochasticTrajectories -> (fcnBiasedAssignmentProb)
%                -> fcnSimulate_N_Patients
%
% Usage:  matlab -batch "n_seeds=200; N=2000; multiseed_matlab"
% Output: multiseed_matlab_results.csv

if ~exist('n_seeds','var'); n_seeds = 200; end
if ~exist('N','var');       N       = 2000; end

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab'));

% --- Simulation parameters: identical to a0_GenerateTrialData.m --------------
th = 0.1;
ki = 10; Amax = 50;
ke = 0.5;
parmsControl = [ki Amax];
parmsY = [-7, 0.3, 20, 5];
parmsV = [-5, 2.0, 0.1, -5, 2, 1.5];
parmsL = [0.25, 1, 0.15, 0.05, 0.15, 0.03, 40];
dt = 2; t = 0:dt:168;

res = zeros(n_seeds, 8);

for s = 1:n_seeds
    seed = s - 1;                      % seeds 0..n_seeds-1, matching Python
    rng(seed);

    [age, sofa, C, g, ~] = fcnGeneratePatientParameters(N, ...
        'TargetCMean', 3, 'TargetGMean', 4, 'CV', 0.1);

    Tc = cell(1,2);
    for RCT = 0:1
        L0 = fcnGenerateStochasticTrajectories(t, parmsL, N);
        if RCT == 1
            treatProb = 0.5*ones(1,N);
        else
            evalc('treatProb = fcnBiasedAssignmentProb(age, sofa, L0(:,1:5));');
        end
        Tc{RCT+1} = fcnSimulate_N_Patients(N, RCT, treatProb, th, C, g, ke, ...
            L0, parmsControl, parmsY, parmsV, age, sofa);
    end

    % --- RCT ground truth -----------------------------------------------
    [s0_true, s1_true] = fcnPlotKM(Tc{2});
    ate_true = s1_true(end) - s0_true(end);

    % --- Naive KM on the observational cohort ---------------------------
    T0 = Tc{1};
    sids = unique(T0.sid);
    times = zeros(numel(sids),1); events = zeros(numel(sids),1); arms = zeros(numel(sids),1);
    for k = 1:numel(sids)
        pd = T0(T0.sid == sids(k), :);
        if any(pd.Y == 1)
            times(k) = min(pd.t(pd.Y == 1)); events(k) = 1;
        else
            times(k) = max(pd.t);            events(k) = 0;
        end
        arms(k) = pd.Rx(1);
    end
    [f1, ~] = ecdf(times(arms==1), 'Censoring', ~events(arms==1));
    [f0, ~] = ecdf(times(arms==0), 'Censoring', ~events(arms==0));
    s1_naive = 1 - f1(end);
    s0_naive = 1 - f0(end);
    ate_naive = s1_naive - s0_naive;

    res(s,:) = [seed, ate_true, s1_true(end), s0_true(end), ...
                ate_naive, s1_naive, s0_naive, ate_naive - ate_true];

    fprintf('seed %3d  truth %+6.2f%%  naive %+6.2f%%  bias %+6.2fpp  reversal=%d\n', ...
        seed, 100*ate_true, 100*ate_naive, 100*(ate_naive-ate_true), ate_naive < 0);
end

Tout = array2table(res, 'VariableNames', ...
    {'seed','ate_true','s1_true','s0_true','ate_naive','s1_naive','s0_naive','bias'});
Tout.reversal = double(Tout.ate_naive < 0);
outfile = fullfile(fileparts(mfilename('fullpath')), 'multiseed_matlab_results.csv');
writetable(Tout, outfile);

fprintf('\n%s\n', repmat('=',1,78));
fprintf('MATLAB PIPELINE — %d seeds, N=%d per arm-cohort\n', n_seeds, N);
fprintf('%s\n', repmat('=',1,78));
fprintf('%-16s%8s%8s%8s   95%% range\n','quantity','mean','SD','MCSE');
for c = {'ate_true','ate_naive','bias'}
    v = 100*Tout.(c{1});
    fprintf('%-16s%+8.2f%8.2f%8.3f   [%+6.2f, %+6.2f]\n', c{1}, mean(v), std(v), ...
        std(v)/sqrt(numel(v)), prctile(v,2.5), prctile(v,97.5));
end
fprintf('\nsign reversal (naive < 0) in %d/%d = %.0f%% of seeds\n', ...
    sum(Tout.reversal), n_seeds, 100*mean(Tout.reversal));
fprintf('saved -> %s\n', outfile);
