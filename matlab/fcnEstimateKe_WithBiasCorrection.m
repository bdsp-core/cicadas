function [ke_est, ke_raw, results] = fcnEstimateKe_WithBiasCorrection(L_obs, A_obs, varargin)
%% Standalone ke Estimation with Empirical Bias Correction
%
% Based on empirical testing showing ~29% systematic underestimation,
% this function applies a calibrated correction factor.
%
% OUTPUTS:
%   ke_est: bias-corrected ke estimate
%   ke_raw: raw ke estimate (before correction)
%   results: detailed results including both estimates

    % Parse inputs
    p = inputParser;
    addParameter(p, 'CorrectionFactor', 1.41, @isnumeric);  % Empirically derived
    addParameter(p, 'UsePrior', false, @islogical);
    addParameter(p, 'PriorKe', 0.5, @isnumeric);
    addParameter(p, 'PriorWeight', 0.3, @isnumeric);  % Weight for prior
    addParameter(p, 'KeRange', [0.3, 0.9], @(x) isnumeric(x) && length(x)==2);
    addParameter(p, 'AssumeC', 3.0, @isnumeric);
    addParameter(p, 'AssumeG', 4.0, @isnumeric);
    addParameter(p, 'Verbose', true, @islogical);
    parse(p, varargin{:});
    
    correction_factor = p.Results.CorrectionFactor;
    use_prior = p.Results.UsePrior;
    prior_ke = p.Results.PriorKe;
    prior_weight = p.Results.PriorWeight;
    ke_range = p.Results.KeRange;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('\n========================================\n');
        fprintf('ke ESTIMATION WITH BIAS CORRECTION\n');
        fprintf('========================================\n');
        fprintf('Correction factor: %.2f\n', correction_factor);
        if use_prior
            fprintf('Using prior: ke = %.3f (weight = %.2f)\n', prior_ke, prior_weight);
        end
    end
    
    %% Step 1: Get raw ke estimate
    [ke_raw, results_raw] = fcnEstimateKe_Standalone(L_obs, A_obs, ...
        'AssumeC', p.Results.AssumeC, ...
        'AssumeG', p.Results.AssumeG, ...
        'KeRange', ke_range, ...
        'WindowSize', 10, ...
        'MinDoseChange', 1.0, ...
        'EstimationMethod', 'robust', ...
        'Verbose', false);
    
    if verbose
        fprintf('\nRaw estimate: %.3f\n', ke_raw);
    end
    
    %% Step 2: Apply bias correction
    ke_corrected = ke_raw * correction_factor;
    
    if verbose
        fprintf('After correction: %.3f\n', ke_corrected);
    end
    
    %% Step 3: Optional prior incorporation
    if use_prior
        % Weighted average with prior
        ke_with_prior = (1 - prior_weight) * ke_corrected + prior_weight * prior_ke;
        
        if verbose
            fprintf('After prior incorporation: %.3f\n', ke_with_prior);
        end
        
        ke_est = ke_with_prior;
    else
        ke_est = ke_corrected;
    end
    
    %% Step 4: Ensure within physiological bounds
    ke_est = max(ke_range(1), min(ke_range(2), ke_est));
    
    if verbose && (ke_est == ke_range(1) || ke_est == ke_range(2))
        fprintf('⚠ Estimate bounded to physiological range\n');
    end
    
    %% Compile results
    results = results_raw;
    results.ke_raw = ke_raw;
    results.ke_corrected = ke_corrected;
    results.ke_final = ke_est;
    results.correction_factor = correction_factor;
    results.used_prior = use_prior;
    
    if use_prior
        results.prior_ke = prior_ke;
        results.prior_weight = prior_weight;
    end
    
    % Confidence intervals with correction
    if isfield(results_raw, 'ke_ci95')
        results.ci95_raw = results_raw.ke_ci95;
        results.ci95_corrected = results_raw.ke_ci95 * correction_factor;
        results.ci95_final = [max(ke_range(1), results.ci95_corrected(1)), ...
                              min(ke_range(2), results.ci95_corrected(2))];
    end
    
    if verbose
        fprintf('\n========================================\n');
        fprintf('FINAL RESULT:\n');
        fprintf('========================================\n');
        fprintf('ke estimate: %.3f\n', ke_est);
        if isfield(results, 'ci95_final')
            fprintf('95%% CI: [%.3f, %.3f]\n', results.ci95_final(1), results.ci95_final(2));
        end
        fprintf('========================================\n');
    end
end