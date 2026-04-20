clear; clc; format compact;
rng(0) % Set random seed for reproducibility

%% 1. SIMULATION PARAMETERS ==========================================
N = 1000;   % Number of patients to simulate per trial type
nTrials = 1; % Number of trials to take median over (adjust as needed)

th = 0.1;  % Control target threshold (very low for tight control)
ki = 10;    % Integral control gain (aggressive disease suppression)
Amax = 50;  % Maximum pump rate (treatment upper bound)
parmsControl = [ki Amax];
ke = 0.5;

% Mortality (Y) hazard parameters
% logit_y = a0 + a1*(t(j)/170)^2 + (a2*sofa).*(cumsum_L/24)^2 + (a3*(age/90)).*(cumsum_A/207); 
a0 = -7;     
a1 = .3;
a2 = 20; % harmL
a3 = 5; % harmA

parmsY = [a0 a1 a2 a3]; 

% Censoring (V) hazard parameters
% logit_v = Rx(j)*(b0 + b1*(cumsum_A/207) + b2*(t(j)/170)^2) + (1-Rx(j))*(b3 + b4*cumsum_L/24 + b5*(t(j)/170)^2);                  
b0 = -5;    % Baseline censoring for treated
b1 = 2.0;    % Treatment burden effect
b2 = .1;   % Time effect for treated
b3 = -5;    % Baseline for untreated   
b4 = 2;     % Disease burden effect
b5 = 1.5;   % Time effect for untreated
parmsV = [b0 b1 b2 b3 b4 b5];

parmsL = [0.25, 1, 0.15, 0.05, 0.15, 0.03, 40];

dt = 2; t = 0:dt:168; 

%% ==============================================
% logit_y = a0 + a1*(t(j)/170)^2 + (a2*sofa).*(cumsum_L/24)^2 + (a3*(age/90)).*(cumsum_A/207); 

Th = linspace(0.01,1.1,20);
A2 = linspace(1,50,20); 
A3 = linspace(1,50,20); 

% load OptData
s0 = nan(20,20); 
s1 = nan(20,20); 

% make patients
[age, sofa, C, g, parmsPD] = fcnGeneratePatientParameters(N,'TargetCMean', 3, 'TargetGMean', 4, 'CV', 0.1);
[C_est, g_est, ke_est, parmsPD_est] = fcnGetPKPD_parms_est(age, sofa);
L0 = fcnGenerateStochasticTrajectories(t, parmsL, N);
treatProbBiased = fcnBiasedAssignmentProb(age, sofa, L0(:,1:5));
T0 = readtable('trialData0.csv');
parmsY_est = fcnEstimateDeathParms(T0);

for i = 1:length(A2); 
    for j = 1:length(A3); 
        disp('*****************')
        disp([A2(i) A3(j)])
        disp('*****************')
        parmsY = [-7 .3 A2(i) A3(j)]; % twiddle parms: 4, 6
        parmsY_est(3) = A2(i); parmsY_est(4) = A3(j); 
        
        % Arrays to store results from multiple trials
        ATE0_trials = nan(length(Th), nTrials);
        ATE1_trials = nan(length(Th), nTrials);
        
        for trial = 1:nTrials            
            for k = 1:length(Th); 
                th = Th(k); 
            
                % get survival curves from ideal data (RCT=1)
                RCT=1; treatProb = 0.5*ones(1,N); 
                T1 = fcnSimulate_N_Patients(N,RCT,treatProb, th, C, g, ke, L0, parmsControl, parmsY, parmsV, age, sofa);
            
                % estimate survival curves from observational data (RCT=0)
                RCT=0; 
                T0 = fcnSimulate_N_Patients(N,RCT,treatProbBiased, th, C, g, ke, L0, parmsControl, parmsY, parmsV, age, sofa);
                RCT = 1; % simulated RCT using estimated models
                treatProb = 0.5*ones(1,N); % for RCT simulation
                parmsV_est = [0 0 0 0 0 0]; % No censoring in RCT simulation
                T1_est = fcnSimulate_N_Patients(N,RCT,treatProb,th, C_est, g_est, ke_est, L0, parmsControl, parmsY_est, parmsV_est, age, sofa);

                [s0_est, s1_est, t0, t1] = fcnPlotKM(T1_est);
            
                S1_trials(k,trial) = s1_est(end); 
                S0_trials(k,trial) = s0_est(end); 

            end
        end
        
        % Take median across trials
        S0 = median(S0_trials, 2);
        S1 = median(S1_trials, 2);
        
        %% find optimal ATE
        [ii,jj] = max(S1-S0); 
        th_opt = Th(jj); 
        Th_opt(i,j) = th_opt;
        s0(i,j) = S0(jj); 
        s1(i,j) = S1(jj); 
        
        figure(1); clf; 
        subplot(311); 
        imagesc(A2, A3, s1'); axis xy  % Note: transposed
        xlabel('Harm from L')
        ylabel('Harm from A')
        set(gcf,'color','white')
        colorbar

        subplot(312); 
        imagesc(A2, A3, s0'); axis xy  % Note: transposed
        xlabel('Harm from L')
        ylabel('Harm from A')
        set(gcf,'color','white')
        colorbar
        
        subplot(313); 
        imagesc(A2, A3, (s1-s0)'); axis xy  % Note: transposed
        xlabel('Harm from L')
        ylabel('Harm from A')
        set(gcf,'color','white')
        colorbar
        
        drawnow
    end
    Th_opt3 = Th_opt; 
    save HeatMapData
end

%% Export optimization heatmap results to text file for paper
filename = sprintf('optimization_heatmap_results_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(filename, 'w');

fprintf(fid, '==========================================================\n');
fprintf(fid, 'OPTIMIZATION HEATMAP ANALYSIS RESULTS FOR PAPER\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '==========================================================\n\n');

% Study parameters
fprintf(fid, 'STUDY PARAMETERS:\n');
fprintf(fid, '- Sample size: %d patients\n', N);
fprintf(fid, '- Number of trials per parameter set: %d\n', nTrials);
fprintf(fid, '- Study period: 168 hours\n');
fprintf(fid, '- Time step: %d hours\n', dt);
fprintf(fid, '- PI controller gain: %.1f\n', ki);
fprintf(fid, '- Maximum pump rate: %.1f\n', Amax);
fprintf(fid, '- Elimination constant: %.2f\n\n', ke);

% Parameter optimization ranges
fprintf(fid, 'PARAMETER OPTIMIZATION RANGES:\n');
fprintf(fid, '- Treatment thresholds: %.2f to %.2f (20 levels)\n', min(Th), max(Th));
fprintf(fid, '- Disease harm (a2): %.1f to %.1f (20 levels)\n', min(A2), max(A2));
fprintf(fid, '- Treatment harm (a3): %.1f to %.1f (20 levels)\n', min(A3), max(A3));
fprintf(fid, '- Total optimization combinations: %d\n\n', length(A2) * length(A3) * length(Th));

% Baseline mortality parameters
fprintf(fid, 'BASELINE MORTALITY PARAMETERS:\n');
fprintf(fid, '- Baseline hazard (a0): %.1f\n', a0);
fprintf(fid, '- Time effect (a1): %.2f\n', a1);
fprintf(fid, '- Disease harm varies: %.1f to %.1f\n', min(A2), max(A2));
fprintf(fid, '- Treatment harm varies: %.1f to %.1f\n\n', min(A3), max(A3));

% Survival outcome analysis
fprintf(fid, 'SURVIVAL OUTCOME ANALYSIS:\n');
valid_s0 = s0(~isnan(s0));
valid_s1 = s1(~isnan(s1));
valid_ate = (s1 - s0); valid_ate = valid_ate(~isnan(valid_ate));
valid_th_opt = Th_opt(~isnan(Th_opt));

fprintf(fid, 'Optimized untreated survival (s0):\n');
fprintf(fid, '  Range: %.1f%% to %.1f%%\n', min(valid_s0)*100, max(valid_s0)*100);
fprintf(fid, '  Mean: %.1f%% ± %.1f%%\n', mean(valid_s0)*100, std(valid_s0)*100);

fprintf(fid, 'Optimized treated survival (s1):\n');
fprintf(fid, '  Range: %.1f%% to %.1f%%\n', min(valid_s1)*100, max(valid_s1)*100);
fprintf(fid, '  Mean: %.1f%% ± %.1f%%\n', mean(valid_s1)*100, std(valid_s1)*100);

fprintf(fid, 'Optimized treatment effect (ATE):\n');
fprintf(fid, '  Range: %.1f%% to %.1f%%\n', min(valid_ate)*100, max(valid_ate)*100);
fprintf(fid, '  Mean: %.1f%% ± %.1f%%\n', mean(valid_ate)*100, std(valid_ate)*100);

fprintf(fid, 'Optimal treatment thresholds:\n');
fprintf(fid, '  Range: %.3f to %.3f\n', min(valid_th_opt), max(valid_th_opt));
fprintf(fid, '  Mean: %.3f ± %.3f\n', mean(valid_th_opt), std(valid_th_opt));

% Find global optimum
[max_ate, max_idx] = max(valid_ate);
[max_i, max_j] = find((s1 - s0) == max_ate);

fprintf(fid, '\nGLOBAL OPTIMUM:\n');
if ~isempty(max_i)
    fprintf(fid, 'Best parameter combination:\n');
    fprintf(fid, '  Disease harm (a2): %.1f\n', A2(max_i(1)));
    fprintf(fid, '  Treatment harm (a3): %.1f\n', A3(max_j(1)));
    fprintf(fid, '  Optimal threshold: %.3f\n', Th_opt(max_i(1), max_j(1)));
    fprintf(fid, '  Maximum ATE: %.1f%% (%.1f%% vs %.1f%% survival)\n', ...
        max_ate*100, s1(max_i(1), max_j(1))*100, s0(max_i(1), max_j(1))*100);
end

% Find worst case scenario
[min_ate, min_idx] = min(valid_ate);
[min_i, min_j] = find((s1 - s0) == min_ate);

fprintf(fid, '\nWORST CASE SCENARIO:\n');
if ~isempty(min_i)
    fprintf(fid, 'Worst parameter combination:\n');
    fprintf(fid, '  Disease harm (a2): %.1f\n', A2(min_i(1)));
    fprintf(fid, '  Treatment harm (a3): %.1f\n', A3(min_j(1)));
    fprintf(fid, '  Optimal threshold: %.3f\n', Th_opt(min_i(1), min_j(1)));
    fprintf(fid, '  Minimum ATE: %.1f%% (%.1f%% vs %.1f%% survival)\n', ...
        min_ate*100, s1(min_i(1), min_j(1))*100, s0(min_i(1), min_j(1))*100);
end

% Treatment optimization effectiveness
fprintf(fid, '\nTREATMENT OPTIMIZATION EFFECTIVENESS:\n');

% Count beneficial vs harmful scenarios after optimization
beneficial_scenarios = sum(valid_ate > 0);
harmful_scenarios = sum(valid_ate < 0);
neutral_scenarios = sum(abs(valid_ate) < 0.01);
total_scenarios = length(valid_ate);

fprintf(fid, 'After optimization, treatment is:\n');
fprintf(fid, '  Beneficial (ATE > 0): %d/%d scenarios (%.1f%%)\n', ...
    beneficial_scenarios, total_scenarios, beneficial_scenarios/total_scenarios*100);
fprintf(fid, '  Harmful (ATE < 0): %d/%d scenarios (%.1f%%)\n', ...
    harmful_scenarios, total_scenarios, harmful_scenarios/total_scenarios*100);
fprintf(fid, '  Neutral (|ATE| < 1%%): %d/%d scenarios (%.1f%%)\n', ...
    neutral_scenarios, total_scenarios, neutral_scenarios/total_scenarios*100);

% High-impact optimization zones
high_benefit_threshold = prctile(valid_ate, 75);
high_benefit_scenarios = sum(valid_ate >= high_benefit_threshold);

fprintf(fid, '\nHIGH-IMPACT OPTIMIZATION ZONES:\n');
fprintf(fid, 'High benefit zone (top 25%% after optimization):\n');
fprintf(fid, '  ATE threshold: ≥ %.1f%%\n', high_benefit_threshold*100);
fprintf(fid, '  Number of scenarios: %d (%.1f%%)\n', ...
    high_benefit_scenarios, high_benefit_scenarios/total_scenarios*100);

% Optimal threshold distribution analysis
fprintf(fid, '\nOPTIMAL THRESHOLD DISTRIBUTION:\n');

% Analyze threshold preferences across parameter space
threshold_bins = [0.01, 0.1, 0.2, 0.5, 1.0, 1.1];
threshold_counts = histcounts(valid_th_opt, threshold_bins);
threshold_percentages = threshold_counts / sum(threshold_counts) * 100;

fprintf(fid, 'Optimal threshold preferences:\n');
for i = 1:length(threshold_counts)
    if i == length(threshold_counts)
        range_str = sprintf('%.2f-%.2f', threshold_bins(i), threshold_bins(i+1));
    else
        range_str = sprintf('%.2f-%.2f', threshold_bins(i), threshold_bins(i+1));
    end
    fprintf(fid, '  %s: %d scenarios (%.1f%%)\n', range_str, threshold_counts(i), threshold_percentages(i));
end

% Most common optimal threshold
[~, mode_idx] = max(threshold_counts);
fprintf(fid, 'Most common threshold range: %.2f-%.2f (%.1f%% of scenarios)\n', ...
    threshold_bins(mode_idx), threshold_bins(mode_idx+1), threshold_percentages(mode_idx));

% Parameter sensitivity for optimization
if length(A2) > 1 && length(A3) > 1
    fprintf(fid, '\nPARAMETER SENSITIVITY FOR OPTIMIZATION:\n');
    
    % Calculate mean optimal thresholds for each parameter level
    mean_th_by_disease = nanmean(Th_opt, 2);  % Average across treatment harm levels
    mean_th_by_treatment = nanmean(Th_opt, 1);  % Average across disease harm levels
    
    % Calculate correlations
    [disease_th_corr, disease_th_p] = corr(A2', mean_th_by_disease, 'rows', 'complete');
    [treatment_th_corr, treatment_th_p] = corr(A3', mean_th_by_treatment', 'rows', 'complete');
    
    fprintf(fid, 'Disease harm (a2) vs Optimal Threshold:\n');
    fprintf(fid, '  Correlation: %.3f (p = %.3f)\n', disease_th_corr, disease_th_p);
    if disease_th_corr > 0.5
        fprintf(fid, '  → Higher disease harm requires HIGHER optimal thresholds\n');
    elseif disease_th_corr < -0.5
        fprintf(fid, '  → Higher disease harm requires LOWER optimal thresholds\n');
    else
        fprintf(fid, '  → Disease harm has moderate effect on optimal thresholds\n');
    end
    
    fprintf(fid, 'Treatment harm (a3) vs Optimal Threshold:\n');
    fprintf(fid, '  Correlation: %.3f (p = %.3f)\n', treatment_th_corr, treatment_th_p);
    if treatment_th_corr > 0.5
        fprintf(fid, '  → Higher treatment harm requires HIGHER optimal thresholds\n');
    elseif treatment_th_corr < -0.5
        fprintf(fid, '  → Higher treatment harm requires LOWER optimal thresholds\n');
    else
        fprintf(fid, '  → Treatment harm has moderate effect on optimal thresholds\n');
    end
    
    % ATE sensitivity to parameters
    mean_ate_by_disease = nanmean(s1 - s0, 2);
    mean_ate_by_treatment = nanmean(s1 - s0, 1);
    
    [disease_ate_corr, disease_ate_p] = corr(A2', mean_ate_by_disease, 'rows', 'complete');
    [treatment_ate_corr, treatment_ate_p] = corr(A3', mean_ate_by_treatment', 'rows', 'complete');
    
    fprintf(fid, '\nDisease harm (a2) vs Optimized ATE:\n');
    fprintf(fid, '  Correlation: %.3f (p = %.3f)\n', disease_ate_corr, disease_ate_p);
    
    fprintf(fid, 'Treatment harm (a3) vs Optimized ATE:\n');
    fprintf(fid, '  Correlation: %.3f (p = %.3f)\n', treatment_ate_corr, treatment_ate_p);
end

% Optimization robustness
fprintf(fid, '\nOPTIMIZATION ROBUSTNESS:\n');

% Calculate coefficient of variation for optimal thresholds
cv_thresholds = std(valid_th_opt) / mean(valid_th_opt) * 100;
fprintf(fid, 'Optimal threshold variability: CV = %.1f%%\n', cv_thresholds);

if cv_thresholds < 20
    fprintf(fid, '  → Optimal thresholds are CONSISTENT across parameter space\n');
elseif cv_thresholds < 50
    fprintf(fid, '  → Optimal thresholds show MODERATE variability\n');
else
    fprintf(fid, '  → Optimal thresholds are HIGHLY variable across parameters\n');
end

% Safe optimization zones
conservative_benefit_threshold = 0.03;  % 3% minimum benefit
safe_optimization_scenarios = sum(valid_ate > conservative_benefit_threshold);

fprintf(fid, '\nSAFE OPTIMIZATION ZONES:\n');
fprintf(fid, 'Conservative benefit threshold (ATE > 3%%):\n');
fprintf(fid, '  Safe scenarios: %d/%d (%.1f%%)\n', ...
    safe_optimization_scenarios, total_scenarios, safe_optimization_scenarios/total_scenarios*100);

% Clinical decision making for optimization
fprintf(fid, '\nCLINICAL DECISION MAKING:\n');

if safe_optimization_scenarios/total_scenarios > 0.8
    fprintf(fid, '✓ OPTIMIZATION is broadly beneficial across parameter space\n');
    fprintf(fid, '  - Aggressive optimization strategies are safe\n');
    fprintf(fid, '  - Standard threshold optimization protocols recommended\n');
elseif beneficial_scenarios/total_scenarios > 0.7
    fprintf(fid, '⚠️  OPTIMIZATION benefits are parameter-dependent\n');
    fprintf(fid, '  - Careful parameter estimation required before optimization\n');
    fprintf(fid, '  - Consider adaptive optimization protocols\n');
else
    fprintf(fid, '✗ HIGH RISK for optimization without accurate parameters\n');
    fprintf(fid, '  - Conservative approaches recommended\n');
    fprintf(fid, '  - Extensive parameter validation required\n');
end

% Parameter estimation priorities for optimization
fprintf(fid, '\nPARAMETER ESTIMATION PRIORITIES FOR OPTIMIZATION:\n');
if abs(disease_th_corr) > abs(treatment_th_corr)
    fprintf(fid, '1. Disease harm parameters (a2) - highest impact on thresholds\n');
    fprintf(fid, '2. Treatment harm parameters (a3)\n');
else
    fprintf(fid, '1. Treatment harm parameters (a3) - highest impact on thresholds\n');
    fprintf(fid, '2. Disease harm parameters (a2)\n');
end

if abs(disease_ate_corr) > abs(treatment_ate_corr)
    fprintf(fid, '3. Disease harm also most critical for ATE optimization\n');
else
    fprintf(fid, '3. Treatment harm also most critical for ATE optimization\n');
end

% Practical optimization recommendations
fprintf(fid, '\nPRACTICAL OPTIMIZATION RECOMMENDATIONS:\n');

% Find parameter combinations that consistently give good results
if length(A2) > 1 && length(A3) > 1
    robust_optimization_mask = (s1 - s0) > mean(valid_ate);
    robust_optimization_count = sum(robust_optimization_mask(:));
    
    fprintf(fid, 'Robust optimization regions (above-average ATE):\n');
    fprintf(fid, '  Count: %d/%d scenarios (%.1f%%)\n', ...
        robust_optimization_count, numel(s1-s0), robust_optimization_count/numel(s1-s0)*100);
end

fprintf(fid, '\nFor clinical implementation:\n');
if mean(valid_th_opt) < 0.2
    fprintf(fid, '- Favor AGGRESSIVE optimization (low thresholds)\n');
elseif mean(valid_th_opt) > 0.5
    fprintf(fid, '- Favor CONSERVATIVE optimization (high thresholds)\n');
else
    fprintf(fid, '- Use MODERATE optimization thresholds\n');
end

fprintf(fid, '- Recommended threshold range: %.3f to %.3f\n', ...
    prctile(valid_th_opt, 25), prctile(valid_th_opt, 75));
fprintf(fid, '- Monitor treatment response and adjust adaptively\n');
fprintf(fid, '- Validate optimization performance with real-world data\n');

% Technical details
fprintf(fid, '\nTECHNICAL DETAILS:\n');
fprintf(fid, '- Optimization method: Exhaustive grid search\n');
fprintf(fid, '- Grid resolution: 20 × 20 × 20 (parameter × parameter × threshold)\n');
fprintf(fid, '- Total evaluations: %d\n', length(A2) * length(A3) * length(Th));
fprintf(fid, '- Optimization criterion: Maximum ATE at 168 hours\n');
fprintf(fid, '- Simulation: RCT emulation with observational parameter estimation\n');
fprintf(fid, '- Aggregation: Median across %d trial(s) per combination\n', nTrials);

% Data availability
fprintf(fid, '\nDATA AVAILABILITY:\n');
fprintf(fid, '- Complete optimization results saved to: HeatMapData.mat\n');
fprintf(fid, '- s0: Optimized untreated survival matrix (20×20)\n');
fprintf(fid, '- s1: Optimized treated survival matrix (20×20)\n');
fprintf(fid, '- Th_opt: Optimal threshold matrix (20×20)\n');
fprintf(fid, '- Parameter grids: A2 (disease harm), A3 (treatment harm), Th (thresholds)\n');

fclose(fid);
fprintf('Optimization heatmap analysis results exported to: %s\n', filename);
