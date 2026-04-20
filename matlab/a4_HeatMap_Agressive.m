%% get survival curves for 2 different scenarios

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

Th = 0.1; % linspace(0.01,1.1,20);
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
end

save HeatMapAggressive

%% Export heatmap sensitivity analysis results to text file for paper
filename = sprintf('heatmap_sensitivity_results_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(filename, 'w');

fprintf(fid, '==========================================================\n');
fprintf(fid, 'HEATMAP SENSITIVITY ANALYSIS RESULTS FOR PAPER\n');
fprintf(fid, 'Generated on: %s\n', datestr(now));
fprintf(fid, '==========================================================\n\n');

% Study parameters
fprintf(fid, 'STUDY PARAMETERS:\n');
fprintf(fid, '- Sample size: %d patients\n', N);
fprintf(fid, '- Number of trials per parameter set: %d\n', nTrials);
fprintf(fid, '- Study period: 168 hours\n');
fprintf(fid, '- Time step: %d hours\n', dt);
fprintf(fid, '- Control target threshold: %.2f\n', th);
fprintf(fid, '- PI controller gain: %.1f\n', ki);
fprintf(fid, '- Maximum pump rate: %.1f\n', Amax);
fprintf(fid, '- Elimination constant: %.2f\n\n', ke);

% Parameter ranges
fprintf(fid, 'PARAMETER SENSITIVITY RANGES:\n');
fprintf(fid, '- Harm from disease (a2): %.1f to %.1f (20 levels)\n', min(A2), max(A2));
fprintf(fid, '- Harm from treatment (a3): %.1f to %.1f (20 levels)\n', min(A3), max(A3));
fprintf(fid, '- Total parameter combinations: %d\n\n', length(A2) * length(A3));

% Baseline mortality parameters
fprintf(fid, 'BASELINE MORTALITY PARAMETERS:\n');
fprintf(fid, '- Baseline hazard (a0): %.1f\n', a0);
fprintf(fid, '- Time effect (a1): %.2f\n', a1);
fprintf(fid, '- Disease harm varies: %.1f to %.1f\n', min(A2), max(A2));
fprintf(fid, '- Treatment harm varies: %.1f to %.1f\n\n', min(A3), max(A3));

% Survival outcome ranges
fprintf(fid, 'SURVIVAL OUTCOME RANGES:\n');
valid_s0 = s0(~isnan(s0));
valid_s1 = s1(~isnan(s1));
valid_ate = (s1 - s0); valid_ate = valid_ate(~isnan(valid_ate));

fprintf(fid, 'Untreated survival (s0):\n');
fprintf(fid, '  Range: %.1f%% to %.1f%%\n', min(valid_s0)*100, max(valid_s0)*100);
fprintf(fid, '  Mean: %.1f%% ± %.1f%%\n', mean(valid_s0)*100, std(valid_s0)*100);

fprintf(fid, 'Treated survival (s1):\n');
fprintf(fid, '  Range: %.1f%% to %.1f%%\n', min(valid_s1)*100, max(valid_s1)*100);
fprintf(fid, '  Mean: %.1f%% ± %.1f%%\n', mean(valid_s1)*100, std(valid_s1)*100);

fprintf(fid, 'Treatment effect (ATE):\n');
fprintf(fid, '  Range: %.1f%% to %.1f%%\n', min(valid_ate)*100, max(valid_ate)*100);
fprintf(fid, '  Mean: %.1f%% ± %.1f%%\n', mean(valid_ate)*100, std(valid_ate)*100);

% Optimal treatment zones
[max_ate, max_idx] = max(valid_ate);
[min_ate, min_idx] = min(valid_ate);

% Find indices in the original matrices
[opt_i, opt_j] = find((s1 - s0) == max_ate);
[worst_i, worst_j] = find((s1 - s0) == min_ate);

fprintf(fid, '\nOPTIMAL TREATMENT ZONES:\n');
fprintf(fid, 'Best treatment scenario:\n');
if ~isempty(opt_i)
    fprintf(fid, '  Parameters: Disease harm = %.1f, Treatment harm = %.1f\n', A2(opt_i(1)), A3(opt_j(1)));
    fprintf(fid, '  ATE: %.1f%% (%.1f%% vs %.1f%% survival)\n', ...
        max_ate*100, s1(opt_i(1), opt_j(1))*100, s0(opt_i(1), opt_j(1))*100);
end

fprintf(fid, 'Worst treatment scenario:\n');
if ~isempty(worst_i)
    fprintf(fid, '  Parameters: Disease harm = %.1f, Treatment harm = %.1f\n', A2(worst_i(1)), A3(worst_j(1)));
    fprintf(fid, '  ATE: %.1f%% (%.1f%% vs %.1f%% survival)\n', ...
        min_ate*100, s1(worst_i(1), worst_j(1))*100, s0(worst_i(1), worst_j(1))*100);
end

% Treatment benefit analysis
beneficial_count = sum(valid_ate > 0);
harmful_count = sum(valid_ate < 0);
neutral_count = sum(abs(valid_ate) < 0.01);
total_scenarios = length(valid_ate);

fprintf(fid, '\nTREATMENT BENEFIT ANALYSIS:\n');
fprintf(fid, 'Scenarios where treatment is:\n');
fprintf(fid, '  Beneficial (ATE > 0): %d/%d (%.1f%%)\n', beneficial_count, total_scenarios, beneficial_count/total_scenarios*100);
fprintf(fid, '  Harmful (ATE < 0): %d/%d (%.1f%%)\n', harmful_count, total_scenarios, harmful_count/total_scenarios*100);
fprintf(fid, '  Neutral (|ATE| < 1%%): %d/%d (%.1f%%)\n', neutral_count, total_scenarios, neutral_count/total_scenarios*100);

% Parameter sensitivity analysis
if length(A2) > 1 && length(A3) > 1
    % Effect of disease harm (A2) on outcomes
    ate_matrix = s1 - s0;
    mean_ate_by_disease = nanmean(ate_matrix, 2);  % Average across treatment harm levels
    mean_ate_by_treatment = nanmean(ate_matrix, 1);  % Average across disease harm levels
    
    fprintf(fid, '\nPARAMETER SENSITIVITY:\n');
    
    % Disease harm sensitivity
    [disease_corr, disease_p] = corr(A2', mean_ate_by_disease, 'rows', 'complete');
    fprintf(fid, 'Disease harm (a2) vs Treatment Effect:\n');
    fprintf(fid, '  Correlation: %.3f (p = %.3f)\n', disease_corr, disease_p);
    if disease_corr < -0.5
        fprintf(fid, '  → Higher disease harm REDUCES treatment benefit\n');
    elseif disease_corr > 0.5
        fprintf(fid, '  → Higher disease harm INCREASES treatment benefit\n');
    else
        fprintf(fid, '  → Disease harm has moderate effect on treatment benefit\n');
    end
    
    % Treatment harm sensitivity
    [treatment_corr, treatment_p] = corr(A3', mean_ate_by_treatment', 'rows', 'complete');
    fprintf(fid, 'Treatment harm (a3) vs Treatment Effect:\n');
    fprintf(fid, '  Correlation: %.3f (p = %.3f)\n', treatment_corr, treatment_p);
    if treatment_corr < -0.5
        fprintf(fid, '  → Higher treatment harm REDUCES treatment benefit\n');
    elseif treatment_corr > 0.5
        fprintf(fid, '  → Higher treatment harm INCREASES treatment benefit\n');
    else
        fprintf(fid, '  → Treatment harm has moderate effect on treatment benefit\n');
    end
end

% Risk-benefit zones
fprintf(fid, '\nRISK-BENEFIT ZONES:\n');

% High benefit zone (top 25% of ATE values)
high_benefit_threshold = prctile(valid_ate, 75);
high_benefit_count = sum(valid_ate >= high_benefit_threshold);

fprintf(fid, 'High benefit zone (top 25%% of scenarios):\n');
fprintf(fid, '  ATE threshold: ≥ %.1f%%\n', high_benefit_threshold*100);
fprintf(fid, '  Number of scenarios: %d\n', high_benefit_count);

% High risk zone (bottom 25% of ATE values)
high_risk_threshold = prctile(valid_ate, 25);
high_risk_count = sum(valid_ate <= high_risk_threshold);

fprintf(fid, 'High risk zone (bottom 25%% of scenarios):\n');
fprintf(fid, '  ATE threshold: ≤ %.1f%%\n', high_risk_threshold*100);
fprintf(fid, '  Number of scenarios: %d\n', high_risk_count);

% Clinical decision making
fprintf(fid, '\nCLINICAL DECISION MAKING:\n');

% Safe treatment zone (consistently beneficial)
safe_threshold = 0.05;  % 5% benefit
safe_scenarios = sum(valid_ate > safe_threshold);

fprintf(fid, 'Treatment recommendations:\n');
fprintf(fid, '  Safe treatment zone (ATE > 5%%): %d scenarios (%.1f%%)\n', ...
    safe_scenarios, safe_scenarios/total_scenarios*100);

dangerous_threshold = -0.02;  % 2% harm
dangerous_scenarios = sum(valid_ate < dangerous_threshold);
fprintf(fid, '  Dangerous treatment zone (ATE < -2%%): %d scenarios (%.1f%%)\n', ...
    dangerous_scenarios, dangerous_scenarios/total_scenarios*100);

% Parameter estimation implications
fprintf(fid, '\nPARAMETER ESTIMATION IMPLICATIONS:\n');
fprintf(fid, 'For clinical implementation:\n');
if abs(disease_corr) > 0.5
    fprintf(fid, '- Accurate estimation of disease harm (a2) is CRITICAL\n');
else
    fprintf(fid, '- Accurate estimation of disease harm (a2) is IMPORTANT\n');
end
if abs(treatment_corr) > 0.5
    fprintf(fid, '- Accurate estimation of treatment harm (a3) is CRITICAL\n');
else
    fprintf(fid, '- Accurate estimation of treatment harm (a3) is IMPORTANT\n');
end

if beneficial_count/total_scenarios > 0.8
    fprintf(fid, '- Treatment is broadly beneficial across parameter space\n');
elseif harmful_count/total_scenarios > 0.3
    fprintf(fid, '- Treatment can be harmful - careful parameter estimation required\n');
else
    fprintf(fid, '- Treatment benefit is parameter-dependent\n');
end

% Robust treatment strategies
fprintf(fid, '\nROBUST TREATMENT STRATEGIES:\n');

% Find parameter combinations that are consistently beneficial
if length(A2) > 1 && length(A3) > 1
    robust_benefit_mask = ate_matrix > 0.02;  % 2% benefit threshold
    robust_regions = sum(robust_benefit_mask(:));
    
    fprintf(fid, 'Robust benefit regions (ATE > 2%% regardless of exact parameters):\n');
    fprintf(fid, '  Count: %d/%d scenarios (%.1f%%)\n', ...
        robust_regions, numel(ate_matrix), robust_regions/numel(ate_matrix)*100);
    
    if robust_regions > 0.5 * numel(ate_matrix)
        fprintf(fid, '  → Treatment is ROBUST across most parameter combinations\n');
    else
        fprintf(fid, '  → Treatment benefit is SENSITIVE to parameter values\n');
    end
end

% Practical recommendations
fprintf(fid, '\nPRACTICAL RECOMMENDATIONS:\n');

fprintf(fid, 'For clinical practice:\n');
if safe_scenarios/total_scenarios > 0.7
    fprintf(fid, '✓ Treatment is generally safe and beneficial\n');
    fprintf(fid, '  - Consider aggressive treatment protocols\n');
    fprintf(fid, '  - Standard parameter estimation may be sufficient\n');
else
    fprintf(fid, '⚠️  Treatment benefit is parameter-dependent\n');
    fprintf(fid, '  - Invest in accurate parameter estimation\n');
    fprintf(fid, '  - Consider adaptive treatment protocols\n');
    fprintf(fid, '  - Monitor for treatment-related harm\n');
end

fprintf(fid, '\nFor parameter estimation:\n');
fprintf(fid, '- Priority should be given to accurate estimation of:\n');
if abs(disease_corr) > abs(treatment_corr)
    fprintf(fid, '  1. Disease harm parameters (higher impact)\n');
    fprintf(fid, '  2. Treatment harm parameters\n');
else
    fprintf(fid, '  1. Treatment harm parameters (higher impact)\n');
    fprintf(fid, '  2. Disease harm parameters\n');
end

% Technical details
fprintf(fid, '\nTECHNICAL DETAILS:\n');
fprintf(fid, '- Analysis type: Grid search sensitivity analysis\n');
fprintf(fid, '- Grid resolution: 20 × 20 parameter combinations\n');
fprintf(fid, '- Mortality model: Logistic hazard with time-varying effects\n');
fprintf(fid, '- Treatment assignment: 50%% randomization (RCT simulation)\n');
fprintf(fid, '- Endpoint: Survival probability at 168 hours\n');
fprintf(fid, '- Aggregation: Median across %d trial(s) per parameter set\n', nTrials);

% Data availability
fprintf(fid, '\nDATA AVAILABILITY:\n');
fprintf(fid, '- Complete results saved to: HeatMapAggressive.mat\n');
fprintf(fid, '- Survival matrices: s0 (untreated), s1 (treated)\n');
fprintf(fid, '- Parameter grids: A2 (disease harm), A3 (treatment harm)\n');
fprintf(fid, '- Optimal thresholds: Th_opt (if computed)\n');

fclose(fid);
fprintf('Heatmap sensitivity analysis results exported to: %s\n', filename);
