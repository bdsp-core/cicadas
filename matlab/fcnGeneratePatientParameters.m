function [age, sofa, C, g, parmsPD] = fcnGeneratePatientParameters(N, varargin)

    % Parse inputs
    p = inputParser;
    addRequired(p, 'N', @(x) isnumeric(x) && isscalar(x) && x > 0 && x == floor(x));
    addParameter(p, 'TargetCMean', 3, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'TargetGMean', 4, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'CV', 0.1, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
    addParameter(p, 'AgeRange', [18 90], @(x) isnumeric(x) && length(x) == 2 && x(1) < x(2));
    addParameter(p, 'SofaRange', [0 24], @(x) isnumeric(x) && length(x) == 2 && x(1) <= x(2));
    addParameter(p, 'RandomSeed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
    addParameter(p, 'MinValue', 0.1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    
    parse(p, N, varargin{:});
    
    % Extract parameters
    target_C_mean = p.Results.TargetCMean;
    target_g_mean = p.Results.TargetGMean;
    cv = p.Results.CV;
    age_range = p.Results.AgeRange;
    sofa_range = p.Results.SofaRange;
    random_seed = p.Results.RandomSeed;
    min_val = p.Results.MinValue;
    
    % Set random seed if provided
    if ~isempty(random_seed)
        rng(random_seed);
    end
    
    % Generate age and SOFA score data
    age = age_range(1) + (age_range(2) - age_range(1)) * rand(N, 1);
    sofa = sofa_range(1) + (sofa_range(2) - sofa_range(1)) * rand(N, 1);
    
    % Calculate standard deviations for desired CV
    sigma_C = cv * target_C_mean;
    sigma_g = cv * target_g_mean;
    
    % Normalize predictors
    age_norm = (age - mean(age)) / 20.4;
    sofa_norm = (sofa - mean(sofa)) / 6.8468;
    
    % Define regression coefficients
    % Intercepts set to target means
    b0_C = target_C_mean;
    b1_C = 0.1;  % age coefficient for C
    b2_C = 0.15; % sofa coefficient for C
    
    b0_g = target_g_mean;
    b1_g = 0.08; % age coefficient for g
    b2_g = 0.12; % sofa coefficient for g
    
    % Generate C values
    noise_C = sigma_C * randn(N, 1);
    C_linear = b0_C + b1_C * age_norm + b2_C * sofa_norm;
    C = C_linear + noise_C;
    
    % Generate g values
    noise_g = sigma_g * randn(N, 1);
    g_linear = b0_g + b1_g * age_norm + b2_g * sofa_norm;
    g = g_linear + noise_g;
    
    % Ensure all values are positive
    C(C < min_val) = min_val;
    g(g < min_val) = min_val;
    
    % Return PD parameters vector
    parmsPD = [b0_C, b1_C, b2_C, b0_g, b1_g, b2_g];
   
    
end
