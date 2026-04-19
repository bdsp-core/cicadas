function p = fcnBiasedAssignmentProb(age, sofa, L_early)
    % age: n_patients x 1 vector
    % sofa: n_patients x 1 vector  
    % L_early: n_patients x 5 matrix (first 5 time points of L)
    
    n_patients = length(age);
    p = zeros(n_patients, 1);
    
    % Process each patient
    for i = 1:n_patients
        % Get this patient's early L trajectory
        L_traj = L_early(i, :);
        
        % Normalize age and sofa
        age_norm = (age(i) - 18) / (90 - 18);
        sofa_norm = sofa(i) / 24;
        
        % Calculate metrics for this patient
        L_initial = L_traj(1);
        L_final = L_traj(end);
        L_max = max(L_traj);
        L_mean = mean(L_traj);
        
        % Relative growth with epsilon
        epsilon = 0.001;
        L_relative_growth = (L_final - L_initial) / (L_initial + epsilon);
        
        % Absolute slope
        L_slope = (L_final - L_initial) / length(L_traj);
        
        % Acceleration
        mid_point = ceil(length(L_traj) / 2);  % Dynamically set midpoint
        if length(L_traj) >= 3
            first_half_slope = (L_traj(mid_point) - L_traj(1)) / (mid_point - 1);
            second_half_slope = (L_traj(end) - L_traj(mid_point)) / (length(L_traj) - mid_point);
            acceleration = second_half_slope - first_half_slope;
        else
            acceleration = 0;  % Not enough points to calculate acceleration
        end
        
        % REVISED NORMALIZATION - scale to create more variation
        L_mean_norm = L_mean * 20;              % If L_mean ~ 0.05, this gives 1.0
        L_max_norm = L_max * 10;                % If L_max ~ 0.1, this gives 1.0
        L_relative_growth_norm = tanh(L_relative_growth / 2);  % More sensitive
        L_slope_norm = L_slope * 100;           % If slope ~ 0.01, this gives 1.0
        acceleration_norm = acceleration * 200;  % If accel ~ 0.005, this gives 1.0
        
        % MODERATE COEFFICIENTS for realistic bias
        L_mean_effect = 0.8;           
        L_max_effect = 0.6;            
        L_relative_growth_effect = 1.2; 
        L_slope_effect = 1.0;          
        acceleration_effect = 0.8;      
        sofa_effect = 0.5;   
        age_effect = -0.3;   
        
        % MODERATE INTERCEPT for ~50% baseline probability
        intercept = -0.5;    
        
        % Calculate log-odds
        z = intercept + ...
            L_mean_effect * L_mean_norm + ...
            L_max_effect * L_max_norm + ...
            L_relative_growth_effect * L_relative_growth_norm + ...
            L_slope_effect * L_slope_norm + ...
            acceleration_effect * max(0, acceleration_norm) + ...
            sofa_effect * sofa_norm + ...
            age_effect * age_norm;
        
        % Add some random noise to create variation
        z = z + 0.3 * randn();  % Larger random component for more variation
        
        % Calculate probability
        p(i) = 1 / (1 + exp(-z));
    end
    
    % Ensure reasonable bounds
    p = min(max(p, 0.05), 0.95);
    
    % Debug output to check distribution
    fprintf('Treatment assignment probabilities - Mean: %.3f, Std: %.3f, Min: %.3f, Max: %.3f\n', ...
        mean(p), std(p), min(p), max(p));
end