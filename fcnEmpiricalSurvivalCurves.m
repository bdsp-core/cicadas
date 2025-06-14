function [h_rx1, h_rx0, S_rx1, S_rx0, t_rx1, t_rx0] = fcnEmpiricalSurvivalCurves(T)
    % Estimates hazard functions and survival curves for treatment (Rx=1) and control (Rx=0) groups
    % T: Table with columns sid, t, Rx, harmE, harmA, b0, L, A, V, Y in LONG format
    
    % Get unique subject IDs
    subject_ids = unique(T.sid);
    n_subjects = length(subject_ids);
    
    % Initialize arrays
    survival_times = zeros(n_subjects, 1);
    is_death = zeros(n_subjects, 1);
    initial_treatment = zeros(n_subjects, 1);
    
    % Extract subject-level data from long format
    for i = 1:n_subjects
        % Get data for this subject
        subject_data = T(T.sid == subject_ids(i), :);
        
        % Sort by time
        subject_data = sortrows(subject_data, 't');
        
        % Get initial treatment (first observation)
        initial_treatment(i) = subject_data.Rx(1);
        
        % Check if death occurred (Y=1)
        death_idx = find(subject_data.Y > 0, 1, 'first');
        if ~isempty(death_idx)
            % Subject died - use death time
            survival_times(i) = subject_data.t(death_idx);
            is_death(i) = 1;
        else
            % Subject didn't die - use last observation time
            survival_times(i) = max(subject_data.t);
            is_death(i) = 0;
        end
    end
    
    % Split subjects by initial treatment assignment
    rx1_idx = initial_treatment == 1;
    rx0_idx = initial_treatment == 0;
    
    % Create time grid for evaluation (0 to max survival time)
    t_max = max(survival_times);
    t_grid = linspace(0, t_max, 100);
    
    % Calculate KM estimates for treatment group
    if any(rx1_idx)
        [S_rx1, t_rx1, h_rx1] = calculateKM(survival_times(rx1_idx), is_death(rx1_idx), t_grid, T, subject_ids(rx1_idx));
    else
        S_rx1 = ones(length(t_grid), 1);
        t_rx1 = t_grid';
        h_rx1 = zeros(length(t_grid), 1);
    end
    
    % Calculate KM estimates for control group
    if any(rx0_idx)
        [S_rx0, t_rx0, h_rx0] = calculateKM(survival_times(rx0_idx), is_death(rx0_idx), t_grid, T, subject_ids(rx0_idx));
    else
        S_rx0 = ones(length(t_grid), 1);
        t_rx0 = t_grid';
        h_rx0 = zeros(length(t_grid), 1);
    end
end

function [S, t, h] = calculateKM(times, events, t_grid, T, subject_ids_in_group)
    % Calculate Kaplan-Meier estimates using standard algorithm
    
    if isempty(times) || all(events == 0)
        % No data or no deaths
        S = ones(length(t_grid), 1);
        t = t_grid';
        h = zeros(length(t_grid), 1);
        return;
    end
    
    % Sort by time
    [sorted_times, idx] = sort(times);
    sorted_events = events(idx);
    
    % Find unique death times
    death_times = sorted_times(sorted_events == 1);
    [unique_death_times, ~, death_idx] = unique(death_times);
    
    if isempty(unique_death_times)
        S = ones(length(t_grid), 1);
        t = t_grid';
        h = zeros(length(t_grid), 1);
        return;
    end
    
    % Count deaths at each unique time
    n_deaths = accumarray(death_idx, 1);
    
    % Calculate survival function at death times
    S_at_deaths = ones(length(unique_death_times) + 1, 1);
    
    for i = 1:length(unique_death_times)
        death_time = unique_death_times(i);
        n_at_risk = sum(sorted_times >= death_time);
        deaths_at_time = n_deaths(i);
        
        if n_at_risk > 0
            S_at_deaths(i+1) = S_at_deaths(i) * (1 - deaths_at_time/n_at_risk);
        else
            S_at_deaths(i+1) = S_at_deaths(i);
        end
    end
    
    % Interpolate to grid points (step function)
    time_points = [0; unique_death_times];
    S = zeros(length(t_grid), 1);
    
    for i = 1:length(t_grid)
        % Find last death time <= current time
        last_idx = find(time_points <= t_grid(i), 1, 'last');
        if ~isempty(last_idx)
            S(i) = S_at_deaths(last_idx);
        else
            S(i) = 1;  % Before first death
        end
    end
    
    % Estimate hazard from KM curve using smoothing spline approach
    % h(t) = -d/dt log(S(t))
    h = zeros(length(t_grid), 1);
    
    if length(S) > 10
        fprintf('Estimating hazard from smoothed Kaplan-Meier curve...\n');
        
        % Remove any S=0 points (can't take log)
        valid_idx = S > 1e-6;
        if sum(valid_idx) < 5
            fprintf('Too few valid survival points for hazard estimation\n');
            return;
        end
        
        t_valid = t_grid(valid_idx);
        S_valid = S(valid_idx);
        log_S_valid = log(S_valid);
        
        % Fit smoothing spline with moderate smoothing parameter
        try
            % Use a smoothing parameter to control smoothness (0=interpolating, 1=very smooth)
            smoothing_param = 0.1;  % Moderate smoothing
            
            % Create a smoothing spline (requires Curve Fitting Toolbox)
            if exist('csaps', 'file')
                % Use smoothing spline from Curve Fitting Toolbox
                pp = csaps(t_valid, log_S_valid, smoothing_param);
                
                % Evaluate the spline and its derivative
                log_S_smooth = ppval(pp, t_grid);
                pp_deriv = fnder(pp);
                dlogS_dt = ppval(pp_deriv, t_grid);
                
                h = -dlogS_dt;
                h = max(h, 0);
                
                fprintf('Hazard estimated using smoothing spline (p=%.2f)\n', smoothing_param);
            else
                % Fallback: Use polynomial fitting with lower order
                if length(t_valid) > 6
                    poly_order = min(3, length(t_valid)-2);  % Lower order polynomial
                else
                    poly_order = 1;
                end
                
                % Fit polynomial to log(S(t))
                p = polyfit(t_valid, log_S_valid, poly_order);
                log_S_smooth = polyval(p, t_grid);
                
                % Analytical derivative
                p_deriv = polyder(p);
                dlogS_dt = polyval(p_deriv, t_grid);
                
                h = -dlogS_dt;
                h = max(h, 0);
                
                fprintf('Hazard estimated using polynomial fit (order %d)\n', poly_order);
            end
            
        catch ME
            fprintf('Spline fitting failed (%s), using simple smoothing\n', ME.message);
            
            % Fallback: simple smoothing + finite differences
            log_S = log(max(S, 1e-10));
            
            % Apply moving average smoothing
            window = min(5, floor(length(log_S)/5));
            if window >= 3
                log_S_smooth = log_S;
                for i = window:length(log_S)-window+1
                    log_S_smooth(i) = mean(log_S(i-window+1:i+window-1));
                end
            else
                log_S_smooth = log_S;
            end
            
            % Finite differences
            dt = t_grid(2) - t_grid(1);
            for i = 2:length(t_grid)-1
                h(i) = -(log_S_smooth(i+1) - log_S_smooth(i-1)) / (2*dt);
                h(i) = max(h(i), 0);
            end
            
            % Handle endpoints
            if length(h) > 2
                h(1) = max(h(2), 0);
                h(end) = max(h(end-1), 0);
            end
        end
        
    else
        fprintf('Insufficient data points (%d) for hazard estimation\n', length(S));
    end
    
    % Return time grid as column vector
    t = t_grid';
end

