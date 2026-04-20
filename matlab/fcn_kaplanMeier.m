function [t_km, s_km, se_km, n_risk] = kaplanMeier(event_times, event_indicators)
% KAPLANMEIER Calculate Kaplan-Meier survival estimates
%
% Inputs:
%   event_times     - Vector of event/censoring times
%   event_indicators - Vector of event indicators (1=event, 0=censored)
%
% Outputs:
%   t_km    - Unique event times
%   s_km    - Survival probability estimates at each event time
%   se_km   - Standard errors (Greenwood's formula)
%   n_risk  - Number at risk at each event time
%
% Reference: Klein & Moeschberger (2003), Survival Analysis

% Remove any NaN or missing values
valid_idx = ~isnan(event_times) & ~isnan(event_indicators);
event_times = event_times(valid_idx);
event_indicators = event_indicators(valid_idx);

% Sort by time
[event_times, sort_idx] = sort(event_times);
event_indicators = event_indicators(sort_idx);

% Get unique event times (where actual events occurred)
unique_times = unique(event_times(event_indicators == 1));

% Initialize outputs
n_times = length(unique_times);
t_km = unique_times;
s_km = ones(n_times, 1);
se_km = zeros(n_times, 1);
n_risk = zeros(n_times, 1);

% Calculate KM estimates
survival_prob = 1;
variance_sum = 0;

for i = 1:n_times
    t = unique_times(i);
    
    % Number at risk at time t
    n_at_risk = sum(event_times >= t);
    n_risk(i) = n_at_risk;
    
    % Number of events at time t
    n_events = sum(event_times == t & event_indicators == 1);
    
    % Update survival probability
    if n_at_risk > 0
        survival_prob = survival_prob * (1 - n_events / n_at_risk);
        s_km(i) = survival_prob;
        
        % Update variance sum for Greenwood's formula
        if n_events > 0
            variance_sum = variance_sum + n_events / (n_at_risk * (n_at_risk - n_events));
        end
        
        % Calculate standard error
        if survival_prob > 0
            se_km(i) = survival_prob * sqrt(variance_sum);
        else
            se_km(i) = 0;
        end
    end
end

% Ensure column vectors
t_km = t_km(:);
s_km = s_km(:);
se_km = se_km(:);
n_risk = n_risk(:);

end