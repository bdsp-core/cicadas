function T0_boot = fcn_bootstrapBySID(T0, N)
% BOOTSTRAPBYSID - Create bootstrap sample by resampling subject IDs
%
% Inputs:
%   T0 - Original data table with columns including 'sid' (subject ID)
%   N  - Number of subjects to sample (optional, defaults to unique subjects in T0)
%
% Output:
%   T0_boot - Bootstrap sample table with N subjects (sampled with replacement)
%             Subject IDs are reassigned as sequential integers 1, 2, ..., N

% Get unique subject IDs
unique_sids = unique(T0.sid);
n_subjects = length(unique_sids);

% If N not specified, use same number of subjects as original
if nargin < 2
    N = n_subjects;
end

% Sample N subject IDs with replacement
sampled_sids = unique_sids(randi(n_subjects, N, 1));

% Initialize empty table
T0_boot = table();

% For each sampled subject ID, add all their rows to the bootstrap table
for i = 1:N
    % Get all rows for this subject
    subject_rows = T0(T0.sid == sampled_sids(i), :);
    
    % Assign new sequential subject ID
    subject_rows.sid(:) = i;
    
    % Append to bootstrap table
    T0_boot = [T0_boot; subject_rows];
end

% Display summary
fprintf('Bootstrap sample created:\n');
fprintf('  Original subjects: %d\n', n_subjects);
fprintf('  Sampled subjects: %d\n', N);
fprintf('  Total rows in original: %d\n', height(T0));
fprintf('  Total rows in bootstrap: %d\n', height(T0_boot));
fprintf('  New subject IDs: 1 to %d\n', N);

% Show frequency of each subject in bootstrap sample
sid_counts = histcounts(sampled_sids, [unique_sids; max(unique_sids)+1]);
fprintf('\nSubject sampling frequencies:\n');
fprintf('  Not sampled: %d subjects\n', sum(sid_counts == 0));
fprintf('  Sampled once: %d subjects\n', sum(sid_counts == 1));
fprintf('  Sampled 2+ times: %d subjects\n', sum(sid_counts >= 2));

% Optional: Show mapping of old to new IDs for first few subjects
if N <= 10
    fprintf('\nID mapping (first %d subjects):\n', N);
    fprintf('  New ID -> Original ID\n');
    for i = 1:N
        fprintf('  %6d -> %d\n', i, sampled_sids(i));
    end
elseif N > 10
    fprintf('\nID mapping (first 5 subjects):\n');
    fprintf('  New ID -> Original ID\n');
    for i = 1:5
        fprintf('  %6d -> %d\n', i, sampled_sids(i));
    end
    fprintf('  ... (showing first 5 of %d)\n', N);
end

end