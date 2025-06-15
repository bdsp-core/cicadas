% Test the fixed function with minimal data
clear; clc;

% Create minimal test data
T = table([1; 1; 2; 2], [0; 1; 0; 1], [1; 1; 0; 0], [10; 10; 20; 20], [5; 5; 15; 15], ...
          [0.1; 0.1; 0.1; 0.1], [1; 2; 1; 2], [0; 1; 0; 1], [0; 0; 0; 0], [0; 1; 0; 0], ...
          'VariableNames', {'sid', 't', 'Rx', 'harmE', 'harmA', 'b0', 'L', 'A', 'V', 'Y'});

fprintf('Testing fcnEmpiricalSurvivalCurves...\n');

try
    [h_rx1, h_rx0, S_rx1, S_rx0, t_rx1, t_rx0] = fcnEmpiricalSurvivalCurves(T);
    fprintf('SUCCESS: Function completed\n');
    fprintf('h_rx1 length: %d, h_rx0 length: %d\n', length(h_rx1), length(h_rx0));
catch ME
    fprintf('ERROR: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('Location: %s line %d\n', ME.stack(1).file, ME.stack(1).line);
    end
end

fprintf('Test complete\n');