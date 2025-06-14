% Very simple test
fprintf('Testing basic MATLAB operations...\n');

% Test basic operations
x = [1, 2, 3];
y = ones(3, 1);
z = zeros(3, 1);

fprintf('Basic operations work\n');

% Test table creation
T = table([1; 2], [0; 1], 'VariableNames', {'sid', 't'});
fprintf('Table creation works\n');

% Test unique function
unique_ids = unique(T.sid);
fprintf('Unique function works, found %d unique IDs\n', length(unique_ids));

fprintf('Simple test complete\n');