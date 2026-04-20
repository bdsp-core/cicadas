function run_all_verify()
% RUN_ALL_VERIFY  Wrapper that runs the full pipeline and snapshots outputs
% into matlab_outputs/ for side-by-side comparison against the Python port.
%
% Does NOT modify any of the authoritative a*.m scripts; they already set
% rng(0) internally so the pipeline is reproducible.
%
% Invoke from the cicadas/ directory:
%     >> run('run_all_verify')

repo_root = fileparts(mfilename('fullpath'));
cd(repo_root);

output_dir = fullfile(repo_root, 'matlab_outputs');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
fig_out = fullfile(output_dir, 'figures');
if ~exist(fig_out, 'dir'), mkdir(fig_out); end

fprintf('[run_all_verify] Starting at %s\n', datestr(now));
t_all = tic;

% Snapshot file list BEFORE running (so we know what got created/updated).
before = list_outputs(repo_root);

try
    run(fullfile(repo_root, 'matlab', 'run_all.m'));
catch ME
    fprintf('[run_all_verify] run_all.m FAILED: %s\n', ME.message);
    fprintf('%s\n', getReport(ME, 'extended'));
    rethrow(ME);
end

% Copy every .mat, .csv, .txt output from the repo root to matlab_outputs/.
patterns = {'*.mat', 'trialData*.csv', '*.txt'};
for i = 1:numel(patterns)
    files = dir(fullfile(repo_root, patterns{i}));
    for j = 1:numel(files)
        src = fullfile(files(j).folder, files(j).name);
        dst = fullfile(output_dir, files(j).name);
        copyfile(src, dst);
    end
end

% Copy figure PDFs + any .mat artifacts produced by figure scripts.
fig_dir = fullfile(repo_root, 'CICADA_FIGURES');
fig_patterns = {'Fig*.pdf', '*.mat'};
for i = 1:numel(fig_patterns)
    files = dir(fullfile(fig_dir, fig_patterns{i}));
    for j = 1:numel(files)
        src = fullfile(files(j).folder, files(j).name);
        dst = fullfile(fig_out, files(j).name);
        copyfile(src, dst);
    end
end

fprintf('[run_all_verify] Finished in %.1f min. Outputs in %s\n', toc(t_all)/60, output_dir);

% Write a manifest of what we captured.
manifest_file = fullfile(output_dir, 'MANIFEST.txt');
fid = fopen(manifest_file, 'w');
fprintf(fid, 'MATLAB reference run captured at %s\n', datestr(now));
fprintf(fid, 'Seed: rng(0) set internally by a0/a4 scripts\n\n');
all_files = dir(output_dir);
for k = 1:numel(all_files)
    if ~all_files(k).isdir
        fprintf(fid, '%-60s  %10d bytes\n', all_files(k).name, all_files(k).bytes);
    end
end
fig_files = dir(fig_out);
for k = 1:numel(fig_files)
    if ~fig_files(k).isdir
        fprintf(fid, 'figures/%-52s  %10d bytes\n', fig_files(k).name, fig_files(k).bytes);
    end
end
fclose(fid);
end


function files = list_outputs(root)
    files = containers.Map;
    for ext = {'.mat', '.csv', '.txt', '.pdf'}
        d = dir(fullfile(root, ['*', ext{1}]));
        for j = 1:numel(d)
            files(d(j).name) = d(j).bytes;
        end
    end
end
