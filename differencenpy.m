clc; clear; close all;

% Add path to NPY read/write functions
addpath('path_to_npy_toolbox');  % <- 🔁 CHANGE this to actual path where readNPY.m is

% Select metal and no-metal folders
metal_folder = uigetdir(pwd, 'Select folder: WITH METAL sinograms (.npy)');
no_metal_folder = uigetdir(pwd, 'Select folder: WITHOUT METAL sinograms (.npy)');

% Output folder for saving comparison outputs
output_folder = fullfile('D:\', 'sinogram_differences_npy');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Get list of all *_sinogram.npy files from metal folder
metal_files = dir(fullfile(metal_folder, '*_sinogram.npy'));
fprintf(' Found %d sinogram .npy files in metal folder.\n', length(metal_files));

% Loop through each file
for k = 1:length(metal_files)
    fname = metal_files(k).name;
    fprintf('\n Processing file: %s\n', fname);

    metal_path = fullfile(metal_folder, fname);
    no_metal_path = fullfile(no_metal_folder, fname);

    % Check if corresponding no-metal file exists
    if ~isfile(no_metal_path)
        warning(' No matching no-metal file for: %s — skipping.', fname);
        continue;
    end

    try
        % Load metal and no-metal sinograms
        metal_sino = readNPY(metal_path);
        no_metal_sino = readNPY(no_metal_path);
        diff_sino = abs(metal_sino - no_metal_sino);
    catch ME
        warning(' Error reading NPY files: %s\n%s', fname, ME.message);
        continue;
    end

    % Remove '_sinogram.npy' to get base name
    base_name = erase(fname, '_sinogram.npy');

    try
        % --- Save metal heatmap ---
        fig1 = figure('Visible','off');
        imagesc(metal_sino); axis off image; colormap('hot'); colorbar;
        exportgraphics(fig1, fullfile(output_folder, [base_name '_with_metal.png']), 'Resolution', 300);
        close(fig1);

        % --- Save no-metal heatmap ---
        fig2 = figure('Visible','off');
        imagesc(no_metal_sino); axis off image; colormap('hot'); colorbar;
        exportgraphics(fig2, fullfile(output_folder, [base_name '_no_metal.png']), 'Resolution', 300);
        close(fig2);

        % --- Save difference heatmap ---
        fig3 = figure('Visible','off');
        imagesc(diff_sino); axis off image; colormap('jet'); colorbar;
        exportgraphics(fig3, fullfile(output_folder, [base_name '_difference.png']), 'Resolution', 300);
        close(fig3);

        % --- Save difference array as .npy ---
        writeNPY(diff_sino, fullfile(output_folder, [base_name '_difference.npy']));

        fprintf('✅ [%d/%d] Saved PNGs and .npy for: %s\n', k, length(metal_files), base_name);
    catch ME
        warning('❌ Saving failed for %s:\n%s', base_name, ME.message);
    end
end

fprintf('\n🎉 All done! Files saved in:\n%s\n', output_folder);
