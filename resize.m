clc; clear; close all;

% -------------------- SETUP ---------------------
% Input folders
metal_folder = 'D:\ct data\sinogram\t3 sinogram';
no_metal_folder = 'D:\ct data\sinogram\t3 no metal sinogram';
mask_folder = 'D:\ct data\t3 masks';

% Output folders
output_base = 'D:\ct data\t3 resized_dataset';
output_metal = fullfile(output_base, 't3 resized_metal');
output_no_metal = fullfile(output_base, 't3 resized_no_metal');
output_mask_with = fullfile(output_base, 't3 resized_mask_with_metal');
output_mask_no = fullfile(output_base, 't3 resized_mask_no_metal');
output_mask_diff = fullfile(output_base, 't3 resized_mask_difference');

% Create folders if not exist
mkdir(output_metal);
mkdir(output_no_metal);
mkdir(output_mask_with);
mkdir(output_mask_no);
mkdir(output_mask_diff);

% Desired output size
desired_size = [256, 256];

% -------------------- READ FILES ---------------------
metal_files = dir(fullfile(metal_folder, '*.png'));
no_metal_files = dir(fullfile(no_metal_folder, '*.png'));
mask_with_files = dir(fullfile(mask_folder, '*_with_metal_mask.png'));
mask_no_files = dir(fullfile(mask_folder, '*_no_metal_mask.png'));
mask_diff_files = dir(fullfile(mask_folder, '*_difference_mask.png'));

% Extract IDs (e.g., p-100)
get_id = @(name) regexp(name, 'p-\d+', 'match', 'once');
metal_ids = cellfun(get_id, {metal_files.name}, 'UniformOutput', false);
no_metal_ids = cellfun(get_id, {no_metal_files.name}, 'UniformOutput', false);
mask_with_ids = cellfun(get_id, {mask_with_files.name}, 'UniformOutput', false);
mask_no_ids = cellfun(get_id, {mask_no_files.name}, 'UniformOutput', false);
mask_diff_ids = cellfun(get_id, {mask_diff_files.name}, 'UniformOutput', false);

% Match IDs present in all 5 categories
common_ids = intersect(intersect(intersect(intersect(metal_ids, no_metal_ids), mask_with_ids), mask_no_ids), mask_diff_ids);
fprintf('✅ Found %d matched samples with all masks.\n', numel(common_ids));

% -------------------- PROCESS & SAVE ---------------------
for i = 1:numel(common_ids)
    id = common_ids{i};

    try
        % Filenames
        metal_name = id + "_sinogram.png";
        no_metal_name = id + "_sinogram.png";
        mask_with_name = id + "_with_metal_mask.png";
        mask_no_name = id + "_no_metal_mask.png";
        mask_diff_name = id + "_difference_mask.png";

        % Full paths
        metal_path = fullfile(metal_folder, metal_name);
        no_metal_path = fullfile(no_metal_folder, no_metal_name);
        mask_with_path = fullfile(mask_folder, mask_with_name);
        mask_no_path = fullfile(mask_folder, mask_no_name);
        mask_diff_path = fullfile(mask_folder, mask_diff_name);

        % Read and resize
        metal_img = imresize(im2double(imread(metal_path)), desired_size);
        no_metal_img = imresize(im2double(imread(no_metal_path)), desired_size);
        mask_with = imbinarize(imresize(im2double(imread(mask_with_path)), desired_size));
        mask_no = imbinarize(imresize(im2double(imread(mask_no_path)), desired_size));
        mask_diff = imbinarize(imresize(im2double(imread(mask_diff_path)), desired_size));

        % Save all resized images
        imwrite(metal_img, fullfile(output_metal, metal_name));
        imwrite(no_metal_img, fullfile(output_no_metal, no_metal_name));
        imwrite(mask_with, fullfile(output_mask_with, mask_with_name));
        imwrite(mask_no, fullfile(output_mask_no, mask_no_name));
        imwrite(mask_diff, fullfile(output_mask_diff, mask_diff_name));

        fprintf('✅ Saved: %s\n', id);

    catch ME
        fprintf('❌ Error processing %s: %s\n', id, ME.message);
        continue;
    end
end

fprintf('\n🎉 All 5 image types processed and saved successfully.\n');
