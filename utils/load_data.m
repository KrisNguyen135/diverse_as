function [problem, labels, weights, alpha, nearest_neighbors, similarities] = ...
  load_data(data_name, data_dir, exp, group_size)

max_k = 500;
if ~exist('data_dir', 'var'), data_dir = './data'; end

switch data_name
case 'square'
    load(fullfile(data_dir, sprintf('square/square_nearest_neighbors_%d.mat', exp)));
    k     = size(similarities, 2);
    alpha = [0.6 0.2 0.05 0.05 0.05 0.05];
    % alpha = [0.75 0.13 0.03 0.03 0.03 0.03];

    nearest_neighbors = double(nearest_neighbors)';
    similarities      = double(similarities)';

    num_points = size(x, 1);
    row_index  = kron((1:num_points)', ones(k, 1));
    weights    = sparse(row_index, nearest_neighbors(:), similarities(:), ...
                        num_points, num_points);

    problem.num_classes = 6;
    problem.points      = x;
    problem.num_points  = num_points;

case 'square_small'
    load(fullfile(data_dir, 'square_small/square_small_nearest_neighbors.mat'));
    k     = size(similarities, 2);
    alpha = [0.6 0.2 0.05 0.05 0.05 0.05];
    % alpha = [0.75 0.13 0.03 0.03 0.03 0.03];

    nearest_neighbors = double(nearest_neighbors)';
    similarities      = double(similarities)';

    num_points = size(x, 1);
    row_index  = kron((1:num_points)', ones(k, 1));
    weights    = sparse(row_index, nearest_neighbors(:), similarities(:), ...
                        num_points, num_points);

    problem.num_classes = 6;
    problem.points      = x;
    problem.num_points  = num_points;

case 'citeseer'
    targets = [3 6 22 35];  % NeurIPS, ICML, UAI, JMLR
    % targets = [3 6];  % NeurIPS, ICML

    data_dir  = fullfile(data_dir, 'citeseer');
    data_path = fullfile(data_dir, 'citeseer_data');
    load(data_path);

    alpha              = [1 0.05 * ones(size(targets))];
    num_points         = size(x, 1);
    problem.points     = (1:num_points)';
    problem.num_points = num_points;

    filename = fullfile(data_dir, 'citeseer_data_nearest_neighbors.mat');
    if exist(filename, 'file')
        load(filename);
    else
        [nearest_neighbors, distances] = knnsearch(x, x, 'k', max_k + 1);
        save(filename, 'nearest_neighbors', 'distances');
    end

    %% there are duplicates in the data
    % e.g. nearest_neighbors(160, 1:2) = [18, 160]
    % that means x(160,:) and x(18,:) are identical
    for i = 1:num_points
        if nearest_neighbors(i, 1) ~= i
            dup_idx = find(nearest_neighbors(i, 2:end) == i);
            nearest_neighbors(i, 1+dup_idx) = nearest_neighbors(i, 1);
            nearest_neighbors(i, 1) = i;
        end
    end

    % limit to only top k
    k = 50;
    nearest_neighbors = nearest_neighbors(:, 2:(k + 1))';
    similarities      = ones(size(nearest_neighbors));

    % precompute sparse weight matrix
    row_index = kron((1:num_points)', ones(k, 1));
    weights   = sparse(row_index, nearest_neighbors(:), 1, num_points, num_points);

    % create label vector
    % labels = 2 * ones(size(x, 1), 1);
    % labels(connected_labels == 3) = 1;
    % problem.num_classes = 2;
    labels = ones(size(x, 1), 1);
    for target_ind = 1:size(targets, 2)
        labels(connected_labels == targets(target_ind)) = target_ind + 1;
    end

    problem.num_classes = 1 + size(targets, 2);
otherwise   % drug discovery
    % this needs to be the same as in
    % `../active_virtual_screening/calculate_nearest_neighbors_multiclass`
    if ~exist('group_size', 'var'), group_size = 4; end

    alpha        = [1 0.001 * ones(1, group_size)];
    num_inactive = 100000;

    if contains(data_name, 'ecfp')
        filename = sprintf('group_%s_ecfp4_nearest_neighbors_%d.mat', ...
                           data_name(5:end), num_inactive);
        fingerprint = 'ecfp4';
    elseif contains(data_name, 'gpidaph')
        filename = sprintf('group_%s_gpidaph3_nearest_neighbors_%d.mat', ...
                           data_name(8:end), num_inactive);
        fingerprint = 'gpidaph3';
    else
        error(sprintf('unrecognized data name %s\n', data_name));
    end

    data_dir  = fullfile(data_dir, 'drug/precomputed', sprintf('%d', group_size));
    data_path = fullfile(data_dir, filename);
    load(data_path);

    labels     = this_labels;
    num_points = numel(labels);

    problem.num_points  = num_points;
    problem.points      = (1:num_points)';
    problem.num_classes = group_size + 1;

    % limit to k-nearest neighbors
    k = 100;
    nearest_neighbors = nearest_neighbors(:, 1:k)';
    similarities      = similarities(:, 1:k)';

    % precompute sparse weight matrix
    row_index = kron((1:num_points)', ones(k, 1));
    weights = sparse(row_index, nearest_neighbors(:), similarities(:), ...
                     num_points, num_points);

end

problem.max_num_influence = max(sum(weights > 0, 1));  % used for pruning
