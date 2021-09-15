function [problem, labels, weights, alpha, nearest_neighbors, similarities] = ...
  load_data(data_name, data_dir, exp)

max_k = 500;
if ~exist('data_dir', 'var')
  data_dir = './data';
end

switch data_name
case {'toy_problem0', 'toy_problem1'}
    % toy 2D grid problem with deterministic(toy_problem0) or
    % probabilistic(toy_problem1) labels
    data_dir = fullfile(data_dir, 'toy_problem');
    data_path = fullfile(data_dir, data_name);
    load(data_path(1:end-1));
    % prior probabilities of the two classes
    alpha               = [0.1 0.9];

    nn_file  = sprintf('%s_nearest_neighbors.mat', data_name(1:end-1));
    filename = fullfile(data_dir, nn_file);

    if exist(filename, 'file')
      load(filename);
    else
      [nearest_neighbors, distances] = ...
        knnsearch(problem.points, problem.points, ...
        'k', max_k + 1);
      save(filename, 'nearest_neighbors', 'distances');
    end
    k = 50;
    nearest_neighbors = nearest_neighbors(:, 2:(k + 1))';
    distances = distances(:, 2:(k + 1))';
    similarities = 1./distances; %exp(-distances.^2/2);

    % precompute sparse weight matrix
    num_points = problem.num_points;
    row_index = kron((1:num_points)', ones(k, 1));
    weights = sparse(row_index, nearest_neighbors(:), similarities(:), ...
      num_points, num_points);

    labels = 2*ones(problem.num_points, 1);
    rand_label = (data_name(end) == '1');
    if rand_label
      labels(labels_random)        = 1;
    else
      labels(labels_deterministic) = 1;
    end

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
end

problem.max_num_influence = max(sum(weights > 0, 1));  % used for pruning
