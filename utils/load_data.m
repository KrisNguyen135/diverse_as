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

    if group_size == 1
        labels(labels > 1) = 2;
        alpha = [0.6 0.4];
        problem.num_classes = 2;
    end

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
    if group_size == 1
        targets = [6];          % ICML
    elseif group_size == 4
        targets = [3 6 22 35];  % NeurIPS, ICML, UAI, JMLR
    else
        targets = [2 3 6 22 35 39];  % AAAI, NeurIPS, ICML, UAI, JMLR, ML
    end

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

case 'bmg'
    assert(group_size == 1, 'bmg is only for single-class experiments');

    if ~strcmp(data_dir, '../data/')
        data_dir = '/storage1/garnett/Active/activelearning/datasets/';
    end

    data_dir  = fullfile(data_dir, 'bmg');
    data_path = fullfile(data_dir, 'bmg_data');
    load(data_path);
    % remove labels from features
    x = bmg_data(:, 1:(end - 1));

    % create label vector
    labels = ones(size(x, 1), 1);
    labels(bmg_data(:, end) <= 1) = 2;

    % remove rows with nans
    ind    = (~any(isnan(x), 2));
    x      = x(ind, :);
    labels = labels(ind);

    num_points = size(x, 1);

    train_portion = 0.1;
    rng('default');
    train_ind = crossvalind('holdout', num_points, 1 - train_portion);

    % can be reproduced above
    ind = [1, 33, 39, 45, 46, 53, 111, 135, 141, 165, 185, 200, 201];

    % limit features to those selected
    x = x(~train_ind, ind);
    %     x = x(:, ind);
    num_points = size(x, 1);
    labels     = labels(~train_ind);

    % remove features with no variance
    x = x(:, std(x) ~= 0);

    % normalize data
    x = bsxfun(@minus, x,     mean(x));
    x = bsxfun(@times, x, 1 ./ std(x));

    problem.points      = x;
    problem.num_classes = 2;
    problem.num_points  = num_points;

    % filename = fullfile(data_dir, 'bmg_nearest_neighbors.mat');
    % filename = fullfile(data_dir, 'bmg_nearest_neigbors.mat');
    filename = fullfile(data_dir, 'new_bmg_nearest_neigbors.mat');

    if exist(filename, 'file')
      load(filename, 'nearest_neighbors', 'distances');
    else
      [nearest_neighbors, distances] = ...
        knnsearch(problem.points, problem.points, ...
        'k', max_k + 1);

      % deal with a small number of ties in dataset
      for i = 1:num_points
        if (nearest_neighbors(i, 1) ~= i)
          ind = find(nearest_neighbors(i, :) == i);
          nearest_neighbors(i, ind) = nearest_neighbors(i, 1);
          nearest_neighbors(i, 1)   = i;
        end
      end

      save(filename, 'nearest_neighbors', 'distances');
    end

    % limit to only top k
    k = 50;
    nearest_neighbors = nearest_neighbors(:, 2:(k + 1))';
    similarities = ones(size(nearest_neighbors));
    % precompute sparse weight matrix
    row_index = kron((1:num_points)', ones(k, 1));
    weights = sparse(row_index, nearest_neighbors(:), 1, ...
      num_points, num_points);

    alpha = [1, 0.05];

% otherwise  % drug discovery with pruned data
%     if ~exist('group_size', 'var'), group_size = 1; end
%         alpha = [1 0.001 * ones(1, group_size)];
%
%     assert(contains(data_name, 'morgan'), 'only morgan fingerprint is currently supported');
%
%     filename  = 'morgan_nearest_neighbors_100000.mat';
%     group_ind = str2num(data_name(7:end));
%
%     data_dir  = fullfile(data_dir, 'drug/precomputed');
%     data_path = fullfile(data_dir, filename);
%     load(data_path);
%
%     num_points          = numel(labels);
%     problem.num_points  = num_points;
%     problem.points      = (1:num_points)';
%     problem.num_classes = group_size + 1;
%
%     % limit to k-nearest neighbors
%     k = 100;
%     nearest_neighbors = nearest_neighbors(:, 1:k)';
%     similarities      = similarities(:, 1:k)';
%
%     % precompute sparse weight matrix
%     row_index = kron((1:num_points)', ones(k, 1));
%     weights = sparse(row_index, nearest_neighbors(:), similarities(:), ...
%                      num_points, num_points);
%
%     % relabel
%     if group_size == 1  % just use the same class in the single-class case
%         pos_mask         = (labels == group_ind + 1);
%         labels(:)        = 1;
%         labels(pos_mask) = 2;
%     else  % randomly pick out `group_size` classes
%         rng(group_ind);
%
%         selected_classes = randperm(64, group_size);
%         selected_classes
%
%         old_labels = labels;
%
%         labels(:) = 1;
%         for class_ind = 1:group_size
%             pos_mask         = (old_labels == selected_classes(class_ind) + 1);
%             labels(pos_mask) = class_ind + 1;
%         end
%     end

otherwise  % drug discovery with 160k points
    if ~exist('group_size', 'var'), group_size = 1; end
    alpha = [1 0.001 * ones(1, group_size)];

    if contains(data_name, 'ecfp')
        filename  = 'ecfp4_nearest_neighbors_100000.mat';
        group_ind = str2num(data_name(5:end));
    elseif contains(data_name, 'gpidaph')
        filename  = 'gpidaph3_nearest_neighbors_100000.mat';
        group_ind = str2num(data_name(8:end));
    elseif contains(data_name, 'morgan')
        filename  = 'morgan_nearest_neighbors_100000.mat';
        group_ind = str2num(data_name(7:end));
    elseif contains(data_name, 'single')
        assert(group_size == 1, 'group_size needs to be 1 for single-class cases');
        group_ind = str2num(data_name(7:end));
        filename  = sprintf('target_%i_single_nearest_neighbors_100000.mat', group_ind);
    else
        error(sprintf('unrecognized data name %s\n', data_name));
    end

    data_dir  = fullfile(data_dir, 'drug/precomputed');
    data_path = fullfile(data_dir, filename);
    load(data_path);

    if contains(data_name, 'single')  % filter labels
        this_ind = (labels ==  1) | (labels == group_ind + 1);
        labels   = labels(this_ind);
        labels(labels > 1) = 2;
    end

    num_points          = numel(labels);
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

    if ~contains(data_name, 'single')  % relabel for non-single-class cases
        if group_size == 1  % just use the same class in the single-class case
            pos_mask         = (labels == group_ind + 1);
            labels(:)        = 1;
            labels(pos_mask) = 2;
        else  % randomly pick out `group_size` classes
            rng(group_ind);

            selected_classes = randperm(120, group_size);
            selected_classes

            old_labels = labels;

            labels(:) = 1;
            for class_ind = 1:group_size
                pos_mask         = (old_labels == selected_classes(class_ind) + 1);
                labels(pos_mask) = class_ind + 1;
            end
        end
    end

% otherwise   % drug discovery
%     % this needs to be the same as in
%     % `../active_virtual_screening/calculate_nearest_neighbors_multiclass`
%     if ~exist('group_size', 'var'), group_size = 4; end
%
%     alpha        = [1 0.001 * ones(1, group_size)];
%     num_inactive = 100000;
%
%     if contains(data_name, 'ecfp')
%         filename = sprintf('group_%s_ecfp4_nearest_neighbors_%d.mat', ...
%                            data_name(5:end), num_inactive);
%     elseif contains(data_name, 'gpidaph')
%         filename = sprintf('group_%s_gpidaph3_nearest_neighbors_%d.mat', ...
%                            data_name(8:end), num_inactive);
%     else
%         error(sprintf('unrecognized data name %s\n', data_name));
%     end
%
%     data_dir  = fullfile(data_dir, 'drug/precomputed', sprintf('%d', group_size));
%     data_path = fullfile(data_dir, filename);
%     load(data_path);
%
%     labels     = this_labels;
%     num_points = numel(labels);
%
%     problem.num_points  = num_points;
%     problem.points      = (1:num_points)';
%     problem.num_classes = group_size + 1;
%
%     % limit to k-nearest neighbors
%     k = 100;
%     nearest_neighbors = nearest_neighbors(:, 1:k)';
%     similarities      = similarities(:, 1:k)';
%
%     % precompute sparse weight matrix
%     row_index = kron((1:num_points)', ones(k, 1));
%     weights = sparse(row_index, nearest_neighbors(:), similarities(:), ...
%                      num_points, num_points);

end

problem.max_num_influence = max(sum(weights > 0, 1));  % used for pruning
