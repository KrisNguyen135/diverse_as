if ~exist('num_exps',   'var'), num_exps   = 10; end
if ~exist('train_size', 'var'), train_size = 0.01; end
if ~exist('data',       'var'), data       = 'fashion'; end
if ~exist('utility',    'var'), utility    = 'log'; end

addpath(genpath('../../'));
addpath(genpath('../../active_learning'));
addpath(genpath('../../active_search'));

verbose    = true;
data_dir   = '../../data/';
if ~isdir(data_dir)
    data_dir  = '/storage1/garnett/Active/activelearning/quan/diverse_as/data/';
end

[problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir, [], []);

ks   = [1 5 10 25 50];
paks = nan(num_exps, problem.num_classes - 1, numel(ks));

% initialize the k-NN model
model        = get_model(@knn_model_new, weights, alpha);
model_update = get_model_update(@knn_model_update, weights);
selector     = get_selector(@unlabeled_selector);

for exp = 1:num_exps
    rng(exp);

    % generate a random training set
    train_ind    = randsample(problem.points, fix(train_size * problem.num_points));
    train_labels = labels(train_ind);

    % make predictions on the test set
    test_ind      = selector(problem, train_ind, []);
    test_labels   = labels(test_ind);
    [probs, n, d] = model(problem, train_ind, train_labels, test_ind);

    for positive_class = 2:problem.num_classes
        [top_probs, top_ind] = maxk(probs(:, positive_class), max(ks));

        [sort_top_probs, sort_top_ind] = sort(top_probs, 'descend');

        sorted_top_ind = top_ind(sort_top_ind);

        for k_ind = 1:numel(ks)
            k = ks(k_ind);

            this_top_ind    = top_ind(1:k);
            this_top_labels = test_labels(this_top_ind);

            this_pak = sum(this_top_labels == positive_class) / k;
            paks(exp, positive_class - 1, k_ind) = this_pak;
        end
    end
end

% writematrix(paks, sprintf('paks_%.2f.csv', train_size));
save(sprintf('paks_%.2f.mat', train_size), 'paks');
