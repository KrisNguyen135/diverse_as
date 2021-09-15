data = 'square_small';
% data = 'square';

batch_utility = 'exact';

if ~exist('exp',     'var'), exp     =     1; end
if ~exist('utility', 'var'), utility = 'log'; end

addpath(genpath('../../'));
addpath(genpath('../../active_learning'));
addpath(genpath('../../active_search'));

%%% high-level settings
budget   = 10;
verbose  = true;
data_dir = '../../data/';

[problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir);
rng(exp);

train_ind    = [randsample(find(labels > 1), 5)];
train_labels = labels(train_ind);

%%% experiment details
problem.verbose     = verbose;
problem.num_initial = numel(train_ind);
problem.num_queries = budget;
problem.counts      = zeros(1, problem.num_classes);
problem.counts(train_labels) = 1;

model        = get_model(@knn_model_new, weights, alpha);
model_update = get_model_update(@knn_model_update, weights);
selector     = get_selector(@unlabeled_selector);

switch utility
case 'log'
    utility_function = @log_utility;
case 'threshold'
    threshold        = 2;
    utility_function = get_utility_function(@threshold_utility, threshold);
end
problem.utility = utility;

% batch_utility_function = get_batch_utility_function(@exact, model, weights, utility_function);
batch_utility_function = get_batch_utility_function(@jensen, model);

% batch_policy = get_batch_policy(@classical, model);
batch_policy = get_batch_policy(@jensen_greedy, model);

policy = get_policy(@ens_base, model, batch_policy, batch_utility_function);

if problem.verbose
    disp(train_ind);
    disp(train_labels);
    fprintf('utility function: %s\n', problem.utility);
end

test_ind      = selector(problem, train_ind, []);
[probs, n, d] = model(problem, train_ind, train_labels, test_ind);

% [queries, queried_labels, queried_probs, computed, pruned] = diverse_active_search(...
%     problem, train_ind, train_labels, labels, selector, utility_function, policy);

bounds = knn_probability_bound_new(problem, train_ind, train_labels, test_ind, ...
    n, d, budget, weights, nns', sims');
