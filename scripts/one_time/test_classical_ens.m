if ~exist('exp',     'var'), exp     =          1; end
if ~exist('data',    'var'), data    = 'citeseer'; end
if ~exist('utility', 'var'), utility =      'log'; end

if ~exist('policy', 'var'), policy = 'classical ens'; end

addpath(genpath('../../'));
addpath(genpath('../../active_learning'));
addpath(genpath('../../active_search'));

%%% high-level settings
budget   = 500;
verbose  = true;
data_dir = '../../data/';
if ~isdir(data_dir)
    data_dir  = '/storage1/garnett/Active/activelearning/quan/diverse_as/data/';
end

[problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir);
rng(exp);

train_ind    = [];
train_labels = [];
for i = 2:problem.num_classes
    pos_ind      = find(labels == i);
    train_ind    = [train_ind; randsample(pos_ind, 1)];
    train_labels = [train_labels; i];
end

%%% experiment details
problem.verbose     = verbose;
problem.num_initial = numel(train_ind);
problem.num_queries = budget;
problem.counts      = [0 ones(1, problem.num_classes - 1)];

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

name = policy;
switch name
case 'greedy'
    policy = get_policy(@greedy, model, utility_function);
case 'round robin greedy'
    policy = get_policy(@round_robin_greedy, model);
case 'classical ens'
    limit  = 10;
    policy = get_policy(@classical_ens, model, model_update, [], limit);
end

if problem.verbose
    disp(train_ind);
    disp(train_labels);
    fprintf('utility function: %s\n', problem.utility);
    fprintf('policy: %s\n', name);
end

%%% run experiment
message_prefix = sprintf('Exp %d: ', exp);

test_ind = selector(problem, train_ind, []);
[probs, n, d] = model(problem, train_ind, train_labels, test_ind);
reverse_ind = zeros(problem.num_points, 1);
num_test = numel(test_ind);
reverse_ind(test_ind) = 1:num_test;

%%% test one point
this_test_ind = 1865;
i = find(test_ind == this_test_ind);

fake_utilities = zeros(problem.num_classes, 1);
fake_train_ind = [train_ind; this_test_ind];
fake_test_ind = selector(problem, fake_train_ind, []);
for fake_label = 1:problem.num_classes
    [fake_probs, ~, ~] = model(problem, fake_train_ind, ...
                               [train_labels; fake_label], fake_test_ind);

    fake_utilities(fake_label) = sum(maxk(sum(fake_probs(:, 2:end), 2), 499));
end

p = probs(i, :);
utility = p * fake_utilities + 1 - p(1)
