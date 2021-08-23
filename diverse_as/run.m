if ~exist('exp',     'var'), exp     =          1; end
if ~exist('data',    'var'), data    = 'citeseer'; end
if ~exist('utility', 'var'), utility =      'log'; end
if ~exist('policy',  'var'), policy  =   'greedy'; end

addpath(genpath('../'));
addpath(genpath('../active_learning'));
addpath(genpath('../active_search'));

%%% high-level settings
budget   = 10;
verbose  = true;
data_dir = '../data/';
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

switch policy
case 'greedy'
    name = 'greedy';
end

if problem.verbose
    train_ind
    train_labels
    utility_function
    policy
end

%%% run experiment
message_prefix = sprintf('Exp %d:', exp);

[utilities, queries, queried_probs, computed, pruned] = diverse_active_search(...
    problem, train_ind, train_labels, labels, selector, utility_function, policy, ...
    message_prefix);

result_dir = fullfile(data_dir, 'results', data, policy);
if ~isdir(result_dir), mkdir(result_dir); end

writematrix(utilities, ...
    fullfile(result_dir, sprintf('%s__utilities__%d.csv',     policy, exp)));
writematrix(queries, ...
    fullfile(result_dir, sprintf('%s__queries__%d.csv',       policy, exp)));
writematrix(queried_probs, ...
    fullfile(result_dir, sprintf('%s__queried_probs__%d.csv', policy, exp)));
writematrix(computed, ...
    fullfile(result_dir, sprintf('%s__computed__%d.csv',      policy, exp)));
writematrix(pruned, ...
    fullfile(result_dir, sprintf('%s__pruned__%d.csv',        policy, exp)));
