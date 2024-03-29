if ~exist('exp',     'var'), exp     = 1; end
if ~exist('data',    'var'), data    = 'citeseer'; end
if ~exist('utility', 'var'), utility = 'log'; end
if ~exist('policy',  'var'), policy  = 'ens jensen greedy'; end

addpath(genpath('../../'));
addpath(genpath('../../active_learning'));
addpath(genpath('../../active_search'));

%%% high-level settings
budget   = 2;
verbose  = true;
data_dir = '../../data/';
if ~isdir(data_dir)
    data_dir  = '/storage1/garnett/Active/activelearning/quan/diverse_as/data/';
end

[problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir);
rng(exp);

train_ind    = load('../../notes/bjob_output/train_ind_run.1398258');
train_labels = labels(train_ind);

%%% experiment details
problem.verbose     = verbose;
problem.num_initial = numel(train_ind);
problem.num_queries = budget;
problem.counts      = zeros(1, problem.num_classes);
for i = 1:numel(train_labels)
    problem.counts(train_labels(i)) = problem.counts(train_labels(i)) + 1;
end

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
case 'classical greedy'
    policy = get_policy(@classical_greedy, model);
case 'classical ens'
    limit  = 10;
    policy = get_policy(@classical_ens, model, model_update, [], limit);
case 'ens jensen greedy'
    compute_limit = 500;
    sample_limit  = 500;
    batch_utility_function = get_batch_utility_function(@jensen, model);
    batch_policy = get_batch_policy(@jensen_greedy, model);
    utility_upperbound_function = get_utility_upperbound_function( ...
        @jensen_upperbound, weights, nns', sims');
    policy = get_policy(@ens_base, model, batch_policy, batch_utility_function, ...
        utility_upperbound_function, true, compute_limit, sample_limit);
end

if problem.verbose
    fprintf('utility function: %s\n', problem.utility);
    fprintf('policy: %s\n', name);
end

%%% run experiment
message_prefix = sprintf('Exp %d: ', exp);

[queries, queried_labels, queried_probs, computed, pruned] = diverse_active_search(...
    problem, train_ind, train_labels, labels, selector, utility_function, policy, ...
    message_prefix);
