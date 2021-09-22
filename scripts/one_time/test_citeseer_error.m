if ~exist('exp',     'var'), exp     = 1; end
if ~exist('data',    'var'), data    = 'citeseer'; end
if ~exist('utility', 'var'), utility = 'log'; end
if ~exist('policy',  'var'), policy  = 'greedy'; end

addpath(genpath('../'));
addpath(genpath('../active_learning'));
addpath(genpath('../active_search'));

%%% high-level settings
budget   = 2;
verbose  = true;
data_dir = '../data/';
if ~isdir(data_dir)
    data_dir  = '/storage1/garnett/Active/activelearning/quan/diverse_as/data/';
end

[problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir);
rng(exp);

% randomly select a positive
train_ind    = [randsample(find(labels > 1), 1)];
train_labels = labels(train_ind);

% randomly select a positive for each class
% train_ind    = [];
% train_labels = [];
% for i = 2:problem.num_classes
%     pos_ind      = find(labels == i);
%     train_ind    = [train_ind; randsample(pos_ind, 1)];
%     train_labels = [train_labels; i];
% end

% train_ind    = [randsample(find(labels > 1), 1)];
% train_labels = labels(train_ind);

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
    disp(train_ind);
    disp(train_labels);
    fprintf('utility function: %s\n', problem.utility);
    fprintf('policy: %s\n', name);
end

%%% run experiment
message_prefix = sprintf('Exp %d: ', exp);

[queries, queried_labels, queried_probs, computed, pruned] = diverse_active_search(...
    problem, train_ind, train_labels, labels, selector, utility_function, policy, ...
    message_prefix);

result_dir = fullfile(data_dir, 'results', data, name);
if ~isdir(result_dir), mkdir(result_dir); end

writematrix(queries, ...
    fullfile(result_dir, sprintf('%s__queries__%d.csv',        name, exp)));
writematrix(queried_labels, ...
    fullfile(result_dir, sprintf('%s__queried_labels__%d.csv', name, exp)));
writematrix(queried_probs, ...
    fullfile(result_dir, sprintf('%s__queried_probs__%d.csv',  name, exp)));
writematrix(computed, ...
    fullfile(result_dir, sprintf('%s__computed__%d.csv',       name, exp)));
writematrix(pruned, ...
    fullfile(result_dir, sprintf('%s__pruned__%d.csv',         name, exp)));