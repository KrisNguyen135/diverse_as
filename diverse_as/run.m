if ~exist('exp',        'var'), exp        = 1; end
if ~exist('group_size', 'var'), group_size = 4; end
if ~exist('data',       'var'), data       = 'fashion'; end
if ~exist('utility',    'var'), utility    = 'log'; end
% if ~exist('policy',     'var'), policy     = 'classical ens'; end
if ~exist('policy',     'var'), policy     = 'ens jensen greedy'; end
% if ~exist('policy',     'var'), policy     = 'greedy'; end
% if ~exist('policy',     'var'), policy     = 'round robin greedy'; end
% if ~exist('policy',     'var'), policy     = 'round robin ucb'; end
% if ~exist('policy',     'var'), policy     = 'round robin ens'; end
% if ~exist('policy',     'var'), policy     = 'malkomes'; end
% if ~exist('policy',     'var'), policy     = 'he-carbonell'; end
% if ~exist('policy',     'var'), policy     = 'van'; end

addpath(genpath('../'));
addpath(genpath('../active_learning'));
addpath(genpath('../active_search'));

%%% high-level settings
exp
group_size
data
policy

budget     = 500
% result_dir = sprintf('results_%d', budget);
result_dir = 'results';
verbose    = true;
data_dir   = '../data/';
if ~isdir(data_dir)
    data_dir  = '/storage1/garnett/Active/activelearning/quan/diverse_as/data/';
end

[problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir, exp, group_size);
% result_dir = 'results_0002';
rng(exp);
if contains(data, 'gpidaph')
    rng(exp + 1);  % to get different init data than ecfp experiments
end

% randomly select a positive
train_ind = [randsample(find(labels > 1), 1)];

% randomly select a positive for each class
% train_ind = [];
% for i = 2:problem.num_classes
%     train_ind = [train_ind; randsample(find(labels == i), 1)];
% end

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
case 'sqrt'
    utility_function = @sqrt_utility;
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
case 'round robin ucb'
    beta = 10.0;

    name       = sprintf('%s_%.1f', name, beta);
    result_dir = 'results_ucb';

    policy = get_policy(@round_robin_ucb, model, beta);
case 'round robin ens'
    compute_limit = 500;
    sample_limit  = 500;
    name          = sprintf('%s_%d', name, compute_limit);
    result_dir    = 'results_rr_ens';

    policy = get_policy(@round_robin_ens, model, model_update, [], false, ...
                        compute_limit, sample_limit);
case 'classical greedy'
    policy = get_policy(@classical_greedy, model);
case 'classical ens'
    compute_limit = 500;
    sample_limit  = 500;
    name          = sprintf('%s_%d', name, compute_limit);

    policy = get_policy(@classical_ens, model, model_update, [], false, ...
                        compute_limit, sample_limit);
case 'ens jensen greedy'
    compute_limit = 500;
    sample_limit  = 500;
    name          = sprintf('%s_%d', name, compute_limit);

    batch_utility_function = get_batch_utility_function(@jensen, model);

    if group_size == 1
        batch_policy = get_batch_policy(@classical, model, batch_utility_function);
    else
        % batch_policy = get_batch_policy(@jensen_greedy, model);
        % batch_policy = get_batch_policy(@jensen_lazy_greedy, model);
        batch_policy = get_batch_policy(@jensen_greedy_prior_duplicate, model, alpha);
    end

    utility_upperbound_function = get_utility_upperbound_function( ...
        @jensen_upperbound, weights, nns', sims');
    policy = get_policy(@ens_base, model, batch_policy, batch_utility_function, ...
        utility_upperbound_function, true, compute_limit, sample_limit);
case 'malkomes'
    result_dir    = 'results_malkomes';
    sim_threshold = 0.5;
    name          = sprintf('%s_%.2f', name, sim_threshold);
    policy        = get_policy(@malkomes, model, nns', sims', sim_threshold);
case 'he-carbonell'
    result_dir = 'results_he_carbonell';
    name       = 'he_cabornell';
    policy     = get_he_carbonell_policy(nns', sims');
case 'van'
    result_dir     = 'results_van';
    tradeoff_param = 0.75;
    beta           = 10;
    name           = sprintf('%s_%.2f_%.2f', name, tradeoff_param, beta);
    policy         = get_policy(@van, model, tradeoff_param, beta);
end

if problem.verbose
    disp(train_ind);
    disp(train_labels);
    fprintf('utility function: %s\n', problem.utility);
    fprintf('policy: %s\n', name);
    fprintf('result dir: %s\n', result_dir);
end

%%% run experiment
message_prefix = sprintf('Exp %d: ', exp);

[train_ind, train_labels, queried_probs, computed, pruned] = diverse_active_search(...
    problem, train_ind, train_labels, labels, selector, utility_function, policy, ...
    message_prefix);

result_dir = fullfile('./', result_dir, data, int2str(group_size), name)
if ~isdir(result_dir), mkdir(result_dir); end

writematrix(train_ind, ...
    fullfile(result_dir, sprintf('%s__ind__%d.csv',      name, exp)));
writematrix(train_labels, ...
    fullfile(result_dir, sprintf('%s__labels__%d.csv',   name, exp)));
writematrix(queried_probs, ...
    fullfile(result_dir, sprintf('%s__probs__%d.csv',    name, exp)));
writematrix(computed, ...
    fullfile(result_dir, sprintf('%s__computed__%d.csv', name, exp)));
writematrix(pruned, ...
    fullfile(result_dir, sprintf('%s__pruned__%d.csv',   name, exp)));
