% data = 'square_small';
data = 'square';

if ~exist('exp',     'var'), exp     =     1; end
if ~exist('utility', 'var'), utility = 'log'; end

addpath(genpath('../../'));
addpath(genpath('../../active_learning'));
addpath(genpath('../../active_search'));

%%% high-level settings
budget   = 200;
verbose  = true;
data_dir = '../../data/';

[problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir);
rng(exp);

train_ind    = [randsample(find(labels == 2), 1)];
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

if problem.verbose
    disp(train_ind);
    disp(train_labels);
    fprintf('utility function: %s\n', problem.utility);
end

%%% specific batches
batch_ind = [1, 12];
% batch_ind = [12, 1];
utility   = jensen(problem, train_ind, train_labels, batch_ind, model);

[probs, n, d] = model(problem, train_ind, train_labels, batch_ind);
