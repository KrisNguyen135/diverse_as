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
[samples, sample_weights, sample_counts, utility] = independent( ...
    problem, train_ind, train_labels, batch_ind, model, utility_function);

[probs, n, d] = model(problem, train_ind, train_labels, batch_ind);

%%% loop over all batches
% test_ind = selector(problem, train_ind, []);
%
% for i = 1:(numel(test_ind) - 1)
%     fprintf('%d / %d\n', i, numel(test_ind));
%     for j = (i + 1):numel(test_ind)
%         % fprintf('%d, %d: ', i, j);
%
%         batch_ind = test_ind([i j]);
%         [samples, sample_weights, sample_counts, utility1] = exact( ...
%             problem, train_ind, train_labels, batch_ind, model, weights, utility_function);
%
%         batch_ind = test_ind([j i]);
%         [samples, sample_weights, sample_counts, utility2] = exact( ...
%             problem, train_ind, train_labels, batch_ind, model, weights, utility_function);
%
%         if abs(utility1 - utility2) > 1e-3
%             fprintf('%d, %d: %.4f vs. %.4f\n', i, j, utility1, utility2);
%         end
%     end
% end

%%% batches of size 1 should match the greedy score: it does!
% test_ind = selector(problem, train_ind, []);
% [probs, n, d] = model(problem, train_ind, train_labels, test_ind);
%
% for i = 1:numel(test_ind)
%     [samples, sample_weights, sample_counts, utility1] = exact( ...
%         problem, train_ind, train_labels, test_ind(i), ...
%         model, weights, utility_function);
%
%     utility2 = probs(i, 2:end) * ...
%                (log(problem.counts(2:end) + 2)' - log(problem.counts(2:end) + 1)') + ...
%                utility_function(problem);
%
%     if abs(utility1 - utility2) > 1e-5
%         fprintf('%d, %d: %g vs. %g\n', i, test_ind(i), utility1, utility2);
%     end
% end

%%% custom averaging
% [probs, n, d] = model(problem, train_ind, train_labels, batch_ind);
%
% other_utility = ...
%     sample_weights(1) * log(2) + ...  % one 2
%     sample_weights(2) * log(3) + ...  % two 2's
%     sum(sample_weights(3:6)) * log(2) * 2 + ...  % one 2 and one other // first 6 samples
%     sample_weights(7) * log(3) + ...  % two 2's
%     sample_weights(8) * log(4) + ...  % three 2's
%     sum(sample_weights(9:12)) * (log(3) + log(2)) + ...  % two 2's and one other class // second 6 samples
%     sample_weights(13) * log(2) * 2 + ...  % one 2 and one 3
%     sample_weights(14) * (log(2) + log(3)) + ...  % two 2's and one 3
%     sample_weights(15) * (log(2) + log(3)) + ...  % one 2 and two 3's
%     sum(sample_weights(16:18)) * log(2) * 3 + ...  % one 2, one 3, and one other  // third 6 samples
%     sample_weights(19) * log(2) * 2 + ...  % one 2 and one 4
%     sample_weights(20) * (log(3) + log(2)) + ...  % two 2's and one 4
%     sample_weights(21) * log(2) * 3 + ...  % one 2, one 3, and one 4
%     sample_weights(22) * (log(2) + log(3)) + ...  % one 2 and two 4's
%     sum(sample_weights(23:24)) * log(2) * 3 + ...  % one 2, one 4, and one other // forth 6 samples
%     sample_weights(25) * log(2) * 2 + ...  % one 2 and one 5
%     sample_weights(26) * (log(3) + log(2)) + ...  % two 2's and one 5
%     sum(sample_weights(27:28)) * log(2) * 3 + ...  % one 2, one 5, and one other either 3 or 4
%     sample_weights(29) * (log(2) + log(3)) + ...  % one 2 and two 5's
%     sample_weights(30) * log(2) * 3 + ...  % one 2, one 5, and one 6  // fifth 6 samples
%     sample_weights(31) * log(2) * 2 + ...  % one 2 and one 6
%     sample_weights(32) * (log(2) + log(3)) + ...  % two 2's and one 6
%     sum(sample_weights(33:35)) * log(2) * 3 + ...  % one 2, one 6, and one other between 3 and 5
%     sample_weights(36) * (log(2) + log(3));  % one two and two 6's
