% queries the point that gives maximum probability of a given positive class
% in a round robin manner
function [chosen_ind, chosen_prob, num_computed, num_pruned] = round_robin_greedy( ...
    problem, train_ind, train_labels, test_ind, model)

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);
% the class to be targeted this round, between 2 and problem.num_classes
target_class = mod(size(train_ind, 1), problem.num_classes - 1) + 2;

if strcmp(problem.utility, 'weighted')
    remainder          = mod(size(train_ind, 1), sum(problem.weights));
    cumulative_weights = cumsum(problem.weights);
    target_class       = find(remainder < cumulative_weights, 1) + 1;
end

% [~, chosen_ind] = max(probs(:, target_class));
maxprob = max(probs(:, target_class));
indices = find(probs(:, target_class) == maxprob);
if numel(indices) > 1
    chosen_ind = randsample(indices, 1);
else
    chosen_ind = indices;
end

chosen_prob     = probs(chosen_ind, :);
num_computed    = numel(test_ind);
num_pruned      = [0 0];
chosen_ind      = test_ind(chosen_ind);
