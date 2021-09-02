% computes the expected value of the one-step marginal utility gain
function [chosen_ind, chosen_prob, num_computed, num_pruned] = greedy( ...
    problem, train_ind, train_labels, test_ind, model, utility_function)

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);
switch problem.utility
case 'log'
    marginal_utility = probs(:, 2:end) * ...
                       (log(problem.counts(2:end) + 2)' - log(problem.counts(2:end) + 1)');
end

% [~, chosen_ind] = max(marginal_utility);
maxu    = max(marginal_utility);
indices = find(marginal_utility == maxu);
if numel(indices) > 1
    chosen_ind = randsample(indices, 1);
else
    chosen_ind = indices;
end

chosen_prob  = probs(chosen_ind, :);
num_computed = numel(test_ind);
num_pruned   = [0 0];
chosen_ind   = test_ind(chosen_ind);
