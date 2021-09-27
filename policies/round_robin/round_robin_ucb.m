function [chosen_ind, chosen_prob, num_computed, num_pruned] = round_robin_ucb( ...
    problem, train_ind, train_labels, test_ind, model, beta)

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);
% the class to be targeted this round, between 2 and problem.num_classes
target_class = mod(size(train_ind, 1), problem.num_classes - 1) + 2;

positive_probs = probs(:, target_class);
ucb = positive_probs + beta * sqrt(positive_probs .* (1 - positive_probs));

maxucb  = max(ucb);
indices = find(ucb == maxucb);
if numel(indices) > 1
    chosen_ind = randsample(indices, 1);
else
    chosen_ind = indices;
end

chosen_prob     = probs(chosen_ind, :);
num_computed    = numel(test_ind);
num_pruned      = [0 0];
chosen_ind      = test_ind(chosen_ind);
