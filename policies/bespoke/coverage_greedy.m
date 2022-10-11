function [chosen_ind, chosen_prob, num_computed, num_pruned] = coverage_greedy( ...
    problem, train_ind, train_labels, test_ind, model)

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);

missing_mask = (problem.counts(2:end) == 0);
if sum(missing_mask) == 0
    chosen_ind = 1;
else
    pos_probs        = probs(:, 2:end);
    marginal_utility = sum(pos_probs(:, missing_mask), 2);

    maxu    = max(marginal_utility);
    indices = find(marginal_utility == maxu);
    if numel(indices) > 1
        chosen_ind = randsample(indices, 1);
    else
        chosen_ind = indices;
    end
end

chosen_prob  = probs(chosen_ind);
num_computed = numel(test_ind);
num_pruned   = [0 0];
chosen_ind   = test_ind(chosen_ind);
