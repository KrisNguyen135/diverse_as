% queries the points that gives minimum negative probability
function [chosen_ind, chosen_prob, num_computed, num_pruned] = classical_greedy(...
    problem, train_ind, train_labels, test_ind, model)

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);

[~, chosen_ind] = min(probs(:, 1));
chosen_prob     = probs(chosen_ind, :);
num_computed    = numel(test_ind);
num_pruned      = [0 0];
chosen_ind      = test_ind(chosen_ind);
