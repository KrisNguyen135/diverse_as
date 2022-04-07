function [chosen_ind, chosen_prob, num_computed, num_pruned] = van( ...
    problem, train_ind, train_labels, test_ind, model, tradeoff_param, beta)

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);

pos_probs = 1 - probs(:, 1);
variances = pos_probs .* (1  - pos_probs);
ucb       = pos_probs + beta * sqrt(variances);
scores    = (1 - tradeoff_param) * ucb + tradeoff_param / 2 * log(1 + variances);

max_score = max(scores);
indices = find(scores == max_score);
if numel(indices) > 1
    chosen_ind = randsample(indices, 1);
else
    chosen_ind = indices;
end

chosen_prob     = probs(chosen_ind, :);
num_computed    = numel(test_ind);
num_pruned      = [0 0];
chosen_ind      = test_ind(chosen_ind);
