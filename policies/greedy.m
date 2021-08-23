function [chosen_ind, chosen_prob, num_computed, num_pruned] = ...
    greedy(problem, train_ind, train_labels, test_ind, model, model_update)

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);
