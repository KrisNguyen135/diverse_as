function [batch_utility, batch_ind] = classical(problem, train_ind, ...
    train_labels, test_ind, batch_size, model, batch_utility_function)

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);

[min_neg_probs, batch_ind] = mink(probs(:, 1), batch_size);
batch_ind     = test_ind(batch_ind);
batch_utility = batch_utility_function(problem, train_ind, train_labels, batch_ind);
