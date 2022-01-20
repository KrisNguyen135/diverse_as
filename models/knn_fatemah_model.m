function [probs, n, d] = knn_fatemah_model(problem, train_ind, train_labels, ...
                                           test_ind, weights, alpha, core_mask)

n = nan(numel(test_ind), 2);

n(:, 1) = alpha(1) + sum(weights(test_ind, train_ind(train_labels == 1)), 2);
n(:, 2) = alpha(2) + sum(weights(test_ind, train_ind(train_labels > 1)), 2);

d     = sum(n, 2);
probs = n ./ d;

pos_probs = probs(:, 2) .* core_mask(test_ind, :);
probs     = [probs(:, 1) pos_probs];
