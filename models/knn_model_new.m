function [probs, n, d] = knn_model(problem, train_ind, train_labels, ...
                                   test_ind, weights, alpha)

n = nan(numel(test_ind), problem.num_classes);

for i = 1:problem.num_classes
    n(:, i) = alpha(i) + sum(weights(test_ind, train_ind(train_labels == i)), 2);
end

d     = sum(n, 2);
probs = n ./ d;
