[probs, n, d] = knn_model_update(problem, n, d, ind, label, test_ind, weights)
% n and d must already match test_ind

weight = weights(test_ind, ind);

n(:, label) = n(:, label) + weight;
d           = d           + weight;

probs = n ./ d;
