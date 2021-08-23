[probs, d, n] = knn_model_update(problem, n, d, ind, label, test_ind, weights)

weight = weights(test_ind, ind);

n(:, label) = n(:, label) + weight;
d           = d           + weight;

probs = n ./ d;
