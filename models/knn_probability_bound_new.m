function bounds = knn_probability_bound_new(problem, train_ind, train_labels, ...
    test_ind, n, d, budget, weights, knn_ind, knn_weights, alpha)

in_train = ismember(knn_ind(:), train_ind);
knn_weights(in_train) = 0;
success_count_bound = max( ...
    knn_weights(test_ind, 1:min(end, numel(train_ind) + 1)), [], 2);  % column vector of length (test size)

max_alpha = n(:, 2:end) + success_count_bound;  % (test size) x (num_classes - 1)

bounds = nan(1, problem.num_classes - 1);

for i = 2:problem.num_classes
    tmp_max_alpha = max_alpha(:, i - 1);  % column vector
    min_beta      = d - n(:, i);          % column vector
    probabilities = tmp_max_alpha ./ (tmp_max_alpha + min_beta);

    if problem.max_num_influence >= budget
        bound = sum(maxk(probabilities, budget));
    else
        bound = sum(maxk(probabilities, problem.max_num_influence));
    end
    bounds(i - 1) = bound;
end
