function prob_upper_bound = knn_probability_bound_new(...
    problem, train_ind, train_labels, test_ind, num_positives, remain_budget, ...
    weights, knn_ind, knn_weights, alpha)

in_train = ismember(knn_ind(:), train_ind);
knn_weights(in_train) = 0;
if num_positives == 1
    success_count_bound = max(...
        knn_weights(test_ind, 1:min(end, length(train_ind) + 1)), ...
        [], 2);
else
    disp("shouldn't be in here!!!")
end

max_alpha
