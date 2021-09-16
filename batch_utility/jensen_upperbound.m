function bounds = jensen_upperbound(problem, train_ind, train_labels, test_ind, ...
    probs, n, d, budget, weights, knn_ind, knn_weights)

in_train = ismember(knn_ind(:), train_ind);
knn_weights(in_train) = 0;
success_count_bound = max( ...
    knn_weights(test_ind, 1:min(end, numel(train_ind) + 1)), [], 2);  % column vector of length (test size)

max_alpha = n(:, 2:end) + success_count_bound;  % (test size) x (num_classes - 1)

sigma_primes         = nan(1, problem.num_classes - 1);  % sum of top current probabilities
updated_sigma_primes = nan(1, problem.num_classes - 1);  % sum of top updated probabilities

% compute the Σ's
for i = 2:problem.num_classes
    top_probs = maxk(probs(:, i), budget);  % top current probabilities
    sigma_primes(i - 1) = sum(top_probs);

    tmp_max_alpha = max_alpha(:, i - 1);  % column vector
    min_beta      = d - n(:, i);          % column vector
    probabilities = tmp_max_alpha ./ (tmp_max_alpha + min_beta);

    if problem.max_num_influence >= budget
        sigma_prime = sum(maxk(probabilities, budget));
    else
        sigma_prime = sum(maxk(probabilities, problem.max_num_influence)) + ...
                      sum(maxk(top_probs, budget - problem.max_num_influence));
    end
    updated_sigma_primes(i - 1) = sigma_prime;
end

sigma_primes
updated_sigma_primes

% finally, compute the utility upper bounds
bounds = nan(1, problem.num_classes);

bounds(1) = sum(log(problem.counts(2:end) + 1 + sigma_primes));
for i = 2:problem.num_classes
    % updated Σ's for the current positive class + Σ's for the other positive classes
    bounds(i) = log(problem.counts(i) + 1 + updated_sigma_primes(i - 1)) ...
                + bounds(1) - log(problem.counts(i) + 1 + sigma_primes(i - 1));
end
