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

% sigma_primes
% updated_sigma_primes

% finally, compute the utility upper bounds
bounds = nan(1, problem.num_classes);

if strcmp(problem.utility, 'log')
    bounds(1) = sum(log(problem.counts(2:end) + 1 + sigma_primes));
    for i = 2:problem.num_classes
        % updated Σ's for the current positive class + Σ's for the other positive classes
        % adding 2 instead of 1 since we're conditioning on an add. positive of class i
        bounds(i) = log(problem.counts(i) + 2 + updated_sigma_primes(i - 1)) ...
                    + bounds(1) - log(problem.counts(i) + 1 + sigma_primes(i - 1));
    end
elseif strcmp(problem.utility, 'sqrt')
    bounds(1) = sum(sqrt(problem.counts(2:end) + sigma_primes));
    for i = 2:problem.num_classes
        bounds(i) = sqrt(problem.counts(i) + 1 + updated_sigma_primes(i - 1)) ...
                    + bounds(1) - sqrt(problem.counts(i) + sigma_primes(i - 1));
    end
elseif strcmp(problem.utility, 'weighted')
    bounds(1) = log(problem.counts(2:end) + 1 + sigma_primes) * problem.weights;
    for i = 2:problem.num_classes
        bounds(i) = log(problem.counts(i) + 2 + updated_sigma_primes(i - 1)) ...
                    + bounds(1) - log(problem.counts(i) + 1 + sigma_primes(i - 1));
        bounds(i) = bounds(i) * problem.weights(i - 1);
    end
end
