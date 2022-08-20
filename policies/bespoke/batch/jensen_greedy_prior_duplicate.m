function batch_utility = jensen_greedy_prior_duplicate(problem, train_ind, ...
    train_labels, test_ind, batch_size, model, alpha)

% assert(strcmp(problem.utility, 'log'), 'only log utility is currently supported');

batch_ind     = nan(batch_size, 1);
[probs, ~, d] = model(problem, train_ind, train_labels, test_ind);
count_sums    = problem.counts(2:end);

weight_sum = d;  % column vector
weight_sum = weight_sum - sum(alpha);

prior_duplicate_mask = (weight_sum == 0);  % points that haven't changed from the prior
other_mask           = ~prior_duplicate_mask;

% only keep `batch_size` points in the set of priors
pruned_duplicate_ind = find(prior_duplicate_mask);
pruned_duplicate_ind = pruned_duplicate_ind(1:batch_size);
pruned_ind           = [find(other_mask); pruned_duplicate_ind];
pruned_test_ind      = test_ind(pruned_ind);
pruned_probs         = probs(pruned_ind, 2:end);

for i = 1:batch_size
    if strcmp(problem.utility, 'log')
        marginal_utility = sum(log(count_sums + 1 + pruned_probs), 2);
    elseif strcmp(problem.utility, 'sqrt')
        marginal_utility = sum(sqrt(count_sums + pruned_probs), 2);
    end

    % find the candidate that leads to the biggest immediate marginal gain
    maxu    = max(marginal_utility);
    indices = find(marginal_utility == maxu);
    if numel(indices) > 1
        chosen_ind = randsample(indices, 1);
    else
        chosen_ind = indices;
    end

    % add the greedy candidate to the running batch
    batch_ind(i) = pruned_test_ind(chosen_ind);
    count_sums   = count_sums + pruned_probs(chosen_ind, :);

    % remove the greedy candidate from the pool
    pruned_probs(chosen_ind, :) = [];
    pruned_test_ind(chosen_ind) = [];
end

if strcmp(problem.utility, 'log')
    batch_utility = sum(log(count_sums + 1), 2);
elseif strcmp(problem.utility, 'sqrt')
    batch_utility = sum(sqrt(count_sums), 2);
end
