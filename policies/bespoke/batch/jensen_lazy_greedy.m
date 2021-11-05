function batch_ind = jensen_lazy_greedy(problem, train_ind, train_labels, ...
                                        test_ind, batch_size, model)

assert(strcmp(problem.utility, 'log'), 'only log utility is currently supported');

batch_ind     = nan(batch_size, 1);
[probs, ~, ~] = model(problem, train_ind, train_labels, test_ind);
loggend       = 1 + problem.counts(2:end);

upperbounds           = Inf(numel(test_ind), 1);
best_marginal_utility = -Inf;

reverse_ind           = zeros(problem.num_points, 1);
reverse_ind(test_ind) = 1:numel(test_ind);

for i = 1:batch_size
    % only evaluate lazy points
    lazy_mask     = (upperbounds >= best_marginal_utility);
    lazy_test_ind = test_ind(lazy_mask);
    lazy_probs    = probs(lazy_mask, :);

    % Jensen's utility upper bound for each candidate (n x 1 column vector)
    marginal_utility = sum(log(loggend + lazy_probs(:, 2:end)), 2) - sum(log(loggend), 2);

    % find the candidate that leads to the biggest immediate marginal gain
    best_marginal_utility = max(marginal_utility)
    indices               = find(marginal_utility == best_marginal_utility);
    if numel(indices) > 1
        chosen_ind = randsample(indices, 1);
    else
        chosen_ind = indices;
    end

    % add the greedy candidate to the running batch
    batch_ind(i) = lazy_test_ind(chosen_ind);
    batch_ind(i)
    loggend      = loggend + lazy_probs(chosen_ind, 2:end);

    % remove the greedy candidate from the pool
    reverse_chosen_ind           = reverse_ind(batch_ind(i));
    probs(reverse_chosen_ind, :) = [];
    test_ind(reverse_chosen_ind) = [];

    % update lazy greedy stats
    upperbounds(lazy_mask)          = marginal_utility;
    upperbounds(reverse_chosen_ind) = [];

    if i > 10
        quit
    end
end
