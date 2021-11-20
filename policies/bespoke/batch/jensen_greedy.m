function batch_utility = jensen_greedy(problem, train_ind, train_labels, test_ind, ...
                                       batch_size, model)

assert(strcmp(problem.utility, 'log'), 'only log utility is currently supported');

batch_ind     = nan(batch_size, 1);
[probs, ~, ~] = model(problem, train_ind, train_labels, test_ind);
loggend       = 1 + problem.counts(2:end);

for i = 1:batch_size
    % n x 1 column vector --- Jensen's utility upper bound for each candidate
    marginal_utility = sum(log(loggend + probs(:, 2:end)), 2);

    % find the candidate that leads to the biggest immediate marginal gain
    maxu    = max(marginal_utility);
    indices = find(marginal_utility == maxu);
    if numel(indices) > 1
        chosen_ind = randsample(indices, 1);
    else
        chosen_ind = indices;
    end

    % add the greedy candidate to the running batch
    batch_ind(i) = test_ind(chosen_ind);
    loggend      = loggend + probs(chosen_ind, 2:end);

    % remove the greedy candidate from the pool
    probs(chosen_ind, :) = [];
    test_ind(chosen_ind) = [];
end

batch_utility = sum(log(loggend), 2);
