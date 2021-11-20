function batch_utility = jensen_lazy_greedy(problem, train_ind, train_labels, ...
                                            test_ind, batch_size, model)

assert(strcmp(problem.utility, 'log'), 'only log utility is currently supported');

%%% setup
[probs, ~, ~]  = model(problem, train_ind, train_labels, test_ind);
loggend        = 1 + problem.counts(2:end);
jensen_utility = sum(log(loggend), 2);

%%% do the entire first sweep
marginal_utility = sum(log(loggend + probs(:, 2:end)), 2) - jensen_utility;

% find the candidate that leads to the biggest immediate marginal gain
maxu    = max(marginal_utility);
indices = find(marginal_utility == maxu);
if numel(indices) > 1
    chosen_ind = randsample(indices, 1);
else
    chosen_ind = indices;
end

loggend        = loggend + probs(chosen_ind, 2:end);
jensen_utility = sum(log(loggend), 2);

% remove the greedy candidate from the pool
probs(chosen_ind, :)         = [];
test_ind(chosen_ind)         = [];
marginal_utility(chosen_ind) = [];

batch_utility = lazy_greedy(probs(:, 2:end)', loggend(:, :), jensen_utility(:), ...
                            batch_size - 1, marginal_utility, size(probs, 1), ...
                            problem.num_classes - 1);
