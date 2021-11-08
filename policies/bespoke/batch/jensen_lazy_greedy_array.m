% sort the candidates by marginal utilities in an array
% pretty bad in terms of speed
% the bottleneck is the insertion of an evaluated point into the sorted array

function batch_ind = jensen_lazy_greedy(problem, train_ind, train_labels, ...
                                        test_ind, batch_size, model)

assert(strcmp(problem.utility, 'log'), 'only log utility is currently supported');

batch_ind      = nan(batch_size, 1);
[probs, ~, ~]  = model(problem, train_ind, train_labels, test_ind);
loggend        = 1 + problem.counts(2:end);
jensen_utility = sum(log(loggend), 2);

upperbounds           = Inf(numel(test_ind), 1);
best_marginal_utility = -Inf;

reverse_ind           = zeros(problem.num_points, 1);
reverse_ind(test_ind) = 1:numel(test_ind);

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

% add the greedy candidate to the running batch
batch_ind(1)   = test_ind(chosen_ind);
loggend        = loggend + probs(chosen_ind, 2:end);
jensen_utility = sum(log(loggend), 2);

% remove the greedy candidate from the pool
probs(chosen_ind, :)         = [];
test_ind(chosen_ind)         = [];
marginal_utility(chosen_ind) = [];

%%% build the priority queue
[sorted_marginal_utility_bounds, sort_ind] = sort(marginal_utility, 'descend');

%%% find the remaining points in the greedy batch
for i = 2:batch_size
    while true
        tmp_ind = sort_ind(1);
        tmp_marginal_utility = sum(log(loggend + probs(tmp_ind, 2:end)), 2) - jensen_utility;

        if tmp_marginal_utility >= sorted_marginal_utility_bounds(2)
            break;
        else
            % insert the point into the queue appropriately
            insert_ind = find(sorted_marginal_utility_bounds <= tmp_marginal_utility, 1);
            sorted_marginal_utility_bounds = ...
                [sorted_marginal_utility_bounds(2:(insert_ind - 1)); ...
                 tmp_marginal_utility; ...
                 sorted_marginal_utility_bounds(insert_ind:end)];

            sort_ind = [sort_ind(2:(insert_ind - 1)); tmp_ind; sort_ind(insert_ind:end)];
        end
    end

    batch_ind(i)   = test_ind(tmp_ind);
    loggend        = loggend + probs(tmp_ind, 2:end);
    jensen_utility = sum(log(loggend), 2);

    % remove the greedy candidate from the pool
    % probs(tmp_ind, :) = [];
    % test_ind(tmp_ind) = [];

    sorted_marginal_utility_bounds = sorted_marginal_utility_bounds(2:end);
    sort_ind                       = sort_ind(2:end);

    % decrement_mask           = (sort_ind > tmp_ind);
    % sort_ind(decrement_mask) = sort_ind(decrement_mask) - 1;
end
