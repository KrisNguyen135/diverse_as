function batch_ind = jensen_greedy(problem, train_ind, train_labels, test_ind, ...
                                   batch_size, model)

assert(strcmp(problem.utility, 'log'), 'only log utility is currently supported');

batch_ind      = nan(batch_size, 1);
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

% add the greedy candidate to the running batch
batch_ind(1)   = test_ind(chosen_ind);
loggend        = loggend + probs(chosen_ind, 2:end);
jensen_utility = sum(log(loggend), 2);

% remove the greedy candidate from the pool
probs(chosen_ind, :)         = [];
test_ind(chosen_ind)         = [];
marginal_utility(chosen_ind) = [];

%%% build the priority queue
priority_queue = PriorityQueue();
for ind = 1:numel(test_ind)
    priority_queue.push(ind, marginal_utility(ind));
end

%%% find the remaining points in the greedy batch
for i = 2:batch_size
    while true
        tmp_ind = priority_queue.top();
        priority_queue.pop();

        tmp_marginal_utility = sum(log(loggend + probs(tmp_ind, 2:end)), 2) - jensen_utility;
        if tmp_marginal_utility >= priority_queue.top_value()
            break;
        else
            priority_queue.push(tmp_ind, tmp_marginal_utility);
        end
    end

    batch_ind(i)   = test_ind(tmp_ind);
    loggend        = loggend + probs(tmp_ind, 2:end);
    jensen_utility = sum(log(loggend), 2);
end
