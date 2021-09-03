function [chosen_ind, chosen_prob, num_computed, num_pruned] = ens_base( ...
    problem, train_ind, train_labels, test_ind, model, ...
    batch_policy, batch_utility_function, limit)

if ~exist('limit', 'var'), limit = Inf; end

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);
switch problem.utility
case 'log'
    marginal_utility = probs(:, 2:end) * ...
                       (log(problem.counts(2:end) + 2)' - log(problem.counts(2:end) + 1)');
end

if limit < numel(test_ind)
    [marginal_utility, limit_ind] = maxk(marginal_utility, limit);

    test_ind = test_ind(limit_ind);
    probs    = probs(limit_ind)
    n        = n(limit_ind);
    d        = d(limit_ind);
end

num_computed = numel(test_ind);
num_pruned   = [0 0];

remain_budget = problem.num_queries - (numel(train_ind) - problem.num_initial) - 1;
if remain_budget <= 0  % greedy
    maxu    = max(marginal_utility);
    indices = find(marginal_utility == maxu);
    if numel(indices) > 1
        chosen_ind = randsample(indices, 1);
    else
        chosen_ind = indices;
    end

    chosen_prob = probs(chosen_ind, :);
    chosen_ind  = test_ind(chosen_ind);

    return
end

best_utility = -1;
chosen_ind   = -1;
chosen_prob  = 0;

for i = 1:numel(test_ind)
    if i > limit, break; end

    this_test_ind      = test_ind(i);
    fake_train_ind     = [train_ind; this_test_ind];
    fake_unlabeled_ind = unlabeled_selector(problem, fake_train_ind, []);

    % fprintf('computing %d-th point %d:', i, this_test_ind);

    fake_utilities = zeros(problem.num_classes, 1);
    for fake_label = 1:problem.num_classes
        old_counts = problem.counts;

        fake_train_labels          = [train_labels; fake_label];
        problem.counts(fake_label) = problem.counts(fake_label) + 1;

        batch_ind = batch_policy(problem, fake_train_ind, fake_train_labels, ...
                                 fake_unlabeled_ind, remain_budget);

        batch_utility = batch_utility_function( ...
            problem, fake_train_ind, fake_train_labels, batch_ind);

        fake_utilities(fake_label) = batch_utility;
        % fprintf('\n\tfake label %d, utility %.4f\n\t', fake_label, batch_utility);
        % disp(reshape(batch_ind, 1, remain_budget));

        problem.counts = old_counts;
    end

    tmp_utility = probs(i, :) * fake_utilities;
    % fprintf('\tavg utility: %.4f\n', tmp_utility);

    if tmp_utility > best_utility
        best_utility = tmp_utility;
        chosen_ind   = this_test_ind;
        chosen_prob  = probs(i, :);
    end
end
