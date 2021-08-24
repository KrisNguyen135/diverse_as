function [chosen_ind, chosen_prob, num_computed, num_pruned] = ens_base( ...
    problem, train_ind, train_labels, test_ind, model, model_update, ...
    batch_function, utility_function, utility_bound, limit, lookahead)

function [tmp_utility, tmp_pruned] = compute_score(i, this_test_ind)
    this_reverse_ind   = reverse_ind(this_test_ind);
    fake_unlabeled_ind = unlabeled_ind;
    fake_unlabeled_ind(this_reverse_ind) = [];

    tmp_n = n;
    tmp_d = d;
    tmp_n(this_reverse_ind, :) = [];
    tmp_d(this_reverse_ind, :) = [];

    fake_utilities = zeros(problem.num_classes, 1);
    tmp_pruned     = false;
    for fake_label = 1:problem.num_classes
        [new_probs, new_n, new_d] = model_update(...
            problem, tmp_n, tmp_d, this_test_ind, fake_label, fake_unlabeled_ind);

        fake_train_ind    = [train_ind; this_test_ind];
        fake_train_labels = [train_labels; fake_label];

        % utility should be marginal
        [batch_ind, batch_utility] = batch_function( ...
            problem, fake_train_ind, fake_train_labels, fake_unlabeled_ind, ...
            new_probs, new_n, new_d, model, model_update, lookahead);

        fake_utilities(fake_label) = batch_utility;
    end

    p = probs(this_reverse_ind, :)
    tmp_utility = p * fake_utilities + p(2:end) * ...
                 (log(problem.counts(2:end) + 1)' - log(problem.counts(2:end))');
end

if ~exist('limit', 'var'), limit = Inf; end

num_computed = 0;
num_pruned   = [0 0];

remain_budget = problem.num_queries - (numel(train_ind) - problem.num_initial) - 1;
if exist('lookahead', 'var'), remain_budget = min(remain_budget, lookahead); end

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);

failure_probs   = probs(:, 1);
[~, top_ind]    = sort(failure_probs, 'ascend');
sorted_test_ind = unlabeled_ind(top_ind);
num_test        = numel(test_ind);

if remain_budget <= 0
    chosen_ind   = sorted_test_ind(1);
    chosen_prob  = probs(sorted_top_ind(1), :);
    num_computed = num_test;
    num_pruned   = 0;
    return;
end

reverse_ind = zeros(problem.num_points, 1);
reverse_ind(test_ind) = 1:num_test;

pruned = false(num_test, 1);
score  = -1;

for i = 1:num_test
    if pruned(i)
        num_pruned(1) = num_pruned(1) + 1;
        continue;
    end

    if i > limit, break; end

    this_test_ind = test_ind(i);
    [tmp_utility, tmp_pruned] = compute_score(i, this_test_ind);

    if tmp_pruned
        num_pruned(2) = num_pruned(2) + 1;
        continue;
    end

    if tmp_utility > score
        score      = tmp_utility;
        chosen_ind = this_test_ind;
        pruned(score_upper_bound <= score) = true;
    end

    num_computed = num_computed + 1;
end

chosen_prob = probs(reverse_ind(chosen_ind), :);

end
