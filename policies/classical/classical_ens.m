function [chosen_ind, chosen_prob, num_computed, num_pruned] = classical_ens(...
    problem, train_ind, train_labels, ~, model, model_update, ...
    probability_bound, limit)

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

        fake_utilities(fake_label) = sum(1 - mink(new_probs(:, 1), remain_budget));
    end

    p = probs(reverse_ind(this_test_ind), :);
    tmp_utility = p * fake_utilities + sum(p(2:end));
end

if ~exist('limit', 'var'), limit = Inf; end

num_computed = 0;
num_pruned   = [0 0];

remain_budget = problem.num_queries - (numel(train_ind) - problem.num_initial) - 1;

unlabeled_ind = unlabeled_selector(problem, train_ind, ~);
[probs, n, d] = model(problem, train_ind, train_labels, unlabeled_ind);

failure_probs = probs(:, 1);
[~, top_ind]  = sort(failure_probs, 'ascend');
test_ind      = unlabeled_ind(top_ind);
num_test      = numel(test_ind);

if remain_budget <= 0
    chosen_ind   = test_ind(1);
    chosen_prob  = probs(top_ind(1), :);
    num_computed = num_test;
    num_pruned   = 0;
    return;
end

reverse_ind = zeros(problem.num_points, 1);
reverse_ind(unlabeled_ind) = 1:num_test;

prob_upper_bound = probability_bound(...
    problem, train_ind, train_labels, test_ind, 1, remain_budget);

future_utility_if_neg = sum(1 - failure_probs(top_ind(1:remain_budget)));
if problem.max_num_influence >= remain_budget
    future_utility_if_pos = sum(prob_upper_bound(1:remain_budget));
else
    tmp_ind = top_ind(1:(remain_budget - problem.max_num_influence));
    future_utility_if_pos = sum(1 - failure_probs(tmp_ind)) + ...
                            sum(prob_upper_bound((1:problem.max_num_influence)));
end

future_utility = (1 - failure_probs) * future_utility_if_pos + ...
                      failure_probs  * future_utility_if_neg;

score_upper_bound = (1 - failure_probs) + future_utility;
score_upper_bound = score_upper_bound(top_ind)

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
