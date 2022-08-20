function [chosen_ind, chosen_prob, num_computed, num_pruned] = round_robin_ens(...
    problem, train_ind, train_labels, ~, model, model_update, ...
    utility_upperbound_function, pruning, compute_limit, sample_limit)

function [tmp_utility, tmp_pruned] = compute_score(i, this_test_ind)
    this_reverse_ind   = reverse_ind(this_test_ind);
    fake_unlabeled_ind = unlabeled_ind;
    fake_unlabeled_ind(this_reverse_ind) = [];

    fake_train_ind = [train_ind; this_test_ind];

    % for regular model
    tmp_n = n;
    tmp_d = d;
    tmp_n(this_reverse_ind, :) = [];
    tmp_d(this_reverse_ind, :) = [];

    fake_utilities = zeros(problem.num_classes, 1);
    tmp_pruned     = false;
    for fake_label = 1:problem.num_classes
        [new_probs, new_n, new_d] = model_update(...
            problem, tmp_n, tmp_d, this_test_ind, fake_label, fake_unlabeled_ind);

        fake_utilities(fake_label) = sum(maxk(new_probs(:, target_class), remain_budget));
    end

    p = probs(this_reverse_ind, :);
    tmp_utility = p * fake_utilities + p(target_class);

    % for Fatemah's model
    % fake_utilities = zeros(problem.num_classes, 1);
    % tmp_pruned     = false;
    % for fake_label = 1:problem.num_classes
    %     [new_probs, new_n, new_d] = model(...
    %         problem, fake_train_ind, [train_labels; fake_label], fake_unlabeled_ind);
    %
    %     fake_utilities(fake_label) = sum(maxk(new_probs(:, target_class), remain_budget));
    % end
    %
    % p = probs(this_reverse_ind, :);
    % tmp_utility = p * fake_utilities + p(target_class);
end

% don't apply computational tricks unless everything is fully specified
if ~exist('utility_upperbound_function', 'var') || ~exist('pruning', 'var')
    pruning = false;
end
if ~exist('compute_limit', 'var'), compute_limit = Inf; end
if ~exist('sample_limit',  'var'), sample_limit  =   0; end

target_class = mod(size(train_ind, 1), problem.num_classes - 1) + 2;

num_computed = 0;
num_pruned   = zeros(1, problem.num_classes);  % # points pruned before being conditioned on a class label

% budget for each positive class
remain_budget = fix(...
    (problem.num_queries - (numel(train_ind) - problem.num_initial) - 1) ...
    / (problem.num_classes - 1));

unlabeled_ind = unlabeled_selector(problem, train_ind, []);
[probs, n, d] = model(problem, train_ind, train_labels, unlabeled_ind);

success_probs = probs(:, target_class);
[sorted_success_probs, top_ind] = sort(success_probs, 'descend');
idxc = cumsum([1 logical(diff(sorted_success_probs'))]);
top_ind = cell2mat(...
    accumarray(idxc', top_ind', [], ...
               @(sorted_success_probs){sorted_success_probs(randperm(numel(sorted_success_probs)))}));

test_ind      = unlabeled_ind(top_ind);
num_test      = numel(test_ind);

if remain_budget <= 0
    chosen_ind   = test_ind(1);
    chosen_prob  = probs(top_ind(1), :);
    num_computed = num_test;
    return;
end

reverse_ind = zeros(problem.num_points, 1);
reverse_ind(unlabeled_ind) = 1:num_test;

pruned = false(num_test, 1);
score  = -1;

for i = 1:num_test
    if pruning && pruned(i)
        num_pruned(1) = num_pruned(1) + 1;
        continue;
    end

    if i > compute_limit, break; end

    this_test_ind = test_ind(i);
    [tmp_utility, tmp_pruned] = compute_score(i, this_test_ind);

    if tmp_pruned
        num_pruned(2) = num_pruned(2) + 1;
        continue;
    end

    if tmp_utility > score
        score      = tmp_utility;
        chosen_ind = this_test_ind;
    end

    num_computed = num_computed + 1;
end

if i < num_test && sample_limit > 0
    candidates = (i:numel(test_ind));
    candidates = candidates(~pruned(candidates));
    if sample_limit < numel(candidates)
        candidates = sort(randsample(candidates, sample_limit));
    end

    for j = 1:numel(candidates)
        i = candidates(j);

        if pruned(i)
            num_pruned(1) = num_pruned(1) + 1;
            continue;
        end

        this_test_ind = test_ind(i);
        [tmp_utility, tmp_pruned] = compute_score(i, this_test_ind);

        if tmp_pruned
            num_pruned(2) = num_pruned(2) + 1;
            continue;
        end

        if tmp_utility > score
            score      = tmp_utility;
            chosen_ind = this_test_ind;
        end

        num_computed = num_computed + 1;
    end
end

chosen_prob = probs(reverse_ind(chosen_ind), :);

end
