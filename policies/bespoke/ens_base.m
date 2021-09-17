function [chosen_ind, chosen_prob, num_computed, num_pruned] = ens_base( ...
    problem, train_ind, train_labels, test_ind, model, ...
    batch_policy, batch_utility_function, ...
    utility_upperbound_function, pruning, compute_limit, sample_limit)

% don't apply computational tricks unless everything is fully specified
if ~exist('utility_upperbound_function', 'var') || ~exist('pruning', 'var')
    pruning = false;
end
if ~exist('compute_limit', 'var'), compute_limit = Inf; end
if ~exist('sample_limit',  'var'), sample_limit  =   0; end

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);

num_computed = 0;
num_pruned   = zeros(1, problem.num_classes);  % # points pruned before being conditioned on a class label

remain_budget = problem.num_queries - (numel(train_ind) - problem.num_initial) - 1;
if remain_budget <= 0  % greedy
    switch problem.utility
    case 'log'
        marginal_utility = probs(:, 2:end) * (log(problem.counts(2:end) + 2)' ...
                                              - log(problem.counts(2:end) + 1)');
    end

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

if pruning
    % reverse indexing is used for reverse probability lookup later on
    % old_test_ind = test_ind;
    % reverse_ind = zeros(problem.num_points, 1);
    % reverse_ind(test_ind) = 1:numel(test_ind);

    bounds = utility_upperbound_function(problem, train_ind, train_labels, test_ind, ...
        probs, n, d, remain_budget);  % row vector of length problem.num_classes

    upperbounds = sum(probs .* bounds, 2);  % column vector of length test size

    % sort candidates by upper bounds, so that no unnecessary computation
    % will be done
    [upperbounds, top_ind] = sort(upperbounds, 'descend');

    test_ind = test_ind(top_ind);
    probs    = probs(top_ind, :);

    pruned   = false(numel(test_ind), 1);
end

% if the maximum number of computations we want to do is less than the size of
% the candidate pool, take the first `compute_limit` candidates and randomly
% sample `sample_limit` of the remaining candidates
if compute_limit + sample_limit < numel(test_ind)
    % save the complete candidate pool
    old_test_ind = test_ind;
    old_probs    = probs;

    % get the top `compute_limit` candidates
    compute_ind = 1:compute_limit;
    test_ind    = test_ind(compute_ind, :);
    probs       = probs(compute_ind, :);
end

best_utility = -1;
chosen_ind   = -1;
chosen_prob  = 0;

for i = 1:numel(test_ind)
    % misc. housekeeping
    if pruning && pruned(i)
        num_pruned(1) = num_pruned(1) + 1;
        continue;
    end
    if i > compute_limit, break; end

    this_test_ind      = test_ind(i);
    fake_train_ind     = [train_ind; this_test_ind];
    fake_unlabeled_ind = unlabeled_selector(problem, fake_train_ind, []);

    % fprintf('computing %d-th point %d:\n', i, this_test_ind);

    % fake_utilities    = zeros(problem.num_classes, 1);
    running_utility = 0;
    if pruning
        pruned_on_the_fly = false;
        remain_bound      = upperbounds(i);
    end

    for fake_label = 1:problem.num_classes
        old_counts = problem.counts;

        fake_train_labels          = [train_labels; fake_label];
        problem.counts(fake_label) = problem.counts(fake_label) + 1;

        batch_ind = batch_policy(problem, fake_train_ind, fake_train_labels, ...
                                 fake_unlabeled_ind, remain_budget);

        batch_utility = batch_utility_function( ...
            problem, fake_train_ind, fake_train_labels, batch_ind);

        % fake_utilities(fake_label) = batch_utility;
        % fprintf('\n\tfake label %d, utility %.4f\n\t', fake_label, batch_utility);
        % disp(reshape(batch_ind, 1, remain_budget));

        problem.counts  = old_counts;
        tmp_prob        = probs(i, fake_label);
        tmp_utility     = tmp_prob * batch_utility;
        running_utility = running_utility + tmp_utility;

        if pruning && fake_label < problem.num_classes
            remain_bound = remain_bound - tmp_prob * bounds(fake_label);

            if running_utility + remain_bound <= best_utility
                % fprintf('\tpruned after label %d\n', fake_label);
                pruned_on_the_fly          = true;
                num_pruned(fake_label + 1) = num_pruned(fake_label + 1) + 1;
                break;
            end
        end
    end

    if ~pruning || ~pruned_on_the_fly
        % running_utility = probs(i, :) * fake_utilities;
        % fprintf('\tavg utility: %.4f\n', running_utility);

        if pruning
            assert(running_utility <= upperbounds(i));
        end

        num_computed = num_computed + 1;

        if running_utility > best_utility
            best_utility = running_utility;
            chosen_ind   = this_test_ind;
            chosen_prob  = probs(i, :);

            if pruning
                pruned(upperbounds <= best_utility) = true;
            end
        end
    end
end

% get the exact number of `num_pruned(1)` as there might be pruned candidates
% that have been left out of the count when there's a computational limit
num_pruned(1) = num_pruned(1) + sum(pruned(i:end));

if i < numel(test_ind) && sample_limit > 0
    candidates = (i:numel(test_ind));
    candidates = candidates(~pruned(candidates));
    if sample_limit < numel(candidates)
        candidates = sort(randsample(candidates, sample_limit));
    end

    for j = 1:numel(candidates)
        i = candidates(j);

        % misc. housekeeping
        if pruning && pruned(i)
            num_pruned(1) = num_pruned(1) + 1;
            continue;
        end
        if i > compute_limit, break; end

        this_test_ind      = test_ind(i);
        fake_train_ind     = [train_ind; this_test_ind];
        fake_unlabeled_ind = unlabeled_selector(problem, fake_train_ind, []);

        % fprintf('computing %d-th point %d:\n', i, this_test_ind);

        % fake_utilities    = zeros(problem.num_classes, 1);
        running_utility = 0;
        if pruning
            pruned_on_the_fly = false;
            remain_bound      = upperbounds(i);
        end

        for fake_label = 1:problem.num_classes
            old_counts = problem.counts;

            fake_train_labels          = [train_labels; fake_label];
            problem.counts(fake_label) = problem.counts(fake_label) + 1;

            batch_ind = batch_policy(problem, fake_train_ind, fake_train_labels, ...
                                     fake_unlabeled_ind, remain_budget);

            batch_utility = batch_utility_function( ...
                problem, fake_train_ind, fake_train_labels, batch_ind);

            % fake_utilities(fake_label) = batch_utility;
            % fprintf('\n\tfake label %d, utility %.4f\n\t', fake_label, batch_utility);
            % disp(reshape(batch_ind, 1, remain_budget));

            problem.counts  = old_counts;
            tmp_prob        = probs(i, fake_label);
            tmp_utility     = tmp_prob * batch_utility;
            running_utility = running_utility + tmp_utility;

            if pruning && fake_label < problem.num_classes
                remain_bound = remain_bound - tmp_prob * bounds(fake_label);

                if running_utility + remain_bound <= best_utility
                    % fprintf('\tpruned after label %d\n', fake_label);
                    pruned_on_the_fly          = true;
                    num_pruned(fake_label + 1) = num_pruned(fake_label + 1) + 1;
                    break;
                end
            end
        end

        if ~pruning || ~pruned_on_the_fly
            % running_utility = probs(i, :) * fake_utilities;
            % fprintf('\tavg utility: %.4f\n', running_utility);

            if pruning
                assert(running_utility <= upperbounds(i));
            end

            num_computed = num_computed + 1;

            if running_utility > best_utility
                best_utility = running_utility;
                chosen_ind   = this_test_ind;
                chosen_prob  = probs(i, :);

                if pruning
                    pruned(upperbounds <= best_utility) = true;
                end
            end
        end
    end
end
