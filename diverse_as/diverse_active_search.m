function [utilities, queries, queried_probs, computed, pruned] = diverse_active_search( ...
    problem, train_ind, train_labels, labels, selector, utility_function, policy, message_prefix)

if ~exist('message_prefix', 'var'), message_prefix = ''; end
verbose = isfield(problem, 'verbose') && problem.verbose;

utilities     = [];
queries       = [];
queried_probs = [];
computed      = [];
pruned        = [];

for i = 1:problem.num_queries
    if verbose
        tic;
        fprintf('%sIteration %d: ', message_prefix, i);
    end

    fprintf('here0\n');
    test_ind = selector(problem, train_ind, []);
    fprintf('here1\n');

    if isempty(test_ind)
        warning('mf_active_search:no_points_selected', ...
                ['after %d steps, no points were selected. ' ...
                 'Ending run early.'], i);
        return;
    end

    [chosen_ind, chosen_prob, num_computed, num_pruned] = ...
        policy(problem, train_ind, train_labels, test_ind);

    chosen_label = labels(chosen_ind);

    train_ind    = [train_ind;      chosen_ind];
    train_labels = [train_labels; chosen_label];

    problem.counts(chosen_label) = problem.counts(chosen_label) + 1;

    utilities     = [utilities; utility_function(problem)];
    queries       = [queries; chosen_ind];
    queried_probs = [queried_probs; chosen_prob];
    computed      = [computed; num_computed];
    pruned        = [pruned; num_pruned];

    if verbose
        fprintf('(%d, %d) chosen. Took %.4f seconds.\n', ...
                chosen_ind, chosen_label, toc);
    end
end
