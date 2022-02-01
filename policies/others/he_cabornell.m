% only for morgan
% assumes the last columns in the `nearest_neighbors` and `similarities`
% are for the K-th nearest neighbor
function [chosen_ind, chosen_prob, num_computed, num_pruned] = he_cabornell( ...
    problem, train_ind, train_labels, test_ind, ...
    nearest_neighbors, similarities, recip_r_prime, n)

% target only uncovered classes
% in this case, they are the classes with lower counts
% than the most frequently observed class
uncovered_mask = (problem.counts(2:end) == max(problem.counts(2:end)));

%%% book-keeping
scores = zeros(numel(test_ind), 1);
reverse_ind = zeros(problem.num_points, 1);
reverse_ind(test_ind) = 1:numel(test_ind);

%%% block out the points near labeled points
for i = 1:numel(train_ind)
    this_ind = train_ind(i);
    cutoff   = find(similarities(this_ind, :) <= recip_r_prime, 1) - 1;

    if numel(cutoff) > 0
        this_nearest_neighbors = nearest_neighbors(this_ind, 1:cutoff);

        % leave out near-by points
        this_nearest_neighbors_ind = reverse_ind(this_nearest_neighbors);
        this_nearest_neighbors_ind = this_nearest_neighbors_ind(this_nearest_neighbors_ind ~= 0);
        scores(this_nearest_neighbors_ind) = -Inf;
    end
end

%%% acquisition funtion
for i = 1:numel(test_ind)
    if scores(i) == 0  % if not blocked out
        this_ind = test_ind(i);
        expanded_recip_r_prime = recip_r_prime ...
                                 * (numel(train_ind) - problem.num_initial + 1);
        cutoff = find(similarities(this_ind, :) <= expanded_recip_r_prime, 1) - 1;
    end
end
