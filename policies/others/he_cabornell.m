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

%%% block out the points near labeled points
for i = 1:numel(train_ind)
    this_ind = train_ind(i);
end

for i = 1:numel(test_ind)
    this_ind = test_ind(i);
end
