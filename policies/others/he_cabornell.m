% only for morgan
% assumes the last columns in the `nearest_neighbors` and `similarities`
% are for the K-th nearest neighbor
function [chosen_ind, chosen_prob, num_computed, num_pruned] = he_cabornell( ...
    problem, train_ind, train_labels, test_ind, alpha, nearest_neighbors, similarities)

nearest_neighbors = nearest_neighbors';
similarities      = similarities';

% all the r' values are equal for the k-NN model
% it's easier to keep track of (1 - r')
recip_r_prime = min(similarities(:, end));

% number of large-enough weights in each row
n = sum(similarities >= recip_r_prime, 2);
