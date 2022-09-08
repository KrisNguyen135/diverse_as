function policy = get_he_carbonell_policy(nearest_neighbors, similarities, K)

if ~exist('K', 'var'), K =min([size(similarities, 2) 100]); end

% all the r' values are equal for the k-NN model
% it's easier to keep track of (1 - r')
recip_r_prime = min(similarities(:, K));

% number of large-enough weights in each row
n = sum(similarities >= recip_r_prime, 2);

policy = @(problem, train_ind, train_labels, test_ind) ...
         he_cabornell(problem, train_ind, train_labels, test_ind, ...
                      nearest_neighbors, similarities, recip_r_prime, n);
