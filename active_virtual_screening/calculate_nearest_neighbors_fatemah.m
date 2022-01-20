k = 20;

data_dir = './../data/fatemah/';

load([data_dir '/features']);
features = sparse(features(:, 2), features(:, 1), 1);
features = features(any(features, 2), :);  % remove features that are always zero

[nearest_neighbors, similarities] = jaccard_nn(features, k);

filename = sprintf('%snearest_neighbors_%i.mat', data_dir, k);

save(filename, 'nearest_neighbors', 'similarities', '-v7.3');
