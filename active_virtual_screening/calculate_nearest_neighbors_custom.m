num_inactive   = 100000;  % number of inactive proteins to randomly subselect
k              = 10000;     % number of nearest neighbors to compute
inactive_class = 0;       % the label of the negatives
fingerprint    = 'morgan';

rng 'default';

data_dir = './../data/';
if ~isdir(data_dir)
    data_dir  = '/storage1/garnett/Active/activelearning/quan/diverse_as/data/';
end
data_directory        = fullfile(data_dir, 'drug/processed/');
precomputed_directory = fullfile(data_dir, 'drug/precomputed/');

load([data_directory fingerprint '/labels'])
labels = labels + 1;  % zero-indexing to one-indexing

%%% relabel the classes
[label_counts, label_vals] = hist(labels, unique(labels, 'stable'));

reverse_ind = zeros(max(label_vals), 1);
reverse_ind(label_vals) = (1:numel(label_vals))';

labels = reverse_ind(labels);

%%% compute the nearest neighbors from the features
load([data_directory fingerprint '/features'])
features = sparse(features(:, 2), features(:, 1), 1);
features = features(any(features, 2), :);  % remove features that are always zero

[nearest_neighbors, similarities] = jaccard_nn(features, k);

filename = sprintf('%s%s_nearest_neighbors_%i_%i.mat', ...
                   precomputed_directory, ...
                   fingerprint, ...
                   num_inactive, ...
                   k);

save(filename, 'nearest_neighbors', 'similarities', 'labels');
