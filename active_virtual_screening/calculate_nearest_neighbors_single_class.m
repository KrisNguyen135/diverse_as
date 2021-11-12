num_inactive   = 100000;  % number of inactive proteins to randomly subselect
k              = 500;     % number of nearest neighbors to compute
inactive_class = 0;       % the label of the negatives
fingerprint    = 'single';

rng 'default';

data_dir = './../data/';
if ~isdir(data_dir)
    data_dir  = '/storage1/garnett/Active/activelearning/quan/diverse_as/data/';
end
data_directory        = fullfile(data_dir, 'drug/processed/');
precomputed_directory = fullfile(data_dir, 'drug/precomputed/');

load([data_directory fingerprint '/labels'])
labels = labels + 1;  % zero-indexing to one-indexing

%%% compute the nearest neighbors from the features
load([data_directory fingerprint '/features'])
features = sparse(features(:, 2), features(:, 1), 1);
features = features(any(features, 2), :);  % remove features that are always zero

num_proteins = max(labels) - 1

for protein_ind = 1:num_proteins
    tic;
    fprintf('\tcomputing nearest neighbors for protein #%i/%i (%i actives) ... ', ...
            protein_ind, num_proteins, nnz(labels == protein_ind + 1));

    filename = sprintf('%starget_%i_%s_nearest_neighbors_%i.mat', ...
                       precomputed_directory, ...
                       protein_ind, ...
                       fingerprint, ...
                       num_inactive);

    this_ind      = (labels == inactive_class + 1) | (labels == protein_ind + 1);
    this_features = features(:, this_ind);
    this_features = this_features(any(this_features, 2), :);

    this_labels = labels(this_ind);
    this_labels(this_labels ~= 1) = 2;

    [nearest_neighbors, similarities] = jaccard_nn(this_features, k);

    save(filename, 'nearest_neighbors', 'similarities', 'labels');

    elapsed = toc;
    fprintf('done, took %0.1fm.\n', ceil(elapsed / 6) / 10);
end
