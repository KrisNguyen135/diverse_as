num_inactive = 100000;  % number of inactive proteins to randomly subselect
k            = 500;     % number of nearest neighbors to compute

rng('default');

data_dir = './../data/';
if ~isdir(data_dir)
    data_dir  = '/storage1/garnett/Active/activelearning/quan/diverse_as/data/';
end
data_directory        = fullfile(data_dir, 'drug/processed/');
precomputed_directory = fullfile(data_dir, 'drug/precomputed/');

fingerprints = {'ecfp4', 'gpidaph3'};

% choose subset of data to use
load([data_directory fingerprints{1} '/labels']);

inactive_class = max(labels);

inactive_ind = find(labels == inactive_class);
active_ind   = find(labels ~= inactive_class);

to_keep = [inactive_ind(randperm(numel(inactive_ind), num_inactive)); ...
           active_ind];

labels = labels(to_keep);
labels = labels + 1;
labels(labels > 121) = 1;

num_proteins = max(labels) - 1;

for fingerprint = fingerprints
    tic;

    fprintf('processing fingerprint %s ...\n', fingerprint{:});

    load([data_directory fingerprint{:} '/features']);

    features = sparse(features(:, 2), features(:, 1), 1);

    features = features(:, to_keep);
    % remove features that are always zero
    features = features(any(features, 2), :);

    [nearest_neighbors, similarities] = jaccard_nn(features, k);

    filename = sprintf('%s%s_nearest_neighbors_%i.mat', ...
                       precomputed_directory, ...
                       fingerprint{:}, ...
                       num_inactive);

    save(filename, 'nearest_neighbors', 'similarities', 'labels');

    elapsed = toc;
    if (elapsed < 60)
      fprintf('done, took %is.\n', ceil(elapsed));
    else
      fprintf('done, took %0.1fm.\n', ceil(elapsed / 6) / 10);
    end

end
