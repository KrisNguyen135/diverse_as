num_inactive = 100000;  % number of inactive proteins to randomly subselect
k            = 500;     % number of nearest neighbors to compute
group_size   = 4;       % number of positive classes in a problem

rng('default');

data_directory        = './../data/drug/processed/';
precomputed_directory = './../data/drug/precomputed/';

fingerprints = {'ecfp4', 'gpidaph3'};

% choose subset of data to use
load([data_directory fingerprints{1} '/labels']);

inactive_class = max(labels);

inactive_ind = find(labels == inactive_class);
active_ind   = find(labels ~= inactive_class);

to_keep = [inactive_ind(randperm(numel(inactive_ind), num_inactive)); ...
           active_ind];

labels = labels(to_keep);

num_proteins = max(labels) - 1;
assert(mod(num_proteins, group_size) == 0, ...
       sprintf('number of proteins (%d) must be divisible by group size (%d)', ...
               num_proteins, group_size));

for fingerprint = fingerprints
  fprintf('processing fingerprint %s ...\n', fingerprint{:});

  load([data_directory fingerprint{:} '/features']);

  features = sparse(features(:, 2), features(:, 1), 1);

  features = features(:, to_keep);
  % remove features that are always zero
  features = features(any(features, 2), :);

  shuffled_proteins = randperm(num_proteins);

  for group_ind = 1:(num_proteins / group_size)
    tic;
    fprintf('\tcomputing nearest neighbors for protein group #%i/%i ... ', ...
            group_ind, num_proteins / group_size);

    filename = sprintf('%starget_%i_%s_nearest_neighbors_%i.mat', ...
                       precomputed_directory, ...
                       group_ind, ...
                       fingerprint{:}, ...
                       num_inactive);

    if (exist(filename, 'file') > 0)
      fprintf('file already exists!\n');
      continue;
    end

    this_ind = (labels == inactive_class);
    for protein_ind = 1:group_size
        this_ind = this_ind | (labels == shuffled_proteins(protein_ind));
    end
    this_features = features(:, this_ind);
    this_features = this_features(any(this_features, 2), :);

    [nearest_neighbors, similarities] = jaccard_nn(this_features, k);

    save(filename, 'nearest_neighbors', 'similarities');

    elapsed = toc;
    if (elapsed < 60)
      fprintf('done, took %is.\n', ceil(elapsed));
    else
      fprintf('done, took %0.1fm.\n', ceil(elapsed / 6) / 10);
    end
  end

end
