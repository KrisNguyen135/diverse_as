num_inactive = 100000;  % number of inactive proteins to randomly subselect
k            = 500;     % number of nearest neighbors to compute
group_size   = 120;       % number of positive classes in a problem

rng(group_size);

data_dir = './../data/';
if ~isdir(data_dir)
    data_dir  = '/storage1/garnett/Active/activelearning/quan/diverse_as/data/';
end
processed_dir   = fullfile(data_dir, 'drug/processed/');
precomputed_dir = fullfile(data_dir, 'drug/precomputed/');

fingerprints = {'ecfp4', 'gpidaph3'};

% choose subset of data to use
load([processed_dir fingerprints{1} '/labels']);

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

  load([processed_dir fingerprint{:} '/features']);

  features = sparse(features(:, 2), features(:, 1), 1);
  features = features(:, to_keep);
  % remove features that are always zero
  features = features(any(features, 2), :);

  shuffled_proteins = randperm(num_proteins);

  for group_ind = 1:(num_proteins / group_size)
    tic;
    fprintf('\tcomputing nearest neighbors for protein group #%i/%i\n', ...
            group_ind, num_proteins / group_size);
    tmp_proteins = shuffled_proteins((group_ind * group_size - group_size + 1) ...
                                     :(group_ind * group_size));
    tmp_proteins_str = sprintf('-%d', tmp_proteins);
    fprintf('\tactive classes in group: %s\n', tmp_proteins_str(2:end));

    filename = sprintf('%sgroup_%i_%s_nearest_neighbors_%i.mat', ...
                       precomputed_dir, group_ind, fingerprint{:}, num_inactive);

    if (exist(filename, 'file') > 0)
      fprintf('\tfile already exists!\n');
      continue;
    end

    % filter out the selected classes
    this_ind = (labels == inactive_class);
    reverse_label = zeros(num_proteins + 1, 1);
    reverse_label(inactive_class) = 1;
    for protein_ind = 1:group_size
        this_ind = this_ind | (labels == tmp_proteins(protein_ind));
        reverse_label(tmp_proteins(protein_ind)) = protein_ind + 1;
    end
    this_features = features(:, this_ind);
    this_features = this_features(any(this_features, 2), :);

    this_labels = labels(this_ind, :);
    this_labels = reverse_label(this_labels);
    fprintf('\t%d inactives, %d actives\n', sum(this_labels == 1), sum(this_labels ~= 1));

    [nearest_neighbors, similarities] = jaccard_nn(this_features, k);

    save(filename, 'nearest_neighbors', 'similarities', 'this_labels');

    elapsed = toc;
    if (elapsed < 60)
      fprintf('\tdone, took %is.\n', ceil(elapsed));
    else
      fprintf('\tdone, took %0.1fm.\n', ceil(elapsed / 6) / 10);
    end
  end
end
