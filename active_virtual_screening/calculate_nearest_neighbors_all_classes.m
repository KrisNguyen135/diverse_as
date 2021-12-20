if ~exist('group_size', 'var')
    group_size = 1;
end

if group_size == 1
    num_exps = 120;
else
    num_exps = 20;
end

num_inactive   = 100000;
k              = 500;     % number of nearest neighbors to compute
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

for i = 1:num_exps
    tic;
    fprintf('processing exp %i of group size %i...', i, group_size);

    filename = sprintf('%smorgan_%i_%i_%s_nearest_neighbors.mat', ...
                       precomputed_directory, ...
                       group_size, ...
                       i, ...
                       fingerprint);

    if group_size == 1
        pos_mask         = (labels == i + 1);
        labels(:)        = 1;
        labels(pos_mask) = 2;
    else
        rng(group_ind);

        selected_classes = randperm(120, group_size);
        selected_classes

        old_labels = labels;

        labels(:) = 1;
        for class_ind = 1:group_size
            pos_mask         = (old_labels == selected_classes(class_ind) + 1);
            labels(pos_mask) = class_ind + 1;
        end
    end

    % remove points from untargeted classes
    keep_ind = false(numel(labels), 1);
    keep_ind(1:num_negatives) = true;

    pos_ind           = find(labels > 1);
    num_positives     = numel(pos_ind);
    keep_ind(pos_ind) = true;

    labels            = labels(keep_ind);

    load([data_directory fingerprint '/features']);
    features = sparse(features(:, 2), features(:, 1), 1);
    features = features(any(features, 2), :);  % remove features that are always zero

    [nearest_neighbors, similarities] = jaccard_nn(features, k);

    save(filename, 'nearest_neighbors', 'similarities', 'labels');

    fprintf('took %0.1fm.\n', ceil(toc / 6) / 10);
end
