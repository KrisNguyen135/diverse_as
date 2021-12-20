data_dir  = '../../data';
filename  = 'morgan_nearest_neighbors_100000.mat';
data_dir  = fullfile(data_dir, 'drug/precomputed');
data_path = fullfile(data_dir, filename);
load(data_path);

[counts, unique_labels] = hist(labels, unique(labels, 'stable'));

for i = 2:121
    fprintf('class %d: %.4f\n', i, counts(i) / numel(labels));
end

fprintf('average: %.4f\n', mean(counts(2:121)) / numel(labels));
