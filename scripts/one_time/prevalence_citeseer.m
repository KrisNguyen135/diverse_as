if ~exist('group_size', 'var'), group_size = 9; end
if ~exist('data',       'var'), data       = 'citeseer'; end
if ~exist('utility',    'var'), utility    = 'log'; end

addpath(genpath('../../'));
addpath(genpath('../../active_learning'));
addpath(genpath('../../active_search'));

data_dir = '../../data/'
[problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir, 1, group_size);
