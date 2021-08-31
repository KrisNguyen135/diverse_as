if ~exist('exp', 'var'), exp = 1; end
if ~exist('size', 'var'), size =

addpath(genpath('../../'));
addpath(genpath('../../active_learning'));
addpath(genpath('../../active_search'));

[problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir);
rng(exp);
