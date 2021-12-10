addpath(genpath('../../'));

probs = [[0.4, 0.5]; [0.6, 0.2]; [0.1, 0.8]];
loggend = [4.2 4.7];
jensen_utility = sum(log(loggend));
batch_size = 2;
init_score = [0.21356, 0.20171, 0.22581];
n = 3;
num_pos_classes = 2;

lazy_greedy(probs', loggend(:, :), jensen_utility(:), batch_size, init_score, n, num_pos_classes)
