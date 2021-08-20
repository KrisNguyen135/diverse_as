function chosen_ind = choose_one_batch(...
  data_name, policy, batch_size, num_queries, k ...
)
% data_name   = 'demo_real_iter1';
split_name = split(data_name, 'real_iter');
iteration = split_name{end};
demo = split_name{1};
disp(data_name);
% iteration   = 1;  % change this accordingly
% policy      = 1;  % greedy batch or 2 batch-ens
% batch_size  = 100;
% num_queries = 10;  % number of iterations you plan to perform
% k           = 100;

% the chosen indices will be saved in save_dir/save_name
iter_dir    = sprintf('./%siteration%s', demo, iteration);
save_dir    = sprintf('%s/recommended_batch', iter_dir);
save_name   = sprintf('%s_policy_%g_chosen_ind', data_name, policy);
if ~isdir(save_dir)
  mkdir(save_dir); 
end

addpath(genpath('../../'));

%%%%%%%%%%%%%%%%%% ignore these parameters for now %%%%%
max_num_samples = 16;
resample = 1;
limit = Inf;  % default: dont' limit
sort_upper = 0;  % default: don't sort by upper_bound
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

verbose = true;

%% load data
disp('loading data...');
[problem, labels, weights, alpha, nearest_neighbors, similarities] = ...
  load_data(data_name, k);

%% setup problem
problem.num_queries = num_queries;  % note this is the number of batch queries
problem.batch_size  = batch_size;
problem.verbose     = verbose;  % set to true for debugging/verbose output
problem.do_pruning  = 1;
problem.data_name   = data_name;
problem.score_path  = sprintf('%s/%s_scores', save_dir, problem.data_name);
if policy == 1
  fprintf('scores saved to %s\n', problem.score_path);
end

%% setup model and policy
model = get_model(@knn_model, weights, alpha);
model = get_model(@model_memory_wrapper, model);


%% ignore the pruning details for now
if (policy >= 2 && policy < 3) || (policy >= 4 && policy < 5)
  problem.resample   = resample;
  problem.limit      = limit;
  problem.sort_upper = sort_upper;
end
if ismember(policy, [2 4 31 32 33 34]) || ...
    (policy >= 2 && policy < 3) || (policy >= 4 && policy < 5)
  tight_level = 4;
  probability_bound = get_probability_bound_improved(...
    @knn_probability_bound_improved, ...
    tight_level, weights, nearest_neighbors', similarities', alpha);
else
  probability_bound = get_probability_bound(@knn_probability_bound, ...
    weights, full(max(weights)), alpha);
end

[query_strategy, selector] = get_policy(policy, problem, model, ...
  weights, probability_bound, max_num_samples);

train_ind = (1:length(labels))';
problem.num_initial = length(train_ind);
% get list of points to consider for querying this round
test_ind = selector(problem, train_ind, labels);
% select location(s) of next observation(s) from the given list
disp('computing the batch...');
chosen_ind = query_strategy(problem, train_ind, labels, test_ind);

savepath = fullfile(save_dir, save_name);
fprintf('saving the results to %s...\n', savepath);
fid = fopen(savepath, 'w');
for i = 1:length(chosen_ind)
  fprintf(fid, '%d\n', chosen_ind(i));
end
fclose(fid);
disp('done');
