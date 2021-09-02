function batch_policy = get_batch_policy(policy, varargin)

batch_policy = @(problem, train_ind, train_labels, test_ind, batch_size) ...
    policy(problem, train_ind, train_labels, test_ind, batch_size, varargin{:});
