function policy = get_policy(policy, varargin)

policy = @(problem, train_ind, train_labels, test_ind) ...
         policy(problem, train_ind, train_labels, test_ind, varargin{:});
