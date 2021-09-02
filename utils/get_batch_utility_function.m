function batch_utility_function = get_batch_utility_function(utility_function, varargin)

batch_utility_function = @(problem, train_ind, train_labels, batch_ind) ...
    utility_function(problem, train_ind, train_labels, batch_ind, varargin{:});
