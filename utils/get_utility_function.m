function utility_function = get_utility_function(utility_function, varargin)

utility_function = @(problem) utility_function(problem, varargin{:});
