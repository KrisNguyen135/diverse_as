function model_update = get_model_update(model_update, varargin)

model_update = @(problem, n, d, ind, label, test_ind) ...
               model_update(problem, n, d, ind, label, test_ind, varargin{:});
