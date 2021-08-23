function model = get_model(model, varargin)

model = @(problem, train_ind, train_labels, test_ind) ...
        model(problem, train_ind, train_labels, test_ind, varargin{:});
