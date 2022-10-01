% also assumes independence
function utility = jensen(problem, train_ind, train_labels, batch_ind, model)

% assert(strcmp(problem.utility, 'log'), 'only log utility is currently supported');

[probs, ~, ~] = model(problem, train_ind, train_labels, batch_ind);

if strcmp(problem.utility, 'log')
    utility = sum(log(problem.counts(2:end) + 1 + sum(probs(:, 2:end), 1)));
elseif strcmp(problem.utility, 'sqrt')
    utility = sum(sqrt(problem.counts(2:end) + sum(probs(:, 2:end), 1)));
elseif strcmp(problem.utility, 'weighted')
    utility = log(problem.counts(2:end) + 1 + sum(probs(:, 2:end), 1)) ...
              * problem.weights;
end
