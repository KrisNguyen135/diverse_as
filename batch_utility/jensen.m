% also assumes independence
function utility = jensen(problem, train_ind, train_labels, batch_ind, model)

assert(strcmp(problem.utility, 'log'), 'only log utility is currently supported');

[probs, ~, ~] = model(problem, train_ind, train_labels, batch_ind);

utility = sum(log(problem.counts(2:end) + 1 + sum(probs(:, 2:end))));
