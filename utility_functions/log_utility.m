function utility = log_utility(problem)

utility = sum(log(problem.counts(2:end) + 1));  % the 1 at the end avoids negative infinity
