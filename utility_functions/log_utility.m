function utility = log_utility(problem)

utility = sum(log(problem.counts(2:end)));
