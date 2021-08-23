function utility = threshold_utility(problem, threshold)

utility = sum(min(problem.counts(2:end), threshold))
