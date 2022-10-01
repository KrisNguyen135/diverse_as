function utility = weighted_utility(problem)

utility = log(problem.counts(2:end) + 1) * problem.weights;
