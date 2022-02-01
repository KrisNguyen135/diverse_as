function [chosen_ind, chosen_prob, num_computed, num_pruned] = malkomes( ...
    problem, train_ind, train_labels, test_ind, model, ...
    nearest_neighbors, similarities, sim_threshold)

[probs, n, d] = model(problem, train_ind, train_labels, test_ind);

chosen_prob  = 0;
num_computed = 0;
num_pruned   = 0;

reverse_ind = zeros(problem.num_points, 1);
reverse_ind(test_ind) = 1:numel(test_ind);

%%% block out the region covered by the labeled set
for i = 1:numel(train_ind)
    this_ind = test_ind(i);
    cutoff   = find(similarities(this_ind, :) > sim_threshold, 1);

    if numel(cutoff) > 0
        this_nearest_neighbors = nearest_neighbors(this_ind, 1:cutoff);

        % make the unlabeled probabilities zero if the point is close to a labeled point
        this_nearest_neighbors_ind = reverse_ind(this_nearest_neighbors);
        this_nearest_neighbors_ind = this_nearest_neighbors_ind(this_nearest_neighbors_ind ~= 0);
        probs(this_nearest_neighbors_ind, :) = 0;
    end
end

best_score = 0;
chosen_ind = test_ind(1);

for i = 1:numel(test_ind)
    this_ind = test_ind(i);
    cutoff   = find(similarities(this_ind, :) > sim_threshold, 1);

    if numel(cutoff) > 0
        this_nearest_neighbors = nearest_neighbors(this_ind, 1:cutoff);
        this_nearest_neighbors_ind = reverse_ind(this_nearest_neighbors);
        this_nearest_neighbors_ind = this_nearest_neighbors_ind(this_nearest_neighbors_ind ~= 0);

        score = sum(probs(this_nearest_neighbors_ind, 2:end));

        if score > best_score
            best_score = score;
            chosen_ind = this_ind;
        end
    end
end
