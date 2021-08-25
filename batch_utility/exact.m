function [samples, utility] = exact(problem, train_ind, train_labels, batch_ind, ...
    model, model_update, weights)

weights(train_ind, :) = 0;

batch_size     = numel(batch_ind)
num_samples    = problem.num_classes ^ batch_size
samples        = nan(batch_size, num_samples);
sample_counts  = repmat(problem.counts, num_samples, 1);
sample_weights = ones(1, num_samples);

[probs, ~, ~]  = model(problem, train_ind, train_labels, batch_ind);
all_probs      = repmat(probs, 1, 1, num_samples);

for i = 1:numel(batch_ind)
    chosen_ind             = batch_ind(i);
    weights(chosen_ind, :) = 0;
    updating_ind           = find(weights(:, chosen_ind));

    train_and_selected_ind = [train_ind; batch_ind(1:(i - 1))];
    tmp_num_samples        = min(problem.num_classes ^ (i - 1), num_samples);

    sample_weights0 = sample_weights;
    for j = tmp_num_samples:-1:1
        for fake_label = 1:problem.num_classes
            sample_ind = problem.num_classes * j - problem.num_classes + fake_label;

            samples(1:i, sample_ind)   = [samples(1:(i - 1), j); fake_label];
            sample_weights(sample_ind) = sample_weights0(j) * ...
                                         all_probs(i, j, fake_label);

            labels_and_samples = [train_labels; samples(1:(i - 1), j); fake_label];

            [new_probs, ~, ~] = model( ...
                problem, [train_and_selected_ind; batch_ind(i)], ...
                labels_and_samples, updating_ind);
            all_probs(:, :, sample_ind) = all_probs(:, :, j);
            all_probs(updating_ind, :, sample_ind) = new_probs;

            sample_counts(sample_ind, fake_label) = ...
                sample_counts(sample_ind, fake_label) + 1;
        end
    end

    sample_weights = sample_weights / sum(sample_weights(1:tmp_num_samples));
end

samples
sample_counts
% sample_weights

utility = 0;
