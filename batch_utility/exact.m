function [samples, sample_weights, sample_counts, utility] = exact( ...
    problem, train_ind, train_labels, batch_ind, model, weights, utility_function)

weights(train_ind, :) = 0;

batch_size     = numel(batch_ind);
num_samples    = problem.num_classes ^ batch_size;
samples        = nan(batch_size, num_samples);
sample_weights = ones(1, num_samples);

[probs, ~, ~]  = model(problem, train_ind, train_labels, batch_ind);
all_probs      = repmat(probs, 1, 1, num_samples);  % batch elements by classes by samples

for i = 1:numel(batch_ind)
    chosen_ind             = batch_ind(i);
    weights(chosen_ind, :) = 0;
    updating_ind           = find(weights(batch_ind, chosen_ind));

    train_and_selected_ind = [train_ind; batch_ind(1:(i - 1))];
    tmp_num_samples        = problem.num_classes ^ (i - 1);

    sample_weights0 = sample_weights;
    for j = tmp_num_samples:-1:1
        tmp_probs = squeeze(all_probs(i, :, :));

        for fake_label = 1:problem.num_classes
            sample_ind = problem.num_classes * j - problem.num_classes + fake_label;

            samples(1:i, sample_ind)   = [samples(1:(i - 1), j); fake_label];
            sample_weights(sample_ind) = sample_weights0(j) * ...
                                         tmp_probs(fake_label, j);
            % if i == 2 && j == 1 && fake_label == 1
            %     fprintf('%d, %d, %d, %.4f, %.4f\n', i, j, fake_label, ...
            %         sample_weights(sample_ind), tmp_probs(fake_label, j));
            %     disp(sample_weights');
            % end

            labels_and_samples = [train_labels; samples(1:(i - 1), j); fake_label];

            [new_probs, ~, ~] = model( ...
                problem, [train_and_selected_ind; batch_ind(i)], ...
                labels_and_samples, batch_ind(updating_ind));
            all_probs(:, :, sample_ind) = all_probs(:, :, j);
            all_probs(updating_ind, :, sample_ind) = new_probs;
        end
    end

    sample_weights = sample_weights / ...
        sum(sample_weights(1:(tmp_num_samples * problem.num_classes)));
end

onehot = samples(:) == 1:problem.num_classes;
onehot = reshape(onehot, batch_size, num_samples, problem.num_classes);

sample_counts = reshape(sum(onehot, 1), num_samples, problem.num_classes);

utility = 0;
for i = 1:num_samples
    old_counts     = problem.counts;
    problem.counts = problem.counts + sample_counts(i, :);
    utility        = utility + sample_weights(i) * utility_function(problem);
    problem.counts = old_counts;
end
