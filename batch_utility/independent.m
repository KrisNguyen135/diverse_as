% function [samples, sample_weights, sample_counts, utility] = independent( ...
%     problem, train_ind, train_labels, batch_ind, model, utility_function)
function utility = independent(problem, train_ind, train_labels, batch_ind, ...
                               model, utility_function)

batch_size     = numel(batch_ind);
num_samples    = problem.num_classes ^ batch_size;
samples        = nan(batch_size, num_samples);
sample_weights = ones(1, num_samples);

[probs, ~, ~] = model(problem, train_ind, train_labels, batch_ind);

for i = 1:batch_size
    tmp_num_samples = problem.num_classes ^ (i - 1);
    sample_weights0 = sample_weights;

    for j = tmp_num_samples:-1:1
        for fake_label = 1:problem.num_classes
            sample_ind = problem.num_classes * j - problem.num_classes + fake_label;

            samples(1:i, sample_ind)   = [samples(1:(i - 1), j); fake_label];
            sample_weights(sample_ind) = sample_weights0(j) * probs(i, fake_label);
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
