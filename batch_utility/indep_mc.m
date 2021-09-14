% function [onehot_saples, sample_weights, sample_counts, utility] = indep_mc( ...
%     problem, train_ind, train_labels, batch_ind, model, utility_function, num_samples)
function utility = indep_mc(problem, train_ind, train_labels, batch_ind, ...
                            model, utility_function, num_samples)

batch_size = numel(batch_ind);
% if num_samples >= problem.num_classes ^ batch_size
%     utility = independent(problem, train_ind, train_labels, batch_ind, ...
%                           model, utility_function);
%     return;
% end
onehot_samples = nan(batch_size, problem.num_classes, num_samples);
[probs, ~, ~]  = model(problem, train_ind, train_labels, batch_ind);

for i = 1:num_samples
    onehot_samples(:, :, i) = mnrnd(1, probs);
end

% sample_weights = onehot_samples .* probs;          % keep only the sampled probabilities, batch x C x samples
% sample_weights = squeeze(sum(sample_weights, 2));  % sum across the classes, batch x samples
% sample_weights = prod(sample_weights, 1);          % multiply across the batch elements, 1 x samples
% sample_weights = sample_weights / sum(sample_weights);
sample_weights = ones(1, num_samples) / num_samples;

sample_counts = reshape(sum(onehot_samples, 1), problem.num_classes, num_samples);  % C x samples

utility = 0;
for i = 1:num_samples
    old_counts     = problem.counts;
    problem.counts = problem.counts + sample_counts(:, i)';
    utility        = utility + sample_weights(i) * utility_function(problem);
    problem.counts = old_counts;
end
