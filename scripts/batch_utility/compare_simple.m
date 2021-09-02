rng 'default';

% assume independence
num_classes = 3;
num_exps    = 100;
counts      = [0 1 0];
probs       = [[0.1 0.2 0.7];
               [0.6 0.2 0.2];
               [0.8 0.1 0.1];
               [0.5 0.3 0.2]];

assert(numel(counts)  == num_classes, 'inconsistent number of classes');
assert(size(probs, 2) == num_classes, 'inconsistent number of classes');

indep_u  = independent_lite(probs, counts, num_classes)
jensen_u = jensen_lite(probs, counts)

num_samples = 81;
indepmc_u   = indep_mc_lite(probs, counts, num_classes, num_samples)

function utility = independent_lite(probs, counts, num_classes)
    batch_size     = size(probs, 1);
    num_samples    = num_classes ^ batch_size;
    samples        = nan(batch_size, num_samples);
    sample_weights = ones(1, num_samples);

    for i = 1:batch_size
        tmp_num_samples = num_classes ^ (i - 1);
        sample_weights0 = sample_weights;

        for j = tmp_num_samples:-1:1
            for fake_label = 1:num_classes
                sample_ind = num_classes * j - num_classes + fake_label;

                samples(1:i, sample_ind)   = [samples(1:(i - 1), j); fake_label];
                sample_weights(sample_ind) = sample_weights0(j) * probs(i, fake_label);
            end
        end

        sample_weights = sample_weights / ...
            sum(sample_weights(1:(tmp_num_samples * num_classes)));
    end

    onehot = samples(:) == 1:num_classes;
    onehot = reshape(onehot, batch_size, num_samples, num_classes);

    sample_counts = reshape(sum(onehot, 1), num_samples, num_classes);

    utility = 0;
    for i = 1:num_samples
        tmp_counts = counts  + sample_counts(i, :);
        utility    = utility + sample_weights(i) * sum(log(tmp_counts(2:end) + 1));
    end
end

function utility = jensen_lite(probs, counts)
    utility = sum(log(counts(2:end) + 1 + sum(probs(:, 2:end))));
end

function utility = indep_mc_lite(probs, counts, num_classes, num_samples)
    batch_size = size(probs, 1);
    if num_samples >= num_classes ^ batch_size
        utility = independent_lite(probs, counts, num_classes);
        return;
    end
    onehot_samples = nan(batch_size, num_classes, num_samples);

    for i = 1:num_samples
        onehot_samples(:, :, i) = mnrnd(1, probs);
    end

    sample_weights = ones(1, num_samples) / num_samples;
    sample_counts  = reshape(sum(onehot_samples, 1), num_classes, num_samples);

    utility = 0;
    for i = 1:num_samples
        tmp_counts = counts  + sample_counts(:, i)';
        utility    = utility + sample_weights(i) * sum(log(tmp_counts(2:end) + 1));
    end
end
