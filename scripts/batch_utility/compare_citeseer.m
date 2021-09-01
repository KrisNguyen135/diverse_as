train_size = 50;
batch_size = 50;
num_exps   = 100;

addpath(genpath('../../'));
addpath(genpath('../../active_learning'));
addpath(genpath('../../active_search'));

[problem, labels, weights, alpha, nns, sims] = load_data('citeseer', '../../data/');
train_size = train_size * ones(1, problem.num_classes);

problem.verbose     = true;
problem.num_initial = sum(train_size);
problem.counts      = train_size;

model        = get_model(@knn_model_new, weights, alpha);
model_update = get_model_update(@knn_model_update, weights);
selector     = get_selector(@unlabeled_selector);

for exp = 1:num_exps
    rng(exp);

    train_ind    = [];
    train_labels = [];
    for label = 1:problem.num_classes
        tmp_train_ind = randsample(find(labels == label), train_size(label));
        train_ind     = [train_ind; tmp_train_ind];
        train_labels  = [train_labels; labels(tmp_train_ind)];
    end

    test_ind      = selector(problem, train_ind, []);
    [probs, ~, ~] = model(problem, train_ind, train_labels, test_ind);

    batch_ind
end
