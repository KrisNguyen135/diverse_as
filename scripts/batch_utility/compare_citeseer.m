train_size  = 50;
batch_size  = 300
num_exps    = 10;
num_samples = [2^10, 2^15];

utility    = 'log';
run_full   = false;

addpath(genpath('../../'));
addpath(genpath('../../active_learning'));
addpath(genpath('../../active_search'));

[problem, labels, weights, alpha, nns, sims] = load_data('citeseer', '../../data/');
train_size = train_size * ones(1, problem.num_classes);

problem

problem.verbose     = true;
problem.num_initial = sum(train_size);
problem.counts      = train_size;

model        = get_model(@knn_model_new, weights, alpha);
model_update = get_model_update(@knn_model_update, weights);
selector     = get_selector(@unlabeled_selector);

switch utility
case 'log'
    utility_function = @log_utility;
case 'threshold'
    threshold        = 2;
    utility_function = get_utility_function(@threshold_utility, threshold);
end
problem.utility = utility;

indep_u   = nan(1, num_exps);
indepmc_u = nan(numel(num_samples), num_exps);
jensen_u  = nan(1, num_exps);

indep_t   = nan(1, num_exps);
indepmc_t = nan(numel(num_samples), num_exps);
jensen_t  = nan(1, num_exps);

for exp = 1:num_exps
    fprintf('running experiment %d...\n', exp);
    rng(exp);

    train_ind    = [];
    train_labels = [];
    for label = 1:problem.num_classes
        tmp_train_ind = randsample(find(labels == label), train_size(label));
        train_ind     = [train_ind; tmp_train_ind];
        train_labels  = [train_labels; labels(tmp_train_ind)];
    end

    test_ind  = selector(problem, train_ind, []);
    batch_ind = randsample((1:numel(test_ind))', batch_size);

    if run_full
        tic;
        indep_u(exp) = independent(problem, train_ind, train_labels, batch_ind, ...
                                     model, utility_function);
        indep_t(exp) = toc;
    end

    for mc_ind = 1:numel(num_samples)
        tic;
        indepmc_u(mc_ind, exp) = indep_mc(problem, train_ind, train_labels, ...
                                          batch_ind, model, utility_function, ...
                                          num_samples(mc_ind));
        indepmc_t(mc_ind, exp) = toc;
    end

    tic;
    jensen_u(exp)  = jensen(problem, train_ind, train_labels, batch_ind, model);
    jensen_t(exp)  = toc;
end

if run_full
    results = [mean(indep_t)];
    errors  = [indep_u; indepmc_u; jensen_u] - indep_u;
else
    results = [];
    errors  = [indepmc_u; jensen_u] - indepmc_u(end, :);
end

rmse    = sqrt(mean(errors.^2, 2));
results = [results; mean(indepmc_t, 2); mean(jensen_t)];
results = [results, rmse]
