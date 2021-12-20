for class_ind = 1:1
    fprintf('active class number %i\n', class_ind);

    data       = sprintf('morgan%i', class_ind);
    exp        = 1;
    group_size = 3;

    addpath(genpath('../../'));
    addpath(genpath('../../active_learning'));
    addpath(genpath('../../active_search'));

    data_dir   = '../../data/';

    [problem, labels, weights, alpha, nns, sims] = load_data(data, data_dir, exp, group_size);
    nns = nns';
    sims = sims';

    fprintf('# negatives: %i\n', sum(labels == 1));
    for i = 2:problem.num_classes
        pos_ind = find(labels == i);

        fprintf('# class-%i positives: %i', i, numel(pos_ind));
        fprintf('\tfrom %i to %i\n', pos_ind(1), pos_ind(end));

        fprintf('nearest neighbors of %i:', pos_ind(1));
        disp(nns(pos_ind(1), :));
        disp(sims(pos_ind(1), :))
    end
end
