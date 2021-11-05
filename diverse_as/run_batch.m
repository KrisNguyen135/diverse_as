num_exps = 20;
for i = 1:num_exps
    clearvars -except i;
    exp = i;
    % verbose = false;
    fprintf('running experiment %d...\n', i);
    run;
    % run_square;
end
