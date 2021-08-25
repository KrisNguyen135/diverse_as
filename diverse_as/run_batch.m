num_exps = 20;
for i = 1:num_exps
    clearvars -except i;
    exp = i;
    fprintf('running experiment %d...\n', i);
    run;
end
