a = 26;
b = 30;
for i = a:b
    clearvars -except i;
    exp = i;
    % verbose = false;
    fprintf('running experiment %d...\n', i);

    data       = 'citeseer';
    policy     = 'greedy';
    group_size = 4;

    run;
    % run_square;
end

% num_classes = 120;
% for i = 1:num_classes
%     clearvars -except i;
%
%     data       = 'single'
%     data       = sprintf('%s%d', data, i);
%     policy     = 'greedy';
%     group_size = 1;
%     exp        = 1;
%
%     fprintf('running experiment %d...\n', i);
%     run;
% end
