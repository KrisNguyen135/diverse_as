a = 1;
b = 20;
for i = a:b
    clearvars -except i;
    exp = i;
    % verbose = false;
    fprintf('running experiment %d...\n', i);

    % data       = 'citeseer';
    policy     = 'round robin ucb';
    group_size = 5;

    % run;
    run_square;
end

% a = 1;
% b = 30;
% for i = a:b
%     clearvars -except i;
%
%     data       = 'morgan'
%     data       = sprintf('%s%d', data, i);
%     policy     = 'greedy';
%     group_size = 9;
%     exp        = 1;
%
%     fprintf('running experiment %d...\n', i);
%     run;
% end
