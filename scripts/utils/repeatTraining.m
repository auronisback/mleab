function repeatTraining(net, ds, training, epochs, repetitions, filename)
%REPEATTRAINING Repeats a network's training and saves its statistics to a
%file
%   Executes some repetitions of a training of a neural network in order to
%   produce mean and median values of loss and accuracy on the test set.
%   Outputs will be saved to a specified file.
% Inputs:
%   - net: the network which will be trained
%   - ds: dataset used to train the network
%   - epochs: number of training epochs
%   - repetitions: number of repetitions
%   - filename: name of the file in which outputs will be saved
  
  % Initializing errors, accuracies and elapsed time
  testErrs = zeros(1, repetitions);
  testAccs = zeros(1, repetitions);
  elapsed = zeros(1, repetitions);
  % Repeating training
  for n = 1:repetitions
    net.reinitialize();  % Re-initializes the network
    [~, ~, elTime] = training.train(epochs, net, ds, false);
    elapsed(n) = sum(elTime);  % Getting total elapsed time
    [testErrs(n), testAccs(n)] = training.evaluateOnTestSet(net, ds);
    fprintf('.');
  end
  fprintf('\n');
  % Writing results on a xls file
  fprintf('Writing to %s...\n', filename);
  writecell({'#', 'loss', 'acc', 'time'}, filename, 'Sheet', 1, ...
    'Range', 'A1:D1');
  writematrix([(1:repetitions).', testErrs.', testAccs.', elapsed.'], ...
    filename, 'Sheet', 1, 'Range', sprintf('A2:D%d', repetitions + 1));
  writecell({'mean', 'median'}, filename, 'Sheet', 1, 'Range', ...
    sprintf('B%d:C%d', repetitions + 3, repetitions + 3), 'UseExcel', false);
  writecell({'loss'; 'accuracy'; 'elapsed'}, filename, 'Sheet', 1, ...
    'Range', sprintf('A%d:A%d', repetitions + 4, repetitions + 6), 'UseExcel', false);
  writematrix([mean(testErrs), median(testErrs); ...
    mean(testAccs), median(testAccs); mean(elapsed), median(elapsed)], ...
    filename, 'Sheet', 1, 'Range', ...
    sprintf('B%d:C%d', repetitions + 4, repetitions + 6), 'UseExcel', false);
end

