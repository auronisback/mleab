%Author: Francesco Altiero
%Date: 26/12/2018

function plotErrors(errors, bestEpoch, title)
  %plotErrors Plots the error structure given after net's training
  %   Manages the plotting of errors for training and validation sets given
  %   after the net has been trained.
  %
  % Inputs:
  %   - errors: a cell-array obtained by training, with evaluation of 
  %     loss on training and validation set and accuracy on both these sets
  %   - bestEpoch: the epoch in which the training reached best accuracy
  %     value on validation set
  %   - title: an optional string which will be a title
  if nargin < 3
    title = 'Error Plot';
  end
  figure('Name', title);
  subplot(1, 2, 1);
  % Plotting training set error
  nEpochs = size(errors{1}, 2) - 1;
  plot(0:nEpochs, errors{1});
  hold on;
  % Plotting validation set error
  plot(0:nEpochs, errors{2});
  grid on;
  legend({'Training Loss', 'Validation Loss'});
  % Plotting training set accuracy
  subplot(1, 2, 2);
  plot(0:nEpochs, errors{3});
  hold on;
  % Plotting validation set accuracy
  plot(0:nEpochs, errors{4});
  if bestEpoch > 0
    plot(bestEpoch - 1, errors{4}(bestEpoch), '*r');
  end
  grid on;
  legend({'Training Accuracy', 'Validation Accuracy'});
  hold off;
end

