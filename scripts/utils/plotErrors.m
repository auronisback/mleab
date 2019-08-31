%Author: Francesco Altiero
%Date: 26/12/2018

function plotErrors(errors)
  %plotErrors Plots the error structure given after net's training
  %   Manages the plotting of errors for training and validation sets given
  %   after the net has been trained.
  %
  %   Inputs:
  %     - errors: the error structure, with fields:
  %         + training: per-epoch error on training set
  %         + validation: per-epoch error on validation set
  
  figure('NumberTitle', 'off', 'Name', 'Training Error');
  % Plotting training set error
  plot(errors.training);
  hold on;
  % Plotting validation set error
  plot(errors.validation);
  % Plotting best epoch
  plot(errors.bestEpoch, errors.validation(errors.bestEpoch), '*r');
  legend({'Training Set', 'Validation Set', 'Mininum Error'});
  hold off;
end

