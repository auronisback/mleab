%Author: Francesco Altiero
%Date: 10/12/2018

classdef StopCriterion
  %STOPCRITERION Interface used to implement early stop criteria
  %   Interface that each stop criterion has to implement in order to be
  %   used in neural network training.
  
  methods (Abstract)
    stop = isMet(this, train, net, errors);
      %isMet Boolean method that checks if the criterion is met
      %   Checks if the early stop criterion that implments the method is
      %   met when training.
      %
      %   Inputs:
      %     - train: the AbstractTraining object representing the training
      %     - net: the NeuralNet object which is trained
      %     - errors: the error structure with errors on training and
      %         validation sets
      %   Outputs:
      %     - stop: logical value that is true if the criterion is met, or
      %         false otherwise
  end
end

