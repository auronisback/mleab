classdef Sgd < ann.optimizers.Optimizer
  %SGD Stochastic Gradient Descent optimizer
  %   Defines an optimizer which uses the Stochastic Gradient Descent (SGD)
  %   method in order to evaluate delta of network parameters. Deltas are
  %   obtained using the formula:
  %   deltaW = - eta * dW
  %   deltaB = - eta * dB
  %   with eta hyper-parameter which models the learning rate.
  
  properties(SetAccess = private)
    eta;  % Learning rate
  end
  
  methods
    function this = Sgd(eta)
      %SGD Creates a new instance of SGD optimizer
      %   Creates a new Stochastic Gradient Descent optimizer specifying
      %   the eta parameter.
      % Inputs:
      %   - eta: learning rate of SGD
      this.eta = eta;
    end
    
    function [deltaW, deltaB] = evaluateDeltas(this, dW, db, N)
      %evaluateDeltas Performs calculation of SGD for each layer
      %   Evaluates delta for all layers' weights and biases using the
      %   Stochastic Gradient Descent algorithm.
      % Inputs:
      %   - dW: cell-array with weights derivatives for each layer
      %   - db: cell-array with biases derivatives for each layer
      %   - N: size of the training batch
      % Outputs:
      %   - deltaW: cell-array with weights delta for each layer
      %   - deltaB: cell-array with biases delta for each layer
      
      % Pre-allocating output
      deltaW = cell(1, size(dW, 2));
      deltaB = cell(1, size(db, 2));
      % Calculating deltas for each network's layer
      for l = 1:size(dW, 2)
        deltaW{l} = - this.eta .* dW{l} ./ N;
        deltaB{l} = - this.eta .* db{l} ./ N;
      end
    end
    
    function clear(~)
      %clear Does nothing
      %   Cleares up the object. Nothing to do with SGD.
    end
  end
end

