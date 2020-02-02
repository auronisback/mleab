classdef Optimizer < handle
  %OPTIMIZER Baseline class for defining optimizer
  %   Baseline class used to define a network optimizer, which manages
  %   operations to calculate deltas used to tune weights and biases in a
  %   neural network.
  
  methods(Abstract)
    [deltaW, deltaB] = evaluateDeltas(this, dW, db, N);
      %evaluateDeltas Performs calculation of delta values for each layer
      %in the network
      %   Manages operation used to obtain delta values by using
      %   derivatives of weights and biases in a neural network, optionally
      %   using the size of the input fed to the network.
      % Inputs:
      %   - dW: cell-array with weights derivatives for each layer
      %   - db: cell-array with biases derivatives for each layer
      %   - N: size of the training batch
      % Outputs:
      %   - deltaW: cell-array with weights delta for each layer
      %   - deltaB: cell-array with biases delta for each layer
      
    clear(this);
      %clear Performs operations in order to reset the object after
      %training
      %   Abstract method used after training in order to reset default
      %   values of concrete object and to make them perform cleanup
      %   operations.
      
    s = toString(this);
      %toString Gets the string representation of the optimizer
      %   Converts the optimizer object into a human-readable format.
      % Output:
      %   - s: string representation of the optimizer
      
  end
end

