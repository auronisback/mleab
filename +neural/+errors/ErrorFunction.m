classdef ErrorFunction < handle
  %ERRORFUNCTION Abstract class managin error functions
  %   Define base methods in order to define error functions used to train
  %   the network.
  
  methods(Abstract)
    
    e = evaluate(this, Y, T);
      %evaluate Evaluates the error function
      %   Evaluates the error function on given outputs and labels.
      % Inputs:
      %   - Y: outputs of the network
      %   - T: ground truth labels of samples
      % Outputs:
      %   - e: a measure of error between Y and T
    
    dE = derive(this, Y, T);
      %derive Derives the error function w.r.t. outputs
      %   Evaluates the derivative of the error function given the
      %   network's output and labels of samples which were fed to the
      %   network.
      % Inputs:
      %   - Y: outputs of the network
      %   - T: fround truth labels of samples
      % Outputs:
      %   - dE: derivatives of the error function w.r.t. network's outputs
  end
end
