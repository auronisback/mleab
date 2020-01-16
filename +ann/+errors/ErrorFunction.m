classdef ErrorFunction
  %ERRORFUNCTION Baseline class for error function
  %   Defines baseline operations for a loss function used when training a
  %   neural network.
  
  methods(Abstract)
    E = eval(this, Y, T);
      %eval Evaluates the error function
      %   Evaluates the error function between net's output and target
      %   values.
      % Inputs:
      %   - Y: output of network
      %   - T: ground truth values
      % Output:
      %   - E: evaluation of concrete error function between Y and T
      
    dE = derive(this, Y, T);
      %derive Calculate derivatives of error function w.r.t. outputs
      %   Derives the error function between network's output and target
      %   values, producing a matrix with derivatives of all outputs (on a
      %   row) and all output's parameter (on a column).
      % Inputs:
      %   - Y: network's output values
      %   - T: target values
      % Output:
      %   - dE: derivative of error function with respect to outputs; it is
      %       a matrix which has one row for sample and a column for all
      %       elements of an output.
  end
end

