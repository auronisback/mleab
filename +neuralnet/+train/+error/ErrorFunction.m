%Author: Francesco Altiero
%Date: 11/12/2018

classdef ErrorFunction < handle
  %ERRORFUNCTION Base class used to create error functions
  %   Defines methods in order to implement error function used in the
  %   training of a NeuralNet.
  
  methods (Abstract)
    error = eval(this, Y, T);
      %eval Evaluate the error function on output and labels
      %   This method evaluates the concrete error function given outputs
      %   of the net and target values.
      %
      %   Inputs:
      %     - Y: the output of the net
      %     - T: labels, or target values, of patterns given as net's input
      %   Outputs:
      %     - error: the error function evaluated between Y and T, as a
      %         scalar value 
      
    derror = derivative(this, Y, T);
      %derivative Evaluates the error function derivatives w.r.t. outputs
      %   Calculates the derivative of the concrete error function with
      %   respect to the net's outputs; used to perform backpropagation.
      %
      %   Inputs:
      %     - Y: the net's outputs
      %     - T: labels, or target values, of the training set
      %   Outputs:
      %     - derror: the error derivative as a matrix whose rows and 
      %         columns are related to patterns in the training set and 
      %         output nodes, respectively. It stores the error function
      %         derivatives with respect to output nodes and patterns
  end
end

