%Author: Francesco Altiero
%Date: 11/12/2018

classdef SumOfSquare < neuralnet.train.error.ErrorFunction
  %SUMOFSQUARE Sum of squared error function
  %   Manages the calculation of the sum of squared error between net's
  %   outputs and target values. The error function is calculated as:
  %     E = 1 / 2 * sum(|Y-T|^2)
  
  methods
    function error = eval(~, Y, T)
      %eval Evaluate the sum of squared error on output and labels
      %   This method evaluates the error between outputs and target values
      %   using the Sum of Squared error function.
      %
      %   Inputs:
      %     - Y: the output of the net
      %     - T: labels, or target values, of patterns given as net's input
      %   Outputs:
      %     - error: the sum of squared error function evaluated between Y
      %         and T, as a scalar value
      
      D = (Y - T).^2; %Squared distance
      error = sum(D(:));
    end
      
    function derror = derivative(~, Y, T)
      %derivative Evaluates the sum os squared error function derivatives
      %   Calculates the derivative of the sum of squared error function
      %   with respect to the net's outputs. The derivative is simply
      %     dE = Y - T
      %
      %   Inputs:
      %     - Y: the net's outputs
      %     - T: labels, or target values, of the training set
      %   Outputs:
      %     - derror: the error derivative as a matrix whose rows and 
      %         columns are related to patterns in the training set and 
      %         output nodes, respectively. It stores the error function
      %         derivatives with respect to output nodes and patterns
      
      derror = Y - T;
    end
  end
end

