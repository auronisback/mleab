%Author: Francesco Altiero
%Date: 22/12/2018

classdef CrossEntropy < neuralnet.train.error.ErrorFunction
  %CROSSENTROPY Cross-entropy error function
  %   Manages the calculation of the cross-entropy error between net's
  %   outputs and target values. The error function is calculated as:
  %     E = - sum_n sum_k t_k ln (y_k / t_k)
  
  methods
    function error = eval(~, Y, T)
      %eval Evaluate the cross entropy error on output and labels
      %   This method evaluates the error between outputs and target values
      %   using the cross-entropy error function.
      %
      %   Inputs:
      %     - Y: the output of the net
      %     - T: labels, or target values, of patterns given as net's input
      %   Outputs:
      %     - error: the cross-entropy error function evaluated between Y
      %         and T, as a scalar value
      
      error = - sum(sum(T .* reallog(Y)));
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

