classdef Identity < ann.activations.ActivationFunction
  %IDENTITY Identity activation function
  %   Identity activation function, which output same values given in
  %   input.
  
  methods
    function Z = eval(~, A)
      %eval Evaluates the identity activation function
      %   Evaluates the identity activation function simply returning the
      %   same input.
      % Inputs:
      %   - A: activation values of the layer
      % Output:
      %   - Z: same values as A
      Z = A;
    end
      
    function dA = derive(~, dZ)
      %derive Derives the identity activation function
      %   Derives the identity using derivatives of outputs. Returned
      %   values is the just output's derivatives.
      % Inputs:
      %   - dZ: derivatives with respect to layer's output
      % Output:
      %   - dA: values of dZ
      dA = dZ;
    end
  end
end

