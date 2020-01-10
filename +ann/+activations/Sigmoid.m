classdef Sigmoid < ann.activations.ActivationFunction
  %SIGMOID Sigmoid activation function
  %   Defines methods in order to use a sigmoid activation function.
  
  methods
    function Z = eval(~, A)
      %eval Evaluates the sigmoid activation function
      %   Calculates the sigmoid activation function on layer's activation
      %   values, using:
      %           1
      %   Z = ----------
      %       1 + e^(-a)
      % Inputs:
      %  - A: layer's activation values
      % Output:
      %   - Z: sigmoid evaluated in A
      Z = 1 ./ (1 + exp(-A));
    end
    
    function dA = derive(this, dZ)
      %derive Derives the sigmoid function
      %   Derives the sigmoid function, using layer's cached outputs.
      %   Derivatives are efficiently evaluated using the formula:
      %   ds   
      %   -- = s(A)(1 - s(A))
      %   dA
      %   and then multiplied element-wise for dZ.
      % Inputs:
      %   - dZ: derivatives of layer's output
      % Output:
      %   - dA: derivative of sigmoid function, obtained using values
      %     cached in the linked layer
      dA = dZ .* this.layer.Z .* (1 - this.layer.Z);
    end
  end
end

