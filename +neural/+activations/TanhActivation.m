classdef TanhActivation < neural.activations.ActivationFunction
  %TANHACTIVATION Hyperbolic tangent activation function
  %   Implementation of the hyperbonic tangent as activation function used
  %   in network layers.
  
  properties(Constant, Access = private)
    NAME_TANH = 'tanh';  % Name of activation function
  end
  
  methods
    function this = TanhActivation(layer)
      %TANHACTIVATION Construct an instance of tanh activation function
      %   Creates a new hyperbolic tangent activation function optionally
      %   specifying the layer in which it is used.
      % Inputs:
      %   - layer: the optional layer in which the function is used
      this = this@neural.activations.ActivationFunction(...
        neural.activations.TanhActivation.NAME_TANH);
      if nargin > 0
        this.setLayer(layer);
      end
    end
    
    function Z = evaluate(~, X)
      %evaluate Evaluates the tanh activation function
      %   Performs an evaluation of the tanh activation function on the
      %   given input.
      % Inputs:
      %   - X: values on which evaluate the tanh function
      % Outputs:
      %   - Z: calculated values
      Z = tanh(X);
    end
    
    function dX = derive(~, X)
      %derive Derives the tanh function
      %   Derives tanh activation function on calculated values.
      % Inputs:
      %   - X: tanh calculated values
      % Outputs:
      %   - dX: the value of derivatives
      dX = 1 - (X * X);
    end
  end
end

