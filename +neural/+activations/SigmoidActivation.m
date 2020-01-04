classdef SigmoidActivation < neural.activations.ActivationFunction
  %SIGMOIDACTIVATION Class managing the sigmoid activation function
  %   Manages data and operation in order to use a sigmoid activation
  %   function of network's layers.
  
  properties(Constant, Access = private)
    NAME_SIGMOID = 'sigmoid';  % Name for the activation function
  end
  
  methods
    function this = SigmoidActivation(layer)
      %SIGMOIDACTIVATION Construct an instance of sigmoid activation
      %   Creates a new sigmoid activation function object, optionally
      %   specifying the layer to which it is linked.
      % Input:
      %   - layer: the layer which uses this activation function
      this = this@neural.activations.ActivationFunction(...
        neural.activations.SigmoidActivation.NAME_SIGMOID);
      if nargin > 0
        this.setLayer(layer);
      end
    end
    
    function Z = evaluate(~, X)
      %evaluate Evaluates the concrete activation function
      %   Performs operations in order to evaluate the function.
      % Inputs:
      %   - X: the input data on which the activation function will be
      %     evaluated
      % Output:
      %   - Z: the result of application of the activation function to the
      %     inputs
      Z = 1 ./ (1 + exp(X));
    end
    
    function dX = derive(this, X)
      %derive Derives the sigmoid on given input
      %   Derives the sigmoid activation function on given values.
      % Inputs:
      %   - X: the values of the sigmoid function
      % Outputs:
      %   - dX: the value of derivatives
      if nargin < 2
        X = this.layer.Z;  % Taking layer's output
      end
      dX = X .* (1 - X);
    end
  end
end

