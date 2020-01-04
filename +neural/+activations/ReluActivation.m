classdef ReluActivation < neural.activations.ActivationFunction
  %RELUACTIVATION Class managing a ReLU activation function
  %   Rectified Linear Unit activation function.
  
  properties(Constant, Access = private)
    NAME_RELU = 'ReLU';  % Name constant
  end
  
  methods
    function this = ReluActivation(layer)
      %RELUACTIVATION Construct an instance of ReLU activation function
      %   Creates the object, optionally specifying the layer to which
      %   this activation function belongs.
      % Inputs:
      %   - layer: the layer to which this activation function is linked.
      %       If not given, none layer will be set
      this = this@neural.activations.ActivationFunction(...
          neural.activations.ReluActivation.NAME_RELU);
      if nargin > 0
        this.setLayer(layer);
      end
    end
    
    function Z = evaluate(~, X)
      %evaluate Evaluates the ReLU activation function on input
      %   Evaluates the ReLU activation function on the given input,
      %   performing an element-wise max(0, X) operation.
      % Inputs:
      %   - X: the input
      % Outputs:
      %   - Z: an output with the same shape of input where the negative
      %     values are suppressed
      Z = max(0, X);
    end
    
    function dX = derive(this, X)
      %derive Derives the ReLU activation function on given input
      %   Evaluates derivatives of this activation function on the given
      %   input.
      % Inputs:
      %   - X: the values in which the ReLU function has to be derived
      % Output:
      %   - dX: derivatives of the activation function
      if nargin < 2
        X = this.layer.A;  % Not given, taking from layer
      end
      dX = zeros(size(X));
      dX(X > 0) = 1;  % Derivative is one if positive
    end
  end
end

