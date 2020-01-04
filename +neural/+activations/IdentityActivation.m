classdef IdentityActivation < neural.activations.ActivationFunction
  %IDENTITYACTIVATION Class managing a identity activation function
  %   Identity activation function.
  
  properties(Constant, Access = private)
    NAME_IDENTITY = 'identity';  % Name constant
  end
  
  methods
    function this = IdentityActivation(layer)
      %IDENTITYACTIVATION Construct an instance of identity function
      %   Creates the object, optionally specifying the layer to which
      %   this activation function belongs.
      % Inputs:
      %   - layer: the layer to which this activation function is linked.
      %       If not given, none layer will be set
      this = this@neural.activations.ActivationFunction(...
          neural.activations.IdentityActivation.NAME_IDENTITY);
      if nargin > 0
        this.setLayer(layer);
      end
    end
    
    function Z = evaluate(~, X)
      %evaluate Evaluates the identity activation function on input
      %   Evaluates the identity activation function on the given input,
      %   propagating the input as output.
      % Inputs:
      %   - X: the input
      % Outputs:
      %   - Z: given input
      Z = X;
    end
    
    function dX = derive(this, X)
      %derive Derives the identity activation function on given input
      %   Evaluates derivatives of this activation function on the given
      %   activation values.
      % Inputs:
      %   - X: the values in which derive
      % Output:
      %   - dX: a ones matrix with same shape of X
      if nargin < 2
        X = this.layer.Z;
      end
      dX = ones(size(X));
    end
  end
end

