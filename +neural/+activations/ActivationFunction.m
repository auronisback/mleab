classdef ActivationFunction < handle
  %ACTIVATIONFUNCTION Baseline for generic activation functions
  %   Manages methods used in order to evaluate and derive an activation
  %   function used in layers of a neural network.
  
  properties(Constant, Access = private)
    NAME_UNKNOWN = 'unknown';  % Constant for unknown activation functions
  end
  
  properties(SetAccess = private, GetAccess = public)
    name;  % Name of activation function
    layer;  % The layer this activation function belongs to
  end
  
  methods
    function this = ActivationFunction(name)
      %ACTIVATIONFUNCTION Construct an instance of activation function
      %   Baseline constructor for activation functions, created specifying
      %   their name.
      % Inputs:
      %   - name: the name of activation function
      if nargin > 0
        this.name = name;
      else
        this.name = this.NAME_UNKNOWN;
      end
    end
    
    function setLayer(this, layer)
      %setLayer Sets the layer to which this activation function is linked
      %   Sets the layer which uses the activation function.
      % Inputs:
      %   - layer: the layer object
      if isa(layer, 'neural.layers.Layer')
        this.layer = layer;
      else
        error('ActivationFunction:InvalidLayer', ...
          'Passed layer argument should be a Layer instance, given %s', ...
          class(layer));
      end
    end
  end
  
  methods(Abstract)
    Z = evaluate(this, X);
      %evaluate Evaluates the concrete activation function
      %   Performs operations in order to evaluate the function.
      % Inputs:
      %   - X: the input data on which the activation function will be
      %     evaluated
      % Output:
      %   - Z: the result of application of the activation function to the
      %     inputs
    
    dX = derive(this, X);
      %derive Derives the function
      %   Derives the activation function on given values.
      % Inputs:
      %   - X: the values in which the function has to be derived. If not
      %     given, they should be taken from the layer accordingly to
      %     concrete function
      % Outputs:
      %   - dX: the value of derivatives
  end
  
end

