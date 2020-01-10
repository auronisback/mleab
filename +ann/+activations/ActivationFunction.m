classdef ActivationFunction < handle
  %ACTIVATIONFUNCTION Baseline class to define activation function
  %   Defines properties and operations used to declare a network layer's
  %   activation function.
  
  properties(SetAccess = private)
    layer;  % Layer to which the activation function belongs to
  end
  
  methods
    function setLayer(this, layer)
      %setLayer Links the activation function to a layer
      %   Sets the layer which is linked to this activation function.
      % Inputs:
      %   - layer: the layer instance which has to be linked
      assert(isa(layer, 'ann.layers.Layer'), ...
        'ActivationFunction:invalidLayer', 'Invalid layer type: %s', ...
        class(layer));
      this.layer = layer;
    end
  end
  
  methods(Abstract)
    Z = eval(this, A);
      %eval Evaluates the activation function
      %   Evaluates the activation function on given layer's activation
      %   values.
      % Inputs:
      %   - A: layer's activation values
      % Outputs:
      %   - Z: layer's output
    
    dA = derive(this, dZ);
      %derive Derives the activation function using layer's cached values
      %   Derive the output of this activation function using values cached
      %   in a layer and derivatives of layer's output. It is used only for
      %   training purposes.
      % Inputs:
      %   - dZ: derivatives of layer's output, as in chain rules
      % Output:
      %   - dA: derivatives of this activation function obtained using
      %     values cached in the linked layer
  end
end

