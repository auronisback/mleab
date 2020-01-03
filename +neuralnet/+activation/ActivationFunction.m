%Author: Francesco Altiero
%Date: 08/12/2018

classdef ActivationFunction < handle
  %ACTIVATIONFUNCTION Abstract class managing activation function
  %   Manages activation function objects, in order to apply them to a
  %   neural network layer. Each concrete implementor has to supply methods
  %   for evaluate the activation function and its derivative.
  
  properties
    layer %A link to the layer this activation function belongs to
    name %The unique name of the concrete activation function
  end
  
  methods
    function this = ActivationFunction(name, layer)
      %ACTIVATIONFUNCTION Constructor for subclasses
      %   Constructor that should be called by any concrete class, in order
      %   to link the layer to the object.
      %
      %   Inputs:
      %     - name: the name of the activation function
      %     - layer: the NetLayer object that use the activation function
      
      assert(isa(layer, 'neuralnet.layer.NetLayer'), ...
        'ActivationFunction:invalidLayer', 'Given layer is invalid');
      this.layer = layer;
      %Setting name
      this.name = name; 
    end
  end
  
  methods (Abstract)
    Z = eval(this, A);
    %eval Evaluates the activation function on input A or on the layer's
    %activations.
    %   This method must evaluate the activation function for the layer and
    %   returns its value.
    %
    % Inputs:
    %   - A: the activation value of neuron in the layer. If not given, the
    %       layer activations will be taken
    % Outputs:
    %   - Z: output fired for each neuron in the net
    
    df = derivative(this, A)
    %derivative Derive the activation function w.r.t. activations.
    %   This method should return the derivative of the activation function
    %   for the given activation values.
    % 
    %   Inputs:
    %     - A: the activation value for nodes in the layer. If not given,
    %       the function should use layer-cached activation values
    %   Outputs:
    %     - df: the derivative value of this activation function
    %     - df
  end
end

