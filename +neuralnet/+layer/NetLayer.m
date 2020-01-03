%Author: Francesco Altiero
%Date: 08/12/2018

classdef NetLayer < handle
  %NETLAYER Defines abstraction for a neural network's layer
  %   Abstract class that defines common properties and operation for a
  %   neural network layer. Each NeuralNet layer object must be of a
  %   concrete class extending this.
  
  properties
    inputDim %Array with dimensions of the layer's input
    activations %Cached activation values for each neuron in the layer
    outputs %Cached outputs for each neuron in the layer
  end
  
  methods
    function this = NetLayer(dimensions)
      %NETLAYER Creates a net layer specifying the dimension of its input
      %   Parent constructor for a net layer, in which the dimensions of
      %   the input to the layer has to be specified.
      %
      %   Inputs:
      %     - dimensions: an array whose length is the number of dimension
      %         of the layer's input and each value is the size of the
      %         related dimension
      
      %Checking dimensions
      assert(isvector(dimensions), 'NetLayer:dimensionNotArray', ...
          'Gived dimensions is not an array');
      assert(all(dimensions > 0), 'NetLayer:invalidDimension', ...
          'Given dimensions has one or more not positive values');
      %Ok, setting dimensions
      this.inputDim = dimensions;
    end
  end
  
  methods (Abstract)
    Z = forwardTraining(this, X);
      %propagate Propagate forward the input when training the net.
      %   In concrete subclass, this method has to calculate output Z from 
      %   the given input X. It should cache data such activations and
      %   outputs in order to be used in training of the net.
      %
      %   Inputs:
      %     - X: the data that have to be processed by layer
      %   Outputs:
      %     - Z: the value calculated by the layer
      
    Z = forward(this, X);
      %predict Predict the input propagating forward.
      %   In concrete subclass, this method must calculate output Z from the
      %   given input X. When not training the net, this method should be
      %   used and in its implementation it should not cache intermediate
      %   data.
      %
      %   Inputs:
      %     - X: the data that have to be processed by layer
      %   Outputs:
      %     - Z: the value calculated by the layer
    
    udpdateWeightsAndBiases(this, deltaW, deltaB);
      %updateWeightsAndBiases Performs the updating of the layer's parameters
      %   Updates weights and biases of this layer given deltas calculated
      %   using any weight update methods.
      %
      %   Inputs:
      %     - deltaW: the differential that has to be added to weights
      %     - deltaB: the differential that has to be added to biases
    
  end
end

