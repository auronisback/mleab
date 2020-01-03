%Author: Francesco Altiero
%Date: 10/12/2018

classdef OutputLayer < neuralnet.layer.NetLayer
  %OUTPUTLAYER Defines net's output layers common methods
  %   Base class for output layers, that are network layers that produces
  %   outputs. Due the fact that during learning with backpropagation those
  %   kind of layer are treated in a different way, this class defines the
  %   method used to evaluate delta for outputs nodes.
  
  properties
    outputDim %Dimension of the output the layer produces
  end
  
  methods
    function this = OutputLayer(dimensions, outputDim)
      %OutputLayer Creates an output layer specifying input's dimension
      %   Parent constructor for a new output layer, in which the
      %   dimensions of input to the layer has to be specified.
      %
      %   Inputs:
      %     - dimensions: an array whose length is the number of dimension
      %         of the layer's input and each value is the size of the
      %         related dimension
      %     - outputDim: dimension of the output of the concrete layer
      
      %Checking output dimensionality
      assert(outputDim > 0, 'OutputLayer:invalidOutDimension', ...
        sprintf('Output dimension invalid: %d', outputDim));
      %Calling parent constructor
      this@neuralnet.layer.NetLayer(dimensions);
      %Setting output dimensionality
      this.outputDim = outputDim;
    end
  end
  
  methods (Abstract)
    dout = calculateDeltaOut(this, errFun, T);
      %calculateDelta Calculates delta for output layers
      %   This method should perform the calculation of backpropagation deltas
      %   deltas for the output layer, using the error function and
      %   training set labels. It is mandatory that outputs of the net are 
      %   cached inside the layer (using forward propagation method).
      %
      %   Inputs:
      %     - errFun: error function used to calculate deltas
      %     - T: dataset labels, used to evaluate the error derivative
      %   Outputs:
      %     - dout: delta values for each node in the output layer
  end
end

