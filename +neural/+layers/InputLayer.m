classdef InputLayer < neural.layers.Layer
  %INPUTLAYER Layer used for network input
  %   Fake layer used in order to forward propagate input and to uniform
  %   backpropagation between all subsequent layers. This is always the
  %   first layer in the neural network.
  
  properties(Constant)
    NAME_INPUT = 'input';
  end
  
  methods
    function this = InputLayer(inputSize, name)
      %INPUTLAYER Construct an instance of input layer class
      %   Creates an input layer object specifying the size of input and
      %   optionally its name.
      % Inputs:
      %   - inputSize: an array with input dimensions
      %   - name: the name of the layer, optional
      this = this@neural.layers.Layer(inputSize, '');
      this.outputSize = inputSize;  % Output size is equal to input's
      if nargin > 1
        this.name = name;
      else
        this.name = neural.layers.InputLayer.NAME_INPUT;
      end
    end
    
    function setInputSize(this, inputSize)
      %setInputSize Sets the input size of the layer
      %   Updates the shape of the input layer.
      % Inputs:
      %   - inputSize: the new shape of input
      assert(~isempty(inputSize), 'InputLayer.invalidInputSize', ...
        'Empty input size given');
      this.inputSize = inputSize;
      this.outputSize = inputSize;
    end
    
    function [W, b] = getWeightsAndBiases(~)
      %getWeightsAndBiases Unused
      %   Unused.
      % Outputs:
      %   - W: empty array
      %   - b: empty array
      W = [];
      b = [];
    end
    
    function setWeigthsAndBiases(~, ~, ~)
      %setWeightsAndBiases Unused
      %   Sets the weights and biases values in the layer. Unused.
    end    
    
    function Y = predict(this, X)
      %predict Propagates input forward 
      %   Forwards given input, returning it.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Y: same values of X
      Y = this.next.predict(X);
    end
    
    function Y = forward(this, X)
      %forward Forward propagates the input, with caching
      %   Propagates forward the input, caching its value in order to be
      %   used for training.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Y: same values of X
      this.Z = X;
      Y = this.next.forward(X);
    end
    
    function dX = backward(this, ~)
      %backward Does nothing, as inputs are not backpropagated
      %   Does nothing.
      % Inputs:
      %   - dZ: derivatives of the next layer
      % Outputs:
      %   - dX: the cached input itself
      dX = this.Z;
    end
    
    function updateWeightsAndBiases(~, ~, ~)
      %updateWeightsAndBiases Does nothing
      % Does nothing as input layer has no weights.
    end
  end
end

