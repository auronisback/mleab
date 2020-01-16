classdef FlattenLayer < ann.layers.Layer
  %FLATTEN Layer used to flatten outputs in hidden layers of an ANN
  %   Layer which is used to linearize output shape of previous layer in
  %   order to be used with fully-connected layer. It should be used after
  %   a convolutional layer befeore feeding outputs to the fully-connected
  %   section of a neural network.
  
  
  methods
    function this = FlattenLayer(inputShape)
      %FLATTENLAYER Creates a new flatten layer given input shape
      %   Creates a layer which will flatten previous layer's output.
      % Inputs:
      %   - inputShape: dimension of layer's input
      this.inputShape = inputShape;
      this.outputShape = prod(inputShape, 'all');
      %Initializing name
      this.name = sprintf('Flatten');
    end
    
    function [W, b] = getParameters(~)
      %getParameters Gets nothing
      %   Does nothing, as this layer has no parameters.
      % Outputs:
      %   - W: empty matrix
      %   - b: empty matrix
      W = [];
      b = [];
    end
    
    function setParameters(~, ~, ~)
      %setParameters Sets nothing
      %   Does nothing, as this layer has no parameters.
    end
    
    function Z = predict(this, X)
      %predict Evaluates the layer on input
      %   Calculates output of the layer, or the input data reshaped in
      %   order to be flattened.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Z: flattened values of X
      Z = reshape(X, [size(X, 1), this.outputShape]);
    end
    
    function Z = forward(this, X)
      %forward Flattens data and caches values
      %   Performs a forward pass in the layer, caching output data.
      % Inputs:
      %   - X: layer's input
      % Outputs:
      %   - Z: flattened input
      this.Z = reshape(X, [size(X, 1), this.outputShape]);
      Z = this.Z;
    end
    
    function [dX, dW, db] = backward(this, dZ, ~)
      %backward Performs a backward pass in the layer
      %   Unflattens derivative of next layer in order to be used with
      %   previous layer.
      % Inputs:
      %   - dZ: derivatives of error w.r.t. layer's output
      % Outputs:
      %   - dX: derivatives w.r.t. layer's input
      %   - dW: derivatives w.r.t. layer's weights (empty)
      %   - db: derivatives w.r.t. layer's biases (empty)
      dW = [];
      db = [];
      dX = reshape(dZ, [size(dZ, 1), this.inputShape]);
    end
    
    function [dX, dW, db] = outputBackward(this, errorFun, ~, T)
      %outputBackward Backpropagates errors if this is an output layer
      %   Manages backpropagation if the FC-layer is the last layer in the
      %   network, using error function, target values and layer's input.
      % Inputs:
      %   - errorFun: error function used to derive error with respect to
      %     network's output
      %   - X: layer's input
      %   - T: target values for the input
      % Outputs:
      %   - dX: derivatives w.r.t. this layer's inputs, used by previous
      %     layer
      %   - dW: derivatives w.r.t. layer's weights (empty)
      %   - db: derivatives w.r.t. layer's biases (empty)
      dW = [];
      db = [];
      dY = errorFun.derive(this.Z, T);
      dX = reshape(dY, [size(dY, 1), this.inputShape]);
    end
    
    function [dW, db] = inputBackward(~, ~, ~)
      %inputBackward Calculates layer's derivatives if it is the first
      %layer
      %   Does nothing.
      % Outputs:
      %   - dW: an empty array
      %   - db: an empty array
      dW = [];
      db = [];
    end
    
    function updateParameters(~, ~, ~)
      %updateParameters Does nothing as this layer has no parameters
      %   Does nothing
    end
  end
end

