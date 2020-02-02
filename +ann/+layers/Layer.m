classdef Layer < handle
  %LAYER Baseline class used to generalize network's layers
  %   Defines basic properties and operations which belongs to a network
  %   layer.
  
  properties(SetAccess = protected)
    name;  % Layer's name
    inputShape;  % Dimensions of each input data
    outputShape;  % Dimension of the layer's output
    activation;  % Layer's activation function
    A;  % Cached activation values
    Z;  % Cached outputs
  end
  
  methods(Abstract)
    [W, b] = getParameters(this);
      %getParameters Gets weights and biases of the layer
      %   Getter for weights and biases of the layer.
      % Outputs:
      %   - W: weights
      %   - b: biases
    
    setParameters(this, W, b);
      %setParameters Sets layer's parameters
      %   Setter for weights and biases of the layer.
      % Inputs:
      %   - W: weights
      %   - b: biases
    
    Z = predict(this, X);
      %predict Performs a propagation in the layer
      %   Propagates the input in order to evaluate layer's output.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Z: layer's output with respect to input X
      
    Z = forward(this, X);
      %forward Propagates the layer, caching data for training
      %   Propagates the input in order to evaluate layer's output and
      %   caches intermediate data used to train the network.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Z: layer's output with respect to input X
    
    [dX, dW, db] = backward(this, dZ, X);
      %backward Backpropagates errors through the layer
      %   Calculates error derivatives of the layer using derivatives of
      %   next layer and the layer's input. Used if the layer is not
      %   network's output layer.
      % Inputs:
      %   - dZ: derivatives of error w.r.t. layer's output
      %   - X: layer's input
      % Outputs:
      %   - dX: derivatives w.r.t. layer's input, used in previous layer
      %   - dW: derivatives w.r.t. layer's weights
      %   - db: derivatives w.r.t. layer's biases
      
    [dX, dW, db] = outputBackward(this, errorFun, X, T);
      %outputBackward Backpropagates errors if the layer is an output layer
      %   Manages backpropagation if this layer is the last layer in the
      %   network, using error function, target values and layer's input.
      % Inputs:
      %   - errorFun: error function used to derive error with respect to
      %     network's output
      %   - X: layer's input
      %   - T: target values for the input
      % Outputs:
      %   - dX: derivatives w.r.t. this layer's inputs, used by previous
      %     layer
      %   - dW: derivatives w.r.t. layer's weights
      %   - db: derivatives w.r.t. layer's biases
      
    [dW, db] = inputBackward(this, dZ, X);
      %inputBackward Calculates layer's derivatives if it is the first
      %layer
      %   Evaluates parameter's derivatives when the layer is the first
      %   layer in the net. This method is added in order to avoid the
      %   evaluation of error function w.r.t. layer's input as the error
      %   isn't going to be backpropagated anymore.
      % Inputs:
      %   - dZ: derivatives w.r.t. layer's output
      %   - X: network's inputs
      % Outputs:
      %   - dW: derivatives w.r.t. weights
      %   - db: derivatives w.r.t. biases
      
    updateParameters(this, deltaW, deltaB);
      %updateParameters Updates layer's parameters
      %   Updates parameters of the layer using delta values calculated by
      %   an optimizer.
      % Inputs:
      %   - deltaW: delta values for weights
      %   - deltaB: delta values for biases
      
    s = toString(this);
      %toString Converts the layer into a string representation
      %   Converts a layer into its string representation.
      % Output:
      %   - s: a string with information on the layer
  end
end

