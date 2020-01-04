classdef Layer < handle
  %LAYER Layer for a neural network
  %   Defines common operations and data used into a NeuralNetwork object.
  %   A layer can propagate forward to next layer, or backward to the
  %   previous layer.
  
  properties(Constant, Access = protected)
    NAME_UNKNOWN = 'unknown'; % Constant used when name was not specifyied
  end
  
  properties(SetAccess = protected, GetAccess = public)
    name;  % Name of the layer
    network;  % A pointer to the network this layer is used in
    position;  % Position of the layer in the net
    inputSize;  % Layer's input dimensions
    outputSize;  % Layer's output dimensions
    next;  % Next layer in the network
    previous;  % Previous layer in the network
    activation;  % Activation function object for the layer
    A;  % Cached activation values
    Z;  % Cached output values
  end
  
  methods
    function this = Layer(inputSize, name)
      %LAYER Construct an instance of this class
      %   Creates a layer with given dimensions and optional name.
      %
      %   Inputs:
      %     - inputSize: the array with sizes of input elements
      %     - name: a string with the optional name of the layer
      
      this.inputSize = inputSize;
      if(nargin > 1) % Name argument given
        this.name = name;
      else
        this.name = this.NAME_UNKNOWN;
      end
    end
    
    function setNetwork(this, network)
      %setNetwork Sets the network for this layer
      %   Sets the network which contains this layer.
      % Inputs:
      %   - network: the neural network object to link
      assert(isa(network, 'neural.NeuralNetwork'), 'Layer:invalidNetwork', ...
        'Invalid network type: %s', class(network));
      this.network = network;
    end
    
    function setPosition(this, position)
      %setPosition Sets the position of the layer in the network
      %   Updates the position of the layer to the position it occupies in
      %   the network.
      % Inputs:
      %   - position: the index of this layer in the list of layers
      assert(position >= 0, 'Layer:invalidPosition', ...
        'Invalid position given: %d', position);
      this.position = position;
    end
    
    function setNext(this, next)
      %setNext Sets the next layer
      %   Sets the subsequential layer in the net.
      %
      %   Inputs:
      %     - layer: the layer after this
      assert(isa(next, 'neural.layers.Layer') || isempty(next), ...
        'Layer:invalidNext', 'Invalid type for next layer: %s', ...
        class(next));
      this.next = next;
    end
    
    function setPrevious(this, previous)
      %setNext Sets the previous layer
      %   Sets the preceding layer in the net.
      %
      %   Inputs:
      %     - layer: the layer before this
      assert(isa(previous, 'neural.layers.Layer') || isempty(previous), ...
        'Layer:invalidPrevious', 'Invalid type for previous layer: %s', ...
        class(previous));
      this.previous = previous;
    end
    
    function setActivation(this, activation)
      %setActivation Sets the activation function for the layer
      %   Sets the activation function which will be used by the layer.
      % Inputs:
      %   - activation: the activation function object
      assert(isa(activation, 'neural.activations.ActivationFunction'), ...
        'Layer:invalidActivation', 'Invalid type for activation: %s', ...
        class(activation));
      this.activation = activation;
    end
  end
  
  methods(Abstract)
    [W, b] = getWeightsAndBiases(this);
      %getWeightsAndBiases Gets weights and biases of the layer
      %   Gets the weights and biases of this layer.
      % Outputs:
      %   - W: weights of the layer
      %   - b: biases of the layer
    
    setWeigthsAndBiases(this, W, b);
      %setWeightsAndBiases Sets weight and bias values for the layer
      %   Sets the weights and biases values in the layer.
      % Inputs:
      %   - W: weights to be set
      %   - b: biases to be set
    
    Y = predict(this, X);
      %predict Evaluates the layer on given input
      %   Performs operations of the layer in order to produce results.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Y: calculated values of the given input
      
    Y = forward(this, X);
      %forward Performs a forward propagation pass during training
      %   Calculates the layer's output using given input, caching data in
      %   order to fasten the training process.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Y: calculated values of given input
      
    dX = backward(this, dZ);
      %backward Performs a back propagation in the layer
      %   Calculates the layer's derivative as in back-propagation
      %   algorithm, producing derivative with respect to input values.
      % Inputs:
      %   - dZ: derivatives of the next layer
      % Outputs:
      %   - dX: derivatives of inputs which will be propagated back
      
    updateWeightsAndBiases(this, deltaW, deltaB);
      %updateWeightsAndBiases Updates layer's weights and biases
      %   Perform an update of weights and biases of the layer using delta
      %   values calculated. These values will be summed to actual values
      %   of weights and biases.
      % Inputs:
      %   - deltaW: quantity which will be added to weights
      %   - deltaB: quantity which will be added to biases
  end
end

