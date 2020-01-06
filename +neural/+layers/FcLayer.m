classdef FcLayer < neural.layers.Layer
  %FCLAYER Fully-Connected layer
  %   MAnages a fully connected layer used in neural network models.
  
  properties(Constant)
    NAME_FC = 'FC';  % Constant for layer's type
  end
  
  properties(SetAccess = private)
    W;  % Weights of the layer
    b;  % Biases in the layer
    numNodes;  % Number of nodes in the layer
  end
  
  methods
    function this = FcLayer(inputSize, numNodes, activation, name)
      %FCLAYER Construct an instance of fully connected layer
      %   Creates a Fully-Connected layer object which is used in neural
      %   networks.
      % Inputs:
      %   - inputSize: input shape
      %   - numNodes: number of nodes in the layer
      %   - activation: layer's activation function
      %   - name: the name of the layer, optional
      this = this@neural.layers.Layer(inputSize, '');
      this.numNodes = numNodes;
      if isa(activation, 'neural.activations.ActivationFunction')
        this.activation = activation;
        this.activation.setLayer(this);
      else
        error('FcLayer:InvalidActivationFunction', ['Invalid activation',...
          ' function type: %s'], class(activation));
      end
      if nargin > 3
        this.name = name;
      else
        this.name = neural.layers.FcLayer.NAME_FC;
      end
      % Initializing weights and biases
      this.outputSize = this.numNodes;
      this.initializeWeightsAndBiases();
    end
    
    function [W, b] = getWeightsAndBiases(this)
      %getWeightsAndBiases Gets weights and biases of the layer
      %   Gets the weights and biases of this layer.
      % Outputs:
      %   - W: weights of the layer
      %   - b: biases of the layer
      W = this.W;
      b = this.b;
    end
    
    function setWeigthsAndBiases(this, W, b)
      %setWeightsAndBiases Sets weight and bias values for the layer
      %   Sets the weights and biases values in the layer.
      % Inputs:
      %   - W: weights to be set
      %   - b: biases to be set
      assert(all(size(W) == size(this.W), 'all'), ...
        'FcLayer:invalidWeightSize', 'Invalid weight size');
      assert(all(size(b) == size(this.b), 'all'), ...
        'FcLayer:invalidBiasSize', 'Invalid bias size');
      this.W = W;
      this.b = b;
    end
    
    function Y = predict(this, X)
      %predict Evaluates the layer on given input
      %   Performs operations of the layer in order to produce results.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Y: calculated values of the given input
      Y = X * this.W' + repmat(this.b, size(X, 1), 1);
      Y = this.activation.evaluate(Y);
      if ~isempty(this.next)
        Y = this.next.predict(Y);
      end
    end
      
    function Y = forward(this, X)
      %forward Performs a forward propagation pass during training
      %   Calculates the layer's output using given input, caching data in
      %   order to fasten the training process.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Y: calculated values of given input
      this.A = X * this.W' + repmat(this.b, size(X, 1), 1);
      this.Z = this.activation.evaluate(this.A);
      Y = this.Z;
      if ~isempty(this.next)
        Y = this.next.forward(Y);
      end
    end
      
    function dX = backward(this, dZ)
      %backward Performs a back propagation in the layer
      %   Calculates the layer's derivative as in back-propagation
      %   algorithm, producing derivative with respect to input values.
      % Inputs:
      %   - dZ: derivatives of the next layer
      % Outputs:
      %   - dX: derivatives of inputs which will be propagated back
      delta = dZ .* this.activation.derive();  % Common term for all derivatives
      dW = delta' * this.previous.Z;  % Taking output of previous layer
      db = sum(delta);  % Summing delta to have derivative of bias
      dX = delta * this.W;
      N = size(dZ, 1);  % Caching batch size
      % Using optimizer in order to update weights
      [deltaW, deltaB] = this.network.optimizer.calculateDeltas(dW, db, ...
        N, this.position);
      this.updateWeightsAndBiases(deltaW, deltaB);
      this.previous.backward(dX);
    end
    
    function updateWeightsAndBiases(this, deltaW, deltaB)
      %updateWeightsAndBiases Updates layer's weights and biases
      %   Perform an update of weights and biases of the layer using delta
      %   values calculated. These values will be summed to actual values
      %   of weights and biases.
      % Inputs:
      %   - deltaW: quantity which will be added to weights
      %   - deltaB: quantity which will be added to biases      
      this.W = this.W + deltaW;
      this.b = this.b + deltaB;
    end
  end
  
  methods(Access = private)
    function initializeWeightsAndBiases(this)
      %initializeWeightsAndBiases Inits layer's parameters
      this.W = 1 - 2 * randn([this.numNodes, prod(this.inputSize, 'all')]);
      this.b = 1 - 2 *randn([1, this.numNodes]);
    end
  end
end

