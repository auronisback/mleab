classdef SoftmaxLayer < neuralnet.layer.OutputLayer
  %SOFTMAXLAYER Manages a softmax activation function layer
  %   Defines a neural network output layer with the softmax activation
  %   function, in order to output probability values for a classification
  %   problem. It is used when a cross-entropy activation function is given
  %   when training the net.
  
  properties
    weights %Weights of the layer
    biases %Biases of the net
    nodeNumber %Number of nodes in the net
  end

  methods
    function this = SoftmaxLayer(inputDim, nodeNumber)
      %SoftmaxLayer Constructs a new softamx layer object
      %   Constructor that creates an output layer specifying the
      %   dimensionality of the inputs to the level and the number of nodes
      %   in the layer. Weights and biases are initially set to random
      %   values in [-1, 1].
      %
      %   Inputs:
      %     - inputDim: array with input dimensionality
      %     - nodeNumber: number of nodes in the layer
      
      %Calling parent's contructor with output dimensionality equal as the
      %number of nodes in the layer
      this = this@neuralnet.layer.OutputLayer(inputDim, nodeNumber);
      %Checking node number and activation function
      assert(nodeNumber > 0, 'SoftmaxLayer:invalidNodeNumber', ...
          sprintf('Invalid node number: %d', nodeNumber));
      %Ok, initializing properties
      this.nodeNumber = nodeNumber;
      %Initializing weights and biases
      this.initializeWeightsAndBiases();
    end
    
    function Z = forwardTraining(this, X)
      %propagate Propagate forward the input.
      %   Calculate output Z from the given input X using softmax. Data
      %   such as activations and outputs will be cached in order to have a
      %   quick access to them while training.
      %
      %   Inputs:
      %     - X: the data that have to be processed by layer
      %   Outputs:
      %     - Z: the softmax value for data
      
      %Calculating activations
      this.activations = X * this.weights.' ...
        + repmat(this.biases, size(X, 1), 1);
      %Evaluating activation function directly on the layer
      this.outputs = this.softmax(this.activations);
      %Returning
      Z = this.outputs;
    end
      
    function Z = forward(this, X)
      %predict Predict the input propagating forward.
      %   Calculates output Z from the layer's input X. This method does
      %   not cache any data and it is used when the net has been already
      %   trained, in order to use less space.
      %
      %   Inputs:
      %     - X: the data that have to be processed by layer
      %   Outputs:
      %     - Z: the value calculated by the layer
      
      %Calculating activations
      A = X * this.weights.' ...
        + repmat(this.biases, size(X, 1), 1);
      %Evaluating activation function on A
      Z = this.softmax(A);
    end

    function dout = calculateDeltaOut(this, errFun, T)
      %calculateDeltaOut Calculates the delta for the output layer
      %   Performs the calculation of delta out values used in training the
      %   net. It is used only when the error function is the cross-entropy
      %   error function.
      %   Inputs:
      %     - errFun: the error function, that has to be a cross-entropy
      %     - T: labels for the training set
      %   Outputs:
      %     - dout: delta values for this layer, used as an output layer
      
      %Checking error function
      assert(isa(errFun, 'neuralnet.train.error.CrossEntropy'), ...
        'SoftmaxLayer:invalidErrorFunction', ...
        'Softmax layer can be used only with cross-entropy error function');
      %Returning the subtraction between outputs and labels
      dout = this.outputs - T;
    end

    function udpdateWeightsAndBiases(this, deltaW, deltaB)
      %updateWeightsAndBiases Performs the updating of the layer's parameters
      %   Updates weights and biases of this layer given deltas calculated
      %   using any weight update methods.
      %
      %   Inputs:
      %     - deltaW: the differential that has to be added to weights
      %     - deltaB: the differential that has to be added to biases
      
      %Differential is just added to weights and biases
      this.weights = this.weights + deltaW;
      this.biases = this.biases + deltaB;
    end
  end

  methods (Access=private)
    function initializeWeightsAndBiases(this)
      %initializeWeightsAndBiases Initializes randomly in [-1, 1] weights
      %and biases of the layer.
      %   Initializes weights and biases of the matrix, using the input
      %   dimensionality and the node number.
      this.weights = 1 - 2 * rand([this.nodeNumber, this.inputDim]);
      this.biases = 1 - 2 * rand(1, this.nodeNumber);
    end

    function Z = softmax(this, A)
      %softmax Calculates the softmax output using softmax
      %   Performs the calculation of the softmax output on the given
      %   activations.
      %
      %   Inputs:
      %     - A: layer's activation values
      %   Outputs:
      %     - Z: the softmax evaluation on activation values

      expA = exp(A);
      Z = expA ./ repmat(sum(expA, 2), 1, this.nodeNumber);
    end
  end
end

