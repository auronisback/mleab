%Author: Francesco Altiero
%Date: 08/12/2018

classdef GenericLayer < neuralnet.layer.OutputLayer & neuralnet.layer.HiddenLayer
  %GENERICLAYER Manages a generic network's layer.
  %   Defines data and operation to manage a generic neural network layer,
  %   in which it is possible to specify the number of nodes in the layer
  %   and their activation function. All nodes in the layer are considered
  %   to have the same activation function. It can be used as an hidden or
  %   an output layer in the net.
  
  properties
    weights %Layer weights
    biases %Biases of the layer
    nodeNumber %Number of nodes in this layer
    actFun %Activation function for the layer
  end
  
  methods
    function this = GenericLayer(inputDim, nodeNumber)
      %GENERICLAYER Construct a new generic layer by its parameters
      %   Constructor that creates a new net layer specifying the dimension
      %   of the input, number of nodes in the array and the identity
      %   activation function for the layer. Layer's Weights and biases
      %   are initialized randomly.
      %   Inputs:
      %     - inputDim: array with input dimensionality
      %     - nodeNumber: number of nodes in the layer
      
      %Calling parent's contructor with output dimensionality equal as the
      %number of nodes in the layer
      this = this@neuralnet.layer.OutputLayer(inputDim, nodeNumber);
      %Calling hidden layer's parent constructor
      this = this@neuralnet.layer.HiddenLayer(inputDim);
      %Checking node number and activation function
      assert(nodeNumber > 0, 'GenericLayer:invalidNodeNumber', ...
          sprintf('Invalid node number: %d', nodeNumber));
      %Ok, initializing properties
      this.nodeNumber = nodeNumber;
      %Default: the activation is the identity
      this.actFun = neuralnet.activation.Identity(this);
      %Initializing weights and biases
      this.initializeWeightsAndBiases();
    end
    
    function Z = forwardTraining(this, X)
      %propagate Propagate forward the input.
      %   In concrete subclass, this method has to calculate output Z from 
      %   the given input X. It should cache data such activations and
      %   outputs in order to be used in training of the net.
      %
      %   Inputs:
      %     - X: the data that have to be processed by layer
      %   Outputs:
      %     - Z: the value calculated by the layer
      
      %Calculating activations
      this.activations = X * this.weights.' ...
        + repmat(this.biases, size(X, 1), 1);
      %Evaluating activation function directly on the layer
      this.outputs = this.actFun.eval();
      %Returning
      Z = this.outputs;
    end
      
    function Z = forward(this, X)
      %predict Predict the input propagating forward.
      %   In concrete subclass, this method must calculate output Z from the
      %   given input X. When not training the net, this method should be
      %   used because it is implemented not caching intermediate data.
      %
      %   Inputs:
      %     - X: the data that have to be processed by layer
      %   Outputs:
      %     - Z: the value calculated by the layer
      
      %Calculating activations
      A = X * this.weights.' ...
        + repmat(this.biases, size(X, 1), 1);
      %Evaluating activation function on A
      Z = this.actFun.eval(A);
    end
    
    function dhid = backward(this, delta, Wnext)
      %backward Performs the backward propagation in order to tune weights.
      %   In concrete implementor, this method should perform the back
      %   propagation of input data given deltas obtained by next layer.
      %
      %   Inputs:
      %     - delta: delta value from the next layer, if this layer is not an
      %         output layer
      %     - Wnext: matrix of weights of the next layer
      %   Outputs:
      %     - dhid: the delta value for nodes in this hidden layer 
      dhid = (delta * Wnext) .* this.actFun.derivative();
    end
    
    function dout = calculateDeltaOut(this, errFun, T)
      %calculateDeltaOutput Gets deltas when this is an output layer
      %   Calculates the delta for the layer when it is an output layer.
      %   Since deltas are backpropagated and this is the last layer, it
      %   uses the error function along with net's outputs, that are the
      %   outputs of this very layer.
      %
      %   Inputs:
      %     - errFun: the error function, used to calculate derivatives of
      %         the error w.r.t. net's outputs
      %     - T: target values for the outputs
      %   Outputs:
      %     - dout: the delta for nodes in this output layer
      
      %Calculating the activation function derivative matrix and multiply
      %it point-wise with error function derivative
      dout = this.actFun.derivative() .* ...
        errFun.derivative(this.outputs, T);
    end
    
    function udpdateWeightsAndBiases(this, deltaW, deltaB)
      %updateWeightsAndBiases Performs the updating of the layer's parameters
      %   Updates weights and biases of this layer given deltas calculated
      %   using any weight update methods.
      %
      %   Inputs:
      %     - deltaW: the differential that has to be added to weights
      %     - deltaB: the differential that has to be added to biases
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
  end
end

