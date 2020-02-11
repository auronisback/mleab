classdef FcLayer < ann.layers.Layer
  %FCLAYER Defines a Fully-Connected (FC) layer in the network
  %   Defines operations and properties for a fully-connected, namely
  %   Dense, layer of a neural network.
  
  properties(SetAccess = private)
    nodeNum;  % Number of nodes in the FC-layer
    W;  % Matrix with weigths
    b;  % Biases array
  end
  
  methods
    function this = FcLayer(inputShape, nodeNumber, activation)
      %FCLAYER Creates a new fully-connected layer
      %   Creates a fully connected layer used in neural network,
      %   specifying the shape of its input, the number of nodes and the
      %   activation function used.
      % Inputs:
      %   - inputShape: dimension of layer's input
      %   - nodeNumber: number of hidden units in the layer
      %   - activation: layer's activation function
      this.inputShape = prod(inputShape, 'all');
      this.activation = activation;
      this.activation.setLayer(this);
      this.nodeNum = nodeNumber;
      this.outputShape = nodeNumber;  % Output equals to # of nodes
      this.initializeWeightsAndBiases();
      %Initializing name
      this.name = 'FC';
    end
    
    function [W, b] = getParameters(this)
      %getParameters Gets weights and biases of the layer
      %   Getter for weights and biases of the layer.
      % Outputs:
      %   - W: weights
      %   - b: biases
      W = this.W;
      b = this.b;
    end
    
    function setParameters(this, W, b)
      %setParameters Sets layer's parameters
      %   Setter for weights and biases of the layer.
      % Inputs:
      %   - W: weights, which should have same size of layer's weights
      %   - b: biases, which should have same size of layer's biases
      assert(all(size(W) == size(this.W)), 'FcLayer:invalidWeights', ...
        'Invalid weight size');
      assert(all(size(b) == size(this.b)), 'FcLayer:invalidBiases', ...
        'Invalid bias size');
      this.W = W;
      this.b = b;
    end
    
    function Z = predict(this, X)
      %predict Evaluates the FC-layer on input
      %   Performs a matrix multiplication, an addition and the evaluation
      %   of activation function in order to produce layer's output.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Z: output value for given input
      Z = this.activation.eval(...
        X * this.W.' + repmat(this.b, size(X, 1), 1));
    end
    
    function Z = forward(this, X)
      %forward Evaluates the FC-layer, caching results used in training
      %   Performs a forward pass of the layer and caches output and
      %   activation values in order to use them in training.
      % Inputs:
      %   - X: layer's input
      % Outputs:
      %   - Z: output for given input
      this.A = (X * this.W.') + repmat(this.b, size(X, 1), 1);
      this.Z = this.activation.eval(this.A);
      Z = this.Z;
    end
    
    function [dX, dW, db] = backward(this, dZ, X)
      %backward Performs a backward pass in the layer
      %   Manages backpropagation on the layer in order to calculate
      %   derivatives w.r.t. weights and biases in the layer. It also
      %   calculates derivatives w.r.t. input used for backpropagation in
      %   previous layers.
      % Inputs:
      %   - dZ: derivatives of error w.r.t. layer's output
      %   - X: layer's input values
      % Outputs:
      %   - dX: derivatives w.r.t. layer's input
      %   - dW: derivatives w.r.t. layer's weights
      %   - db: derivatives w.r.t. layer's biases
      dA = this.activation.derive(dZ);
      dW = dA.' * X;
      db = sum(dA);
      dX = dA * this.W;
    end
    
    function [dX, dW, db] = outputBackward(this, errorFun, X, T)
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
      %   - dW: derivatives w.r.t. layer's weights
      %   - db: derivatives w.r.t. layer's biases
      
      % Calculating derivatives w.r.t. activation
      if isa(errorFun, 'ann.errors.CrossEntropy') && ...
          (isa(this.activation, 'ann.activations.Sigmoid') || ...
           isa(this.activation, 'ann.activations.Softmax'))
         % Softmax or sigmoid layer with cross-entropy
        dA = this.Z - T;
      else
        % General calculation of derivative w.r.t. activation
        dY = errorFun.derive(this.Z, T);
        dA = this.activation.derive(dY);
      end
      dW = dA.' * X;
      db = sum(dA);
      dX = dA * this.W;
    end
    
    function [dW, db] = inputBackward(this, dZ, X)
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
      dA = this.activation.derive(dZ);
      dW = dA.' * X;
      db = sum(dA);
    end
    
    function updateParameters(this, deltaW, deltaB)
      %updateParameters Updates this FC-layer's parameters
      %   Updates parameters of the layer using delta values calculated by
      %   an optimizer.
      % Inputs:
      %   - deltaW: delta values for weights
      %   - deltaB: delta values for biases
      this.W = this.W + deltaW;
      this.b = this.b + deltaB;
    end
    
    function reinitialize(this)
      %reinitialize Re-initializes weights and biases for the layer.
      this.initializeWeightsAndBiases();
    end
    
    function s = toString(this)
      %toString Gets a string representation of the object
      %   Converts the layer into a string.
      % Output:
      %   s: the string representing this FC layer
      s = [this.name, sprintf(' (in: %d, out: %d)', prod(this.inputShape), ...
        this.nodeNum)];
    end
  end
  
  methods(Access = private)
    function initializeWeightsAndBiases(this)
      %initializeWeightsAndBiases Initializes weights and biases of the
      %layer, in order to make them uniform random in [-1, 1]
      this.W = 1 - 2 * rand(this.nodeNum, this.inputShape);
      this.b = 1 - 2 * rand(1, this.nodeNum);
    end
  end
end

