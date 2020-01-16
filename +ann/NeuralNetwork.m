classdef NeuralNetwork < handle
  %NEURALNETWORK Manages neural network models
  %   Manages data and operations used to work with sequential artifical
  %   neural networks.
  
  properties(SetAccess = private)
    layers;  % Cell-array with network's layers
    depth;  % Depth of the network
    errorFun;  % Error function used to train the network
  end
  
  methods
    function this = NeuralNetwork(layers, errorFun)
      %NEURALNETWORK Creates a new neural network
      %   Creates a neural network optionally specifying layers and error
      %   function used in the network.
      % Inputs:
      %   - layers: cell-array of layers
      %   - errorFun: error function used to train the network
      if nargin < 1
        layers = {};  % No layer specified
      end
      if nargin < 2
        errorFun = [];  % Empty error function
      end
      this.layers = layers;
      this.depth = size(layers, 2);
      this.errorFun = errorFun;
    end
    
    function addLayer(this, layer, index)
      %addLayer Appends or adds a layer to the network
      %   Adds a new layer to the network in the specified position. If
      %   position it is not given, the layer will be appended.
      % Inputs:
      %   - layer: the layer which has to be added
      %   - index: position at which the layer will be inserted. If not
      %     given, it will be appended at the end of the network
      assert(isa(layer, 'ann.layers.Layer'), ...
        'NeuralNetwork:invalidLayer', 'Invalid layer type: %s', ...
        class(layer));
      if nargin < 3
        index = this.depth + 1;
      end
      assert(index > 0 && index <= this.depth + 1,...
        'NeuralNetwork:invalidPosition', 'Invalid position: %d', index);
      if index == this.depth + 1  % Appending
        this.layers{this.depth + 1} = layer;
      elseif index == 1  % Inserting at start
        this.layers = [{layer}, this.layers];
      else  % Inserting in the middle
        this.layers = {this.layers{1:index - 1}, ...
          layer, this.layers{index:end}};
      end
      this.depth = this.depth + 1;  % Incrementing network's depth
    end
    
    function removeLayer(this, index)
      %removeLayer Removes a layer at given position
      %   Deletes a layer from the network specifying its index.
      % Inputs:
      %   - index: the position of the layer which will be removed
      assert(index > 0 && index <= this.depth, ...
        'NeuralNetwork:invalidPosition', 'Invalid position: %d', index);
      this.layers(index) = [];
      this.depth = this.depth - 1;
    end
    
    function setErrorFunction(this, errorFun)
      %setErrorFunction Sets the network's error function
      %   Sets the error function which will be used by the network in
      %   order to train.
      % Inputs:
      %   - errorFun: the error function instance
      this.errorFun = errorFun;
    end
    
    function print(this)
      %print Prints network's topology
      %   Prints names of layers on the standard output.
      for l = 1:this.depth
        fprintf('%s -> ', this.layers{l}.name);
      end
      fprintf('Y\n');
    end
    
    function [W, b] = getParameters(this)
      %getParameters Gets network's parameter
      %   Gets all parameters in the network, as a cell-array with values
      %   for each layer.
      % Outputs:
      %   - W: cell-array with weights for all layers in the network
      %   - b: cell-array with biases for all layers in the network
      W = cell(1, this.depth);
      b = cell(1, this.depth);
      for l = 1:this.depth
        % Getting all layers' weights and biases
        [W{l}, b{l}] = this.layers{l}.getParameters();
      end
    end
    
    function setParameters(this, W, b)
      %setParameters Sets network's parameters
      %   Sets all parameters in the network.
      % Inputs:
      %   - W: a cell-array for weights with # of cells equals to network's
      %     depth
      %   - b: a cell-array for biases with # of cells equals to network's
      %     depth
      assert(size(W, 2) == this.depth, 'NeuralNetwork:invalidWeights', ...
        'Invalid size for weights');
      assert(size(b, 2) == this.depth, 'NeuralNetwork:invalidBiases', ...
        'Invalid size for biases');
      for l = 1:this.depth
        this.layers{l}.setParameters(W{l}, b{l});
      end
    end
    
    function Y = predict(this, X)
      %predict Forward propagate the network to predict results
      %   Propagates input in the network in order to obtain prediction for
      %   the given input.
      % Inputs:
      %   X: input which will be propagated
      % Output:
      %   Y: network's predicted values on input X
      
      Z = X;
      for l = 1:this.depth
        Z = this.layers{l}.predict(Z);
      end
      Y = Z;
    end
    
    function Y = forward(this, X)
      %forward Forward propagate in order to train the network
      %   Evaluates the network on given input in order to obtain outputs
      %   used to train the net. Performs same operations as the 'predict'
      %   method, but caches intermediate values in order to speed-up the
      %   learning process.
      % Inputs:
      %   X: input which will be forwarded
      % Output:
      %   Y: network's output values on input X
      Z = X;
      for l = 1:this.depth
        Z = this.layers{l}.forward(Z);
      end
      Y = Z;
    end
    
    function [dW, db] = backpropagate(this, X, T)
      %backpropagate Backpropagates the network to obtain derivatives
      %   Manages the backpropagation in the network in order to calculate
      %   derivatives w.r.t. all layers' parameters.
      % Inputs:
      %   - X: input of network
      %   - T: target values of the input
      % Outputs:
      %   - dW: a cell-array of derivatives w.r.t. layers' weights
      %   - db: a cell-array of derivatives w.r.t. layers' biases
      dW = cell(1, this.depth);  % Pre-allocating
      db = cell(1, this.depth);
      if this.depth == 1  % Single layer network
        Z = X;  % Setting last layer's input to X
      else 
        Z = this.layers{this.depth - 1}.Z;
      end
      % Start from last layer
      [dX, dW{this.depth}, db{this.depth}] = ...
        this.layers{this.depth}.outputBackward(this.errorFun, Z, T);
      % Backwarding up to second layer
      for l = this.depth - 1:-1:2
        Z = this.layers{l - 1}.Z;
        [dX, dW{l}, db{l}] = this.layers{l}.backward(dX, Z);
      end
      % Managing first layer if the net had more than one layer
      if this.depth > 1
        [dW{1}, db{1}] = this.layers{1}.inputBackward(dX, X);
      end
    end
    
    function updateParameters(this, deltaW, deltaB)
      %updateParameters Updates layers' parameters
      %   Updates parameters of all layers using delta values given by an
      %   optimizer.
      % Inputs:
      %   - deltaW: cell-array with values to add to each layer's weights
      %   - deltaB: cell-array with values to add to each layer's biases
      
      % Calling the update method of each layer
      for l = 1:this.depth
        this.layers{l}.updateParameters(deltaW{l}, deltaB{l});
      end
    end
    
  end
end

