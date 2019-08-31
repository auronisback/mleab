%Author: Francesco Altiero
%Date: 08/12/2018

classdef NeuralNet < handle
  %NEURALNET Manages neural network objects
  %   Defines properties and operations in order to use a Neural Network,
  %   defined by its layers. Layers can be added, retrieved and removed
  %   from the network, and it can propagate inputs forward and performs a
  %   backward propagation in order to update weights in the layer. It can
  %   also predict output on data, after training. The prediction
  %   operation doesn't cache intermediate data, as layers' activation and
  %   output values, in order to reduce memory consumption.
  
  properties
    d %Input dimensionality
    depth %Number of layers of the net, excluding output layer
    layerNum %Total number of layers in the net, including output layer
  end
  
  properties (Access = private)
    hiddenLayers %Cell-array of the net hidden layers
    outputLayer %Net's output layer
  end
  
  methods
    function this = NeuralNet(dimensions)
      %NEURALNET Creates a NeuralNet object specifying its properties
      %   This constructor creates a neural net using data specified in
      %   properties.
      %
      %   Inputs:
      %     - dimensions: an integer positive value specifying features'
      %         dimensionality
      
      %Checking dimensions
      assert(isvector(dimensions), 'NeralNet:dimensionsNotArray', ...
        'Given dimensions is not an array');
      assert(all(dimensions(:) > 0), 'NeuralNet:invalidInputDimensionality', ...
          sprintf('Invalid network input dimension: %d', dimensions));
      %Creating the object
      this.d = dimensions;
      %Initializing layers
      this.hiddenLayers = cell(1, 0);
      this.depth = 0;
    end
    
    function addHiddenLayer(this, layer, index)
      %addLayer Adds a layer to the network at the specified index
      %   This method adds a new layer to the network, optionally
      %   specifying its index.
      %
      %   Inputs:
      %     - layer: the NetLayer object representing the layer that has to
      %         be added
      %     - index: the index in which the layer has to be inserted. If
      %         not given, the layer will be appended at the end of the
      %         list
      
      %Asserting passed layer is an hidden layer
      assert(isa(layer, 'neuralnet.layer.HiddenLayer'), ...
        'NeuralNet:invalidLayer', 'Passed layer is not an hidden layer');
      %Finding the position where the layer has to be put
      if ~exist('index', 'var') || index > this.depth
        index = this.depth + 1;
      end
      %Updating depth
      this.depth = this.depth + 1;
      %Creating the new layer level
      newLayers = cell(1, this.depth);
      j = 1; %index used to point to the previous layers array
      for i = 1:this.depth
        if i ~= index
          newLayers{i} = this.hiddenLayers{j};
          j = j + 1;
        else
          %Adding the new layer
          newLayers{i} = layer;
        end
      end
      %Setting the new layer
      this.hiddenLayers = newLayers;
    end
    
    function removeHiddenLayer(this, index)
      %removeLayer Removes the layer at specified index
      %   This method performs the deletion of a net's layer, specified by
      %   the given index.
      %
      %   Inputs:
      %     - index: the positive integer specifying which layer has to be
      %       removed from the network
      
      assert(index > 0 && index <= this.depth, 'NeuralNet:invalidIndex', ...
        sprintf('Invalid index for layer to remove: %d', index));
      %Creating the new layers cell-array
      this.depth = this.depth - 1;
      newLayers = cell(1, this.depth);
      j = 1;
      for i = 1:this.depth + 1
        if i ~= index
          %Layer will be preserved
          newLayers{j} = this.hiddenLayers{i};
          j = j + 1;
        end
      end
      %Ok, updating net's layers
      this.hiddenLayers = newLayers;
    end
    
    function setOutputLayer(this, layer)
      %setOutputLayer Sets the net's output layer
      %   Adds the output layer to the net. Any pre-existent output layers
      %   will be overwritten.
      %
      %   Inputs:
      %     - layer: an OutputLayer object used to be the net's output
      %         layer
      
      %Checking validity
      assert(isa(layer, 'neuralnet.layer.OutputLayer'), ...
        'NeuralNet:invalidOutputLayer', 'Given output layer is invalid');
      this.outputLayer = layer;
    end
    
    function layer = getLayer(index)
      %getLayer Gets the net's layer of the given index
      %   Getter for layers in the net, used to return the layer specified
      %   at given index.
      %
      %   Inputs:
      %     - index: the integer index value of the layer that has to be
      %         retrieved
      %   Outputs:
      %     - layer: the net's layer with specified index
      
      %Ok, getting the layer
      if index < this.depth
        %Requested layer is the output layer
      end
    end
    
    function Y = forward(this, X)
      %predict Propagates the input forward in the whole net
      %   Performs a forward propagation on all net's layers in order to
      %   obtain the net response. Activations and outputs of the whole net
      %   are not cached, so this method should be used after the net has
      %   been trained.
      %
      %   Inputs:
      %     - X: the input data
      %   Outputs:
      %     - Y: the net outputs produced by forward propagation
      
      %Forwarding for all layers, without caching
      for l = 1:this.depth
        X = this.hiddenLayers{l}.forward(X);
      end
      %Forwarding to the output layer
      Y = this.outputLayer.forward(X);
    end
    
    function Y = forwardTraining(this, X)
      %forward Propagates the input forward in the whole net with caching
      %   Performs a forward propagation on all net's layers in order to
      %   obtain the net response. Activations and outputs of the whole net
      %   are cached, so this method is used in the training of the net.
      %
      %   Inputs:
      %     - X: the input data
      %   Outputs:
      %     - Y: the net outputs produced by forward propagation
      
      %Forwarding for all layers
      for l = 1:this.depth
        X = this.hiddenLayers{l}.forwardTraining(X);
      end
      %Forwarding to the output layer
      Y = this.outputLayer.forwardTraining(X);
    end

    function [derW, derB] = backward(this, X, T, errorFunction)
      %backward Performs the back propagation in the net.
      %   Executes the back propagation on net's layer in order to
      %   evaluate error function derivatives. Used to train the net.
      %   Inputs:
      %     - X: the net's inputs
      %     - T: labels for inputs
      %     - errorFunction: the error function used to perform the
      %         backpropagation
      %   Outputs:
      %     - derW: a cell-array with length equal to all layer of the net
      %         (including output layer) with derivatives of the error
      %         function w.r.t. weights
      %     - derB: a cell-array with lenght equal to all leyers in the net
      %         (including output layer) with derivatives of the error
      %         funciton w.r.t. biases
      
      %Propagating data in the net
      this.forwardTraining(X);
      %Preallocating derivatives
      derW = cell(1, this.depth + 1);
      derB = cell(1, this.depth + 1);
      %Calculating output layer's deltas (using cached outputs)
      delta = this.outputLayer.calculateDeltaOut(errorFunction, T);
      %Caching previous layer's output
      if this.depth == 0
        Z = X; %No hidden layers: calculating derivative of output
        derW{1} = delta.' * Z;
        derB{1} = sum(delta);
      else
        %At least one hidden layer, backpropagating
        Z = this.hiddenLayers{this.depth}.outputs;
        derW{this.depth + 1} = delta.' * Z;
        derB{this.depth + 1} = sum(delta);
        %Setting the next layer's weights
        Wnext = this.outputLayer.weights;
        %Backpropagating each layer except the first
        for l = this.depth:-1:2
          Z = this.hiddenLayers{l - 1}.outputs;
          %Calculating layer's delta
          delta = this.hiddenLayers{l}.backward(delta, Wnext);
          %Calculating derivatives
          derW{l} = delta.' * Z;
          derB{l} = sum(delta);
          %Updating weights of the next layer
          Wnext = this.hiddenLayers{l}.weights;
        end
        %First layer, previous output is the input
        Z = X;
        delta = this.hiddenLayers{1}.backward(delta, Wnext);
        derW{1} = delta.' * Z;
        derB{1} = sum(delta);
      end
    end
    
    function [weights, biases] = exportWeightsAndBiases(this)
      %exportWeightsAndBiases Returns weights and biases of the net
      %   Exports all weights and biases for each level of the net, as a
      %   cell-array. The arrays will have depth + 1 cells, where the last
      %   cell stores weights and biases of output layer.
      %
      %   Outputs:
      %     - weights: the cell-array with net's weights
      %     - biases: the cell-array with net's biases
      
      %Allocating
      weights = cell(1, this.depth + 1);
      biases = cell(1, this.depth + 1);
      %Storing weights and biases for hidden layers
      for l = 1:this.depth
        weights{l} = this.hiddenLayers{l}.weights;
        biases{l} = this.hiddenLayers{l}.biases;
      end
      %Doing the same for the output layer
      weights{this.depth + 1} = this.outputLayer.weights;
      biases{this.depth + 1} = this.outputLayer.biases;
    end
    
    function importWeightsAndBiases(this, W, b)
      %importWeightsAndBiases Sets the net's weights and biases to the
      %given arguments
      %   Performs operations in order to set all net's layers weights and
      %   biases to the arguments that are given. Arguments should be cell
      %   arrays with dimension equals to net's depth plus 1 (for the
      %   output layer).
      %
      %   Inputs:
      %     - W: weights' cell array. Its depth has to be equal to the 
      %         net's depth plus 1, and the last cell stores weights for
      %         the output layer
      %     - b: biases' cell array. Its length has to be equal to net's
      %         depth plus 1 and the last cell must store biases for the
      %         network's output layer
      
      %Asserting lengths are valid
      assert(size(W, 2) == this.depth + 1, 'NeuralNet:invalidImportW', ...
        'Imported weights size is invalid: %d', size(W, 2));
      assert(size(b, 2) == this.depth + 1, 'NeuralNet:invalidImportB', ...
        'Imported biases size is invalid: %d', size(b, 2));
      %Ok, setting weights
      for l = 1 : this.depth
        this.hiddenLayers{l}.weights = W{l};
        this.hiddenLayers{l}.biases = b{l};
      end
      %Managing output layer
      this.outputLayer.weights = W{this.depth + 1};
      this.outputLayer.biases = b{this.depth + 1};
    end
  end
end

