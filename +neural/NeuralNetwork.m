classdef NeuralNetwork < handle
  %NEURALNETWORK Manages a neural network
  %   Class used to instantiate feed-forward neural networks.
  %   It can store network layers and the topology of the net
  %   can be altered dynamically. The net can forward input to
  %   the layer in order to get outputs and can perform the
  %   backpropagation in order to update its weight.
  
  properties(SetAccess = private, GetAccess = public)
    inputLayer;  % Fake layer used for inputs
    outputLayer;  % Last layer in the network
    depth;  % Number of layers in the net
    optimizer;  % Network's weights optimizer
  end
  
  methods
    function this = NeuralNetwork(layers)
      %NEURALNETWORK Construct an instance of this class
      %   Creates a new NeuralNetwork object, optionally specifying layers
      %   in the network.
      %   
      %   Inputs:
      %     - layers: a cell-array of layers defining the topology of the
      %         network. If not specified, the network will have no layer
      
      if nargin > 0 && ~isempty(layers)
        this.depth = size(layers, 2);
        % Adding input layer and linking all layers
        this.setInputLayer(layers{1});
        layers{1}.setPosition(1);
        layers{1}.setNetwork(this);
        for l = 2:this.depth
          layers{l - 1}.setNext(layers{l});
          layers{l}.setPrevious(layers{l - 1});
          layers{l}.setPosition(l);
          layers{l}.setNetwork(this);
        end
        this.outputLayer = layers{this.depth};
      else
        this.depth = 0;
      end
    end
    
    function setOptimizer(this, optimizer)
      %setOptimizer Sets the optmizer for the network
      %   Sets the optimizer instance which is used to update weights and
      %   biases of the network during training.
      % Inputs:
      %   - optimizer: the optimizer instance
      assert(isa(optimizer, 'neural.optimizers.Optimizer'), ...
        'NeuralNetwork:invalidOptimizer', 'Invalid optimizer type: %s', ...
        class(optimizer));
      this.optimizer = optimizer;
    end
    
    function addLayer(this, layer, position)
      %addLayer Adds a layer to the net
      %   Appends a layer to the network or adds a layer at the given
      %   position.
      %
      %   Inputs:
      %     - layer: the layer which has to be added
      %     - position: if specified, adds the layer in this position, at
      %       the end of the network if it is greater than current depth
      %       or at the beginning if it is 0. If not specified, the layer
      %       will be appended at the end of the network
      assert(isa(layer, 'neural.layers.Layer'), 'NeuralNetwork:invalidLayer', ...
        'Invalid type for layer: %s', class(layer));
      if nargin > 2
        assert(position >= 0 && position <= this.depth + 1, ...
          'NeuralNetwork:invalidPosition', ...
          'Invalid position for layer: %d', position);
      else
        position = this.depth + 1;
      end
      % Inserting between the list
      if this.depth == 0  % Empty network
        this.setInputLayer(layer);
        this.outputLayer = layer;
        layer.setPosition(1);
      elseif position <= this.depth  % Insert between layers
        if position == 0  % New first layer
          layer.setNext(this.inputLayer.next);
          this.setInputLayer(layer);
          layer.setPosition(1);
        else
          searched = this.getLayerAt(position);
          searched.previous.setNext(layer);
          layer.setPrevious(searched.previous);
          layer.setNext(searched);
          searched.setPrevious(layer);
          layer.setPosition(position);
        end
        this.incrementPosition(layer.next);
      else  % Appending at the end
        if this.depth > 0
          this.outputLayer.setNext(layer);
        end
        layer.setPrevious(this.outputLayer);
        this.outputLayer = layer;
        layer.setPosition(position);
      end
      this.depth = this.depth + 1;
    end
    
    function layer = getLayerAt(this, position)
      %getLayerAt Gets the net's layer at given position
      %   Returns the layer of the net in the given position. Input layer
      %   is at position 0 for this method.
      % Inputs:
      %   - position: the position of searched layer
      % Output:
      %   - layer: the layer in the network which occupies input position
      assert(position >= 0 && position <= this.depth, ... 
        'NeuralNetwork:invalidPosition', 'No such layer: %d', position);
      layer = this.inputLayer;
      i = 0;
      while i < position
        assert(~isempty(layer), 'NeuralNetwork:noSuchLayer',...
          'No layer in position %d', position);
        layer = layer.next;
        i = i + 1;
      end
    end
    
    function removeLayer(this, position)
      %removeLayer Removes a layer at given position
      %   Removes a layer at the specified position. If the position is
      %   greater than depth or lesser than 1, nothing will happen.
      % Inputs:
      %   - position: the position the layer will occupy in the net
      assert(position > 0 && position <= this.depth,...
        'NeuralNetwork:invalidPosition', 'Invalid layer position: %d',...
        position);
      if position == 1  % Removing first layer
        if this.depth == 1  % Only one layer
          this.inputLayer.setNext([]);
          this.outputLayer.setPrevious([]);
          this.inputLayer = [];
          this.outputLayer = [];
        else
          first = this.inputLayer.next;
          this.setInputLayer(first.next);
          this.decrementPosition(first.next);
          first.setPrevious([]);
          first.setNext([]);
        end
      else
        searched = this.getLayerAt(position);
        searched.previous.setNext(searched.next);
        if position == this.depth  % Last layer
          this.outputLayer = searched.previous;
        else
          searched.next.setPrevious(searched.previous);
          this.decrementPosition(searched.next);
        end
        searched.setPrevious([]);
        searched.setNext([]);
      end
      this.depth = this.depth - 1;
    end
    
    function validate(this)
      %validate Validates the network, assuring coherence between layers
      %   Checks all layers' input and output size in order to assure that
      %   these values are consistent between layers.
      if ~isempty(this.inputLayer)
        layer = this.inputLayer.next;
        while ~isempty(layer)
          assert(all(layer.inputSize == layer.previous.outputSize), ...
            'NeuralNetwork:invalidNetwork', ['Invalid shapes for ', ...
            'layers %d/%d'], layer.previous.position, layer.position);
          layer = layer.next;
        end
      end
    end
    
    function Y = predict(this, X)
      %predict Evaluates the network on given input
      %   Propagates forward input data in order to get net's output.
      % Inputs:
      %   - X: network input
      % Output:
      %   - Y: output of the network
      Y = this.inputLayer.predict(X);
    end
    
    function Y = forward(this, X)
      %forward Evaluates the network on given input with cache
      %   Propagates forward input data in order to get net's output and
      %   caches intermediate values for each layer in order to train.
      % Inputs:
      %   - X: network input
      % Output:
      %   - Y: output of the network
      Y = this.inputLayer.forward(X);
    end
    
    function backward(this, Y, T, errorFunction)
      %backward Backward propagates on the net and updates weights
      %   Performs backpropagation on the net's layers and updates weights
      %   using derivatives w.r.t. the given error function.
      % Inputs:
      %   - Y: network's out
      %   - T: true values of network's labels
      %   - errorFunction: the error function used to evaluate derivatives
      % Outputs:
      %   - errors: evaluation of error function on given input
      dE = errorFunction.derive(Y, T);
      this.outputLayer.backward(dE);
    end
    
    function print(this)
      %print Shows the network layers' name
      %   Prints the name of layers in the network.
      layer = this.inputLayer;
      while ~isempty(layer)
        fprintf(1, '%d: %s -> ', layer.position, layer.name);
        layer = layer.next;
      end
      fprintf(1, 'Y\n');
    end
  end
  
  methods(Access=private)
    function setInputLayer(this, layer)
      %setInputLayer Sets or updates net's input layer when needed
      % Inputs:
      %   - layer: the layer which is connected to this one
      if isempty(this.inputLayer)
        this.inputLayer = neural.layers.InputLayer(layer.inputSize, 'X');
        this.inputLayer.setPosition(0);
        this.inputLayer.setNetwork(this);
      else
        this.inputLayer.setInputSize(layer.inputSize);
      end
      this.inputLayer.setNext(layer);
      layer.setPrevious(this.inputLayer);
    end
    
    function incrementPosition(~, layer)
      %incrementPosition Increment position of all layers starting from the
      %given one.
      while ~isempty(layer)
        layer.setPosition(layer.position + 1);
        layer = layer.next;
      end
    end
    
    function decrementPosition(~, layer)
      %decrementPosition Decrements position of all layers starting from
      %the given one.
      while ~isempty(layer)
        layer.setPosition(layer.position - 1);
        layer = layer.next;
      end
    end
  end
end

