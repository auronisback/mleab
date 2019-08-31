%Author: Francesco Altiero
%Date: 14/12/2018

classdef DeltaRule < neuralnet.train.update.WeightUpdateStrategy
  %DELTARULE Delta rule for weight update in the net
  %   Manage the weight updating strategy using the delta rule, that is
  %   calculated as
  %     deltaW = - eta * derW
  %   where eta is a scalar value representing the learning rate and derW 
  %   is the matrix of the derivatives of the error function with regard to
  %   weights.
  
  properties
    eta %Learning rate
  end
  
  methods
    function this = DeltaRule(eta)
      %DELTARULE Creates a delta rule weight update object
      %   Constructor for a delta rule object, with specified learning
      %   rate.
      %
      %   Inputs:
      %     - eta: a positive scalar value representing the learning rate
      assert(isscalar(eta) && eta > 0, 'DeltaRule:invalidEta', ...
        sprintf('Invalid eta: %d', eta));
      this.eta = eta;
    end
    
    function update(this, net, derW, derB)
      %update Updates net's weights and biases using the delta rule
      %   Calculates the net's weight and biases delta using
      %   the delta rule, with the learning rate specified in the creation
      %   of the object; then, updates weights and biases in the net.
      %
      %   Inputs:
      %     - net: the NeuralNet object
      %     - derW: a cell array with the derivatives of each net's layer
      %         with respect to weights
      %     - derB: a cell array with the derivatives of each net's layer
      %         with respect to biases
      
      %Checking that the cell array sizes match with net's depth
      assert(size(derW, 2) == size(derB, 2) && size(derW, 2) == net.depth + 1, ...
        'DeltaRule:invalidSize', ...
        sprintf('Derivative arrays have invalid size: %d vs %d, net: %d', ...
          size(derW, 2), size(derB, 2), net.depth + 1));
      %Updating each hidden layer
      for l = 1:net.depth
        %Updating subtracting derivatives per the learning rate
        net.hiddenLayers{l}.weights = net.hiddenLayers{l}.weights - this.eta * derW{l};
        net.hiddenLayers{l}.biases = net.hiddenLayers{l}.biases - this.eta * derB{l};
      end
      %Updating output layer
      net.outputLayer.weights = net.outputLayer.weights - this.eta * derW{net.depth + 1};
      net.outputLayer.biases = net.outputLayer.biases - this.eta * derB{net.depth + 1};
    end
  end
end

