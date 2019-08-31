%Author: Francesco Altiero
%Date: 26/12/2018

classdef MnistClassifier
  %MNISTCLASSIFIER Classificator for MNIST dataset
  %   This class manages the classification of the MNIST dataset. It uses a
  %   neural net in order to provide response on MNIST images given.
  
  properties (Constant)
    CLASSES_NUM = 10 %Number of classes for the MNIST dataset
  end
  
  properties
    net %The Neural Net used to classify data
  end
  
  methods
    function this = MnistClassifier(net)
      %MNISTCLASSIFIER Creates a MNIST dataset classifier
      %   Constructor for the class. It creates a new classifier with the
      %   given neural network and uses it to classfy MNIST data.
      %   
      %   Inputs:
      %     - net: the NeuralNet object used in classification. The net
      %         must have 10 output values, representing class membership.
      
      %Asserting the output layer has the right output dimensionality
      assert(net.outputLayer.outputDim == mnist.MnistClassifier.CLASSES_NUM, ...
        'MnistClassifier:invalidOutputDimension', ...
        sprintf("Net\'s output dimensionality is invalid: %d", net.outputLayer.outputDim));
      this.net = net;
    end
    
    function scores = classify(this, entry)
      %classify Classifies a MNIST entry returning class scores
      %   Uses the neural network in order to classify the net's output on
      %   the MNIST entry as input.
      %
      %   Inputs:
      %     - entry: a MNIST dataset entry
      %   Outputs:
      %     - scores: a cell array with scores related to each MNIST class
      
      %Creating scores array
      scores = cell(1, mnist.MnistClassifier.CLASSES_NUM);
      %Getting net's output
      y = this.net.forward(entry);
      %Sorting outputs in descending order
      [val, idx] = sort(y, 'descend');
      %Inserting data in the cell array
      for c = 1:mnist.MnistClassifier.CLASSES_NUM
        scores{c} = struct('label', idx(c) - 1, 'probability', val(c));
      end
    end
  end
  
end

