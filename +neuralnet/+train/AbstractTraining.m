%Author: Francesco Altiero
%Date: 10/12/2018

classdef AbstractTraining < handle
  %ABSTRACTTRAINING Base class for training methods
  %   This class defines properties and methods in order to train a neural
  %   network. It stores the number of epochs of training with an optional
  %   stop criterion. Defines a method that is used, given dataset
  %   with its labels, to train the network. In order to use a validation
  %   set from the dataset, it is possible to specify a splitter object
  %   used to manage the training/validation set division of the whole
  %   dataset.
  
  properties (Constant)
    DEFAULT_SPLITTER_FACTOR = 0.25; %Factor for default splitter
  end
  
  properties
    epochs %Number of maxium epochs of training
    trainErrFun %Error function on the training set (this will be derived)
    weightUpdate %Strategy used to update weights
    stopCriterion %Criterion for early stopping of the training
    splitter %Dataset splitter used to divede the dataset in test/validation
    errFun %Error function used to evaluate errors
  end
  
  methods
    function this = AbstractTraining(epochs, trainErrorFunction, ...
        weightUpdateStrategy, stopCriterion, splitter, errorFunction)
      %ABSTRACTTRAINING Initializes parameters of the base class
      %   Creates the object giving common parameters, such as the number
      %   of epochs, the criterion for early stopping and the dataset
      %   splitter.
      %
      %   Inputs:
      %     - epochs: the maximum number of training epochs if the stop
      %         criterion is not met
      %     - trainErrorFunction: function used during training in order to
      %         evaluate derivatives for back propagation
      %     - weightUpdateStrategy: strategy used to update weights in the
      %         net, that is, to calculate delta for weights and biases for
      %         all net's layers
      %     - stopCriterion: optional, the StopCriterion object used to
      %         check if the learning has to be stopped earlier. If not
      %         given, the train will end when all epochs are elapsed
      %     - splitter: optional, the DatasetSplitter object that manages
      %         division of the dataset into training/validation set. If no
      %         splitter is given, a simple FactorSplitter object will be
      %         used with factor 1/4
      %     - errorFunction: optional, this is the error function used to
      %         evaluate net's performance at each epoch. If none given,
      %         given training error function will be used
      
      %Checking epochs
      assert(epochs > 0, 'AbstractTraining:invalidEpochs',...
        sprintf('Invalid epochs: %d', epochs));
      %Checking error function in training
      assert(isa(trainErrorFunction, 'neuralnet.train.error.ErrorFunction'), ...
        'AbstractTraining:invalidTrainErrorFunction', ...
        'Given training error function is not a valid object');
      %Checking weight update strategy
      assert(isa(weightUpdateStrategy, ...
          'neuralnet.train.update.WeightUpdateStrategy'), ...
        'AbstractTraining:invalidWeightUpdateStrategy', ...
        'Given weight update strategy is not a valid object');
      if ~isempty(stopCriterion) %Stop criterion given
        %Checking is of the right type
        assert(isa(stopCriterion, 'neuralnet.train.criteria.StopCriterion'), ...
          'AbstractTraining:invalidStopCriterion', ...
          'Stop criterion is invalid');
      else
        %Initializing a non-stop criterion
        stopCriterion = neuralnet.train.criteria.NonStopCriterion();
      end
      if ~isempty(splitter) %Splitter given
        assert(isa(splitter, 'neuralnet.train.splitter.DatasetSplitter'), ...
          'AbstractTraining:invalidSplitter', 'Splitter is invalid');
      else
        %Initializing the default factor splitter
        splitter = neuralnet.train.splitter.FactorSplitter(...
          neuralnet.train.AbstractTraining.DEFAULT_SPLITTER_FACTOR);
      end
      %Checking error function
      if ~isempty(errorFunction) %Given, checking it is valid
        assert(isa(errorFunction, 'neuralnet.train.error.ErrorFunction'), ...
          'AbstractTraining:invalidErrorFunction', ...
          'Given error function for evaluation is invalid');
      else %Not given, setting the error as training error
        errorFunction = trainErrorFunction;
      end
      %Ok, storing parameters
      this.epochs = epochs;
      this.trainErrFun = trainErrorFunction;
      this.weightUpdate = weightUpdateStrategy;
      this.stopCriterion = stopCriterion;
      this.splitter = splitter;
      this.errFun = errorFunction;
    end
  end
  
  methods (Abstract)
    errors = train(this, net, dataset);
      %train Train the given network on given dataset
      %   This method trains the net on the given dataset, returning the
      %   error on the training set and on the validation set.
      %
      %   Inputs:
      %     - net: the NeuralNet object that has to be trained
      %     - dataset: the Dataset object representing the dataset
      %   Outputs:
      %     - errors: a structure representing errors during training. It
      %         has fields:
      %         + training: error array (per-epoch) on training set
      %         + validation: error array (per-epoch) on validation set
      %         + bestEpoch: epoch of the minumum error on validation
  end
end

