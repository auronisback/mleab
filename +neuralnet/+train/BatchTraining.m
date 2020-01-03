%Author: Francesco Altiero
%Date: 10/12/2018

classdef BatchTraining < neuralnet.train.AbstractTraining
  %BATCHTRAINING Batch training method
  %   Manages the training of the net using batch method: each weight
  %   update is performed after the back propagation on the whole dataset.
  
  methods
    function this = BatchTraining(epochs, trainErrorFunction, ...
        weightUpdateStrategy, stopCriterion, splitter, errorFunction)
      %BatchTraining Creates a BatchTraining object
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
      
      %Preparing parameters
      if ~exist('stopCriterion', 'var')
        stopCriterion = [];
      end
      if ~exist('splitter', 'var')
        splitter = [];
      end
      if ~exist('errorFunction', 'var')
        errorFunction = [];
      end
      %Calling constructor
      this@neuralnet.train.AbstractTraining(epochs, trainErrorFunction, ...
        weightUpdateStrategy, stopCriterion, splitter, errorFunction);
    end
    
    function errors = train(this, net, dataset)
      %train Train the given network on given dataset using batch learning
      %   This method trains the net on the given dataset using the batch
      %   method, that updates weights and biases of the net only after
      %   performed back-propagation on all training set patterns.
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
      
      %Initializing error structure
      errors = struct('training', zeros(1, this.epochs), ...
        'validation', zeros(1, this.epochs), ...
        'bestEpoch', 0);
      %Handle for minimum error on validation
      minValErr = Inf();
      %Splitting the dataset
      this.splitter.split(dataset);
      for e = 1:this.epochs
        %Getting training and validation sets
        ts = this.splitter.getTrainingSet(this);
        vs = this.splitter.getValidationSet(this);
        %Performing back propagation
        [derW, derB] = net.backward(ts.patterns, ts.labels, this.trainErrFun);
        this.weightUpdate.update(net, derW, derB);
        %Calculating error on training and validation sets
        errors.training(e) = this.errFun.eval(net.forward(ts.patterns), ts.labels);
        errors.validation(e) = this.errFun.eval(net.forward(vs.patterns), vs.labels);
        %Checking if error on validation is lesser than current minimum
        if errors.validation(e) < minValErr
          minValErr = errors.validation(e); %New minimum
          [bestW, bestB] = net.exportWeightsAndBiases(); %Saving weights
          errors.bestEpoch = e; %Saving best epoch
        end
        %If the stop criterion has been met, training ends
        if this.stopCriterion.isMet, break, end
      end
      %End of the training: setting best weights
      net.importWeightsAndBiases(bestW, bestB);
    end
  end
end

