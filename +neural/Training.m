classdef Training < handle
  %TRAINING Class used to train a neural network
  %   Manages the training of a neural network and hyper-parameters related
  %   to a training process.
  
  properties(SetAccess = private, GetAccess = public)
    epochs;  % Epochs of training
    actualEpoch;  % Cached actual epoch of training
    batchSize;  % Size of a training batch before weights update
    validationSplit;  % Factor to split dataset in training and validation
    optimizer;  % Optimizer for network's parameters
    trainError;  % Error function used during training
    metrics;  % Cell-array of metrics used to measure performance of net
  end
  
  methods
    function this = Training(epochs, batchSize, validationSplit, ...
      optimizer, trainError, metrics)
      %TRAINING Construct a training object
      %   Creates the object, initializing its properies.
      % Inputs:
      %   - epochs: number of training epochs
      %   - batchSize: size of a batch
      %   - validationSplit: a scalar in [0, 1[ which is used to split
      %     datasets in training and validation
      %   - optimizer: an optimizer instance used to update parameters
      %   - trainError: error used to train the net
      %   - metrics: metrics used to evaluate net's performance
      this.setEpochs(epochs);
      this.setBatchSize(batchSize);
      this.setValidationSplit(validationSplit);
      this.setOptimizer(optimizer);
      this.setTrainingError(trainError);
      this.setMetrics(metrics);
      this.actualEpoch = 0;  % Actual training epochs are 0
    end
    
    function setEpochs(this, epochs)
      %setEpochs Sets the number of training epochs
      %   Sets the maximum number of training epochs.
      % Inputs:
      %   - epochs: the number of training epochs
      epochs = int32(epochs);
      assert(isscalar(epochs) && isinteger(epochs) && epochs > 0, ...
        'Training:invalidEpochs', 'Invalid epochs: %d', epochs);
      this.epochs = epochs;
    end
    
    function setBatchSize(this, batchSize)
      %setBatchSize Sets the size of batches
      %   Sets the size of a training batch.
      % Inputs:
      %   - batchSize: the size of each batch
      batchSize = int32(batchSize);
      assert(isscalar(batchSize) && isinteger(batchSize),...
        'Training:invalidBatchSize', 'Invalid batch size: %d', batchSize);
      this.batchSize = batchSize;
    end
    
    function setValidationSplit(this, validationSplit)
      %setValidationSplit Sets the validation split factor
      %   Sets the split factor used to divide dataset in training and
      %   validation set.
      % Inputs:
      %   - validationSplit: a number between 0 and 1
      assert(isscalar(validationSplit) && 0 <= validationSplit && ...
        validationSplit < 1, 'Training:invalidSplitFactor', ...
        'Invalid split factor: %f', validationSplit);
      this.validationSplit = validationSplit;
    end
    
    function setOptimizer(this, optimizer)
      %setOptimizer Sets the optimizer used during training
      %   Sets the policy for updating weights and biases of the network
      %   during the training process.
      % Inputs:
      %   - optimizer: the optimizer instance
      assert(isa(optimizer, 'neural.optimizers.Optimizer'), ...
        'Training:invalidOptimizer', 'Invalid optimizer type: %s', ...
        class(optimizer));
      this.optimizer = optimizer;
    end
    
    function setTrainingError(this, trainError)
      %setTrainingError Sets the error function used during training
      %   Sets the error function which is minimized during the training of
      %   the neural network.
      % Inputs:
      %   - trainError: the error to minimize
      assert(isa(trainError, 'neural.errors.ErrorFunction'), ...
        'Training:invalidTrainingError', 'Invalid error type: %s', ...
        class(trainError));
      this.trainError = trainError;
    end
    
    function setMetrics(this, metrics)
      %setTestError Sets metrics used to evaluate net
      %   Sets all metrics used to evaluate net's performances.
      % Inputs:
      %   - metrics: a cell-array of metric functions
      for i = 1:size(metrics, 2)
        assert(isa(metrics{i}, 'neural.metrics.Metric'), ...
          'Training:invalidMetric', 'Invalid metric (%d): %s', ...
        i, class(metrics{i}));
      end
      this.metrics = metrics;
    end
    
    function metrics = train(this, ...
        network, dataset, verbose)
      %train Trains the neural network
      %   Manages the training of a neural network with given dataset.
      %   Optimizer and error functions used are those specified in
      %   creation of this training object.
      % Inputs:
      %   - network: the network to train
      %   - dataset: the dataset on which train the network
      %   - verbose: flag indicating if logs are displayed: default false
      % Outputs:
      %   - trainErrors: errors on training set
      %   - validationErrors: errors on validation set
      if nargin < 4
        verbose = false;
      end
      % Injecting optimizer into network
      network.setOptimizer(this.optimizer);
      % Initializing error and metric reports
      metrics = this.initializeMetricsArray();
      % Splitting dataset
      [tX, tT, vX, vT] = this.splitDataset(dataset);
      % Training for given epochs
      for e = 1:this.epochs
        if verbose
          fprintf('epoch %d: ', e);
        end
        this.actualEpoch = e;  % Caching epoch
        N = size(tX, 1);
        numBatches = floor(double(N) / double(this.batchSize));
        for b = 1:numBatches
          from = (b - 1) * this.batchSize + 1;
          to = b * this.batchSize; 
          trainBatch = reshape(tX(from:to, :), ...
            [this.batchSize, dataset.getDimensions()]);
          this.makeTrainingStep(network, trainBatch, tT(from:to));
        end
        % Checking if there are residual samples
        lastBsize = N - this.batchSize * numBatches;
        if lastBsize > 0
          from = this.batchSize * numBatches + 1;
          to = N;
          trainBatch = reshape(tX(from:to, :), ...
            [lastBsize, dataset.getDimensions()]);
          this.makeTrainingStep(network, trainBatch, tT(from:to));
        end
        % Adding error and metrics
        metrics = this.addEpochErrorsAndMetrics(metrics, network, e, ...
          tX, tT, vX, vT);
        if verbose
          this.printMetricsForEpoch(metrics, e);
        end
      end
    end
    
    function metricValues = evaluateOnTest(this, network, dataset)
      %evaluateOnTest Evaluates the network on test set
      %   Performs an evaluation of the network error with respect to test
      %   set in the dataset. The error is evaluated using the test error
      %   function specified in creation of the object.
      % Inputs:
      %   - network: the network to evaluate
      %   - dataset: the dataset whose test set is used for evaluation
      % Outputs:
      %   - testError: the error on test set
      [X, T] = dataset.getTestSet();
      Y = round(network.predict(X));  % Rounding to next integer
      metricValues = cell(size(this.metrics));
      for i = 1:size(this.metrics, 2)
        metricValues{i} = {...
            this.metrics{i}.name, ...
            this.metrics{i}.evaluate(Y, T)
          };
      end
    end
  end
  
  methods(Access = private)
    function [trainSamples, trainLabels, valSamples, valLabels] = ...
        splitDataset(this, dataset)
      %splitDataset Splits the dataset in training and validation set
      valN = floor(dataset.getTrainingN() * this.validationSplit);
      trainN = dataset.getTrainingN() - valN;
      [trainSamples, trainLabels] = dataset.getTrainingSet(1, trainN);
      [valSamples, valLabels] = dataset.getTrainingSet(trainN + 1);
    end
    
    function metrics = initializeMetricsArray(this)
      %initializeMetricsArray Initializes metrics used to evaluate training
      
      % Initializing a cell for each metric, both for training and
      % validation, and adding the error function values
      metrics = cell(1, size(this.metrics, 2) * 2 + 2);
      % Initializing error values metrics
      metrics{1} = {this.trainError.name, zeros(1, this.epochs)};
      metrics{2} = {['Val_', this.trainError.name], zeros(1, this.epochs)};
      % Initializing all other metrics
      for i = 1:size(this.metrics, 2)
        metrics{i*2 + 1} = {this.metrics{i}.name, zeros(1, this.epochs)};
        metrics{i*2 + 2} = {['Val_', this.metrics{i}.name], ...
          zeros(1, this.epochs)};
      end
    end
    
    function makeTrainingStep(this, network, tX, tT)
      %makeTrainStep Makes a single training step on a batch
      tY = network.forward(tX);
      network.backward(tY, tT, this.trainError);
    end
    
    function metrs = addEpochErrorsAndMetrics(this, metrics, network, e, ...
        tX, tT, vX, vT)
      %evaluateErrorAndMetrics Evaluates error and metrics for the epoch
      %   Updates error and metric values for the given epoch, adding to
      %   the traning report.
      
      % Error values
      tY = network.predict(tX);
      vY = network.predict(vX);
      metrics{1}{2}(e) = this.trainError.evaluate(tY, tT);  % On training
      metrics{2}{2}(e) = this.trainError.evaluate(vY, vT);  % On validation
      % Other metrics
      for i = 1:size(this.metrics, 2)
        metrics{i*2+1}{2}(e) = this.metrics{i}.evaluate(tY, tT);  % On training
        metrics{i*2+2}{2}(e) = this.metrics{i}.evaluate(vY, vT);  % On validation
      end
      metrs = metrics;
    end
    
    function printMetricsForEpoch(~, metrics, e)
      %printMetrics Prints the metrics for the given epoch
      for i = 1:size(metrics, 2)
        fprintf("%s: %.3f ", metrics{i}{1}, metrics{i}{2}(e));
      end
      fprintf("\n");
    end
  end
end

