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
    testError;  % Error function used to benchmark
  end
  
  methods
    function this = Training(epochs, batchSize, validationSplit, ...
      optimizer, trainError, testError)
      %TRAINING Construct a training object
      %   Creates the object, initializing its properies.
      % Inputs:
      %   - epochs: number of training epochs
      %   - batchSize: size of a batch
      %   - validationSplit: a scalar in [0, 1[ which is used to split
      %     datasets in training and validation
      %   - optimizer: an optimizer instance used to update parameters
      %   - trainError: error used to train the net
      %   - testError: error used to benchmark the net. If not given, train
      %     error will be used
      this.setEpochs(epochs);
      this.setBatchSize(batchSize);
      this.setValidationSplit(validationSplit);
      this.setOptimizer(optimizer);
      this.setTrainingError(trainError);
      if nargin < 7
        testError = trainError;
      end
      this.setTestError(testError);
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
    
    function setTestError(this, testError)
      %setTestError Sets the error function used to evaluate test set
      %   Sets the test error function, which is used to evaluate network's
      %   performances on the test set.
      % Inputs:
      %   - testError: the benchmark error function
      assert(isa(testError, 'neural.errors.ErrorFunction'), ...
        'Training:invalidTestError', 'Invalid error type: %s', ...
        class(testError));
      this.testError = testError;
    end
    
    function [trainErrors, validationErrors] = train(this, ...
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
      % Pre-allocating error vectors for speed
      trainErrors = zeros([1, this.epochs]);
      validationErrors = zeros([1, this.epochs]);
      % Splitting dataset
      [tX, tT, vX, vT] = this.splitDataset(dataset);
      % Training for given epochs
      for e = 1:this.epochs
        if verbose
          fprintf('epoch %d: ', e);
        end
        this.actualEpoch = e;  % Caching epoch
        % TODO: add mini batches
        tY = network.forward(tX);
        trErr = this.trainError.evaluate(tY, tT);
        network.backward(tY, tT, this.trainError);
        vY = network.predict(vX);
        valErr = this.trainError.evaluate(vY, vT);
        trainErrors(e) = trErr;
        validationErrors(e) = valErr;
        if verbose
          fprintf('Train error: %f - Validation error: %f\n', ...
            trErr, valErr);
        end
      end
    end
    
    function testError = evaluateOnTest(this, network, dataset)
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
      Y = network.predict(X);
      testError = this.testError.evaluate(Y, T);
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
  end
end

