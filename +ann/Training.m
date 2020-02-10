classdef Training < handle
  %TRAINING Manages training of a neural network
  %   Defines all properties and methods used to train a neural network,
  %   as optimizers, size of a training batch and factor of dataset
  %   splitting in training and validation set.
  
  properties(SetAccess = private)
    optimizer;  % Optimizer used to tune parameters
    batchSize;  % Size of a batch before parameters' update
    validationSplit;  % Validation split factor
  end
  
  methods
    function this = Training(optimizer, batchSize, validationSplit)
      %Training Creates a new training object
      %   Constructs an objec used to train neural networks, specifying
      %   training parameters as the optimizer, the size of batch before
      %   updating network's parameters and the split factor to divide
      %   dataset in training and validation set.
      % Inputs:
      %   - optimizer: the optimizer instance used to tune parameters
      %   - batchSize: size of a batch
      %   - validationSplit: a number in [0, 1[ used to split the dataset
      %     in training and validation
      assert(isa(optimizer, 'ann.optimizers.Optimizer'), ...
        'Training:invalidOptimizer', 'Invalid optimizer type: %s', ...
        class(optimizer));
      assert(0 < batchSize && floor(batchSize) == batchSize, ...
        'Training:invalidBatchSize', 'Invalid batch size');
      assert(isscalar(validationSplit) && 0 <= validationSplit && ...
        validationSplit < 1, 'Training:invalidSplit', ...
        'Invalid validation split factor');
      this.optimizer = optimizer;
      this.batchSize = batchSize;
      this.validationSplit = validationSplit;
      % Preventively clearing optimizer
      this.optimizer.clear();
    end
    
    function [errors, bestEpoch, elapsed] = train(this, epochs, network, dataset)
      %train Trains the network
      %   Performs the training of a neural network on the given dataset
      %   and for the given number of epochs of training.
      % Inputs:
      %   - epochs: number of epochs of training
      %   - network: the neural network which has to be trained
      %   - dataset: the dataset on which the network should be trained
      % Output:
      %   - errors: a cell-array with 4 cells, which are error function
      %     values on training set, error function values on validation
      %     set, accuracy on training set and accuracy on validation. Each
      %     cell is an array of epochs plus one elements, which record
      %     various values starting from epoch 0 (before the training) to
      %     the last training epoch
      %   - bestEpoch: the epoch in which best accuracy on validation set
      %     was reached
      %   - elapsed: an array of elapsed time per epoch
      
      % Initializing errors and accuracy
      errors = cell(1, 4);
      errors{1} = zeros(1, epochs + 1);
      errors{2} = zeros(1, epochs + 1);
      errors{3} = zeros(1, epochs + 1);
      errors{4} = zeros(1, epochs + 1);
      % Initializing elapsed times
      elapsed = zeros(1, epochs);
      % Splitting dataset
      [tX, tT, vX, vT] = this.splitDataset(dataset);
      % Getting error for untrained network (epoch: 0)
      [errors{1}(1), errors{2}(1), errors{3}(1), errors{4}(1)] = ...
        this.getErrorsAndAccuracy(network, tX, tT, vX, vT);
      fprintf('epoch 0: err: %.4f - val_err: %.4f ', errors{1}(1), errors{2}(1));
        fprintf('- acc: %.2f - val_acc: %.2f\n', ...
          errors{3}(1) * 100, errors{4}(1) * 100);
      % Initializing best weights and epoch
      [bestW, bestB] = network.getParameters();
      bestEpoch = 0;
      bestAccuracy = errors{4}(1);
      % Getting size and number of batches
      numBatches = ceil(double(size(tX, 1)) / double(this.batchSize));
      % Training for given number of epochs
      for e = 1:epochs
        fprintf('epoch %d: ', e);
        tic;  % Starting time recording
        % Training on each batch
        for b = 1:numBatches
          [bX, bT] = this.getBatch(tX, tT, b);
          this.makeTrainingStep(network, bX, bT);
        end
        elapsed(e) = toc;  % Getting elapsed time for the epoch
        % Evaluating error and accuracy
        [errors{1}(e + 1), errors{2}(e + 1), errors{3}(e + 1), ...
          errors{4}(e + 1)] = this.getErrorsAndAccuracy(network, tX, tT, vX, vT);
        fprintf('err: %.4f - val_err: %.4f ', errors{1}(e + 1), errors{2}(e + 1));
        fprintf('- acc: %.2f - val_acc: %.2f - elapsed: %d:%06.3fs\n', ...
          errors{3}(e + 1) * 100, errors{4}(e + 1) * 100, ...
          floor(elapsed(e) / 60), rem(elapsed(e), 60));
        % Saving best weights if validation error was reduced
        if errors{4}(e + 1) > bestAccuracy
          [bestW, bestB] = network.getParameters();
          bestEpoch = e + 1;
          bestAccuracy = errors{4}(e + 1);
        end
      end
      % Cleaning up optimizer
      this.optimizer.clear();
      % Resetting best parameters of the network
      network.setParameters(bestW, bestB);
      % Printing total elapsed time
      totalTime = sum(elapsed);
      fprintf('Total training time: %d:%06.3fs\n', floor(totalTime / 60), ...
        rem(totalTime, 60));
    end
    
    function [err, acc] = evaluateOnTestSet(this, network, dataset)
      %evaluateOnTestSet Evaluates the network on the test set
      %   Evaluates error and accuracy on the test set.
      % Inputs:
      %   - network: the neural network which has to be evaluated
      %   - dataset: the dataset from which extract test set
      % Outputs:
      %   - err: training error on the test set
      %   - acc: accuracy on the test set
      [X, T] = dataset.getTestSet();
      Y = network.predict(X);
      err = network.errorFun.eval(Y, T) ./ size(Y, 1);
      acc = this.evalAccuracy(Y, T);
    end
  end
  
  methods(Access = private)
    function [tX, tT, vX, vT] = splitDataset(this, dataset)
      %splitDataset Splits the dataset in training and validation set,
      %using given validation split factor
      N = dataset.getTrainingN();
      [X, T] = dataset.getTrainingSet();
      % Calculating number of elements in training and validation
      valN = floor(N * this.validationSplit);
      trainN = N - valN;
      fprintf('TrainN: %d\n', trainN);
      % Creating training set
      tX = reshape(X(1:trainN, :), [trainN, dataset.inputShape]);
      tT = reshape(T(1:trainN, :), [trainN, dataset.labelShape]);
      % Creating validation set only if it has at least a sample
      if valN > 0  
        vX = reshape(X(trainN + 1:end, :), [valN, dataset.inputShape]);
        vT = reshape(T(trainN + 1:end, :), [valN, dataset.labelShape]);
      else
        % Empty validation set
        vX = [];
        vT = [];
      end
    end
    
    function [trainError, valError, trainAcc, valAcc] = ...
        getErrorsAndAccuracy(this, network, tX, tT, vX, vT)
      %getErrorsAndAccuracy Evaluates error and accuracy of the network on
      %both training and validation set. Errors will be divided for the
      %number of elements in the batch, in order to have comparable results
      %both on training and validation sets.
      
      % Evaluating error and accuracy on training set
      tY = network.predict(tX);
      trainError = network.errorFun.eval(tY, tT) ./ size(tX, 1);
      trainAcc = this.evalAccuracy(tY, tT);
      % Evaluating error and accuracy on validation, if not empty
      if ~isempty(vX)
        vY = network.predict(vX);
        valError = network.errorFun.eval(vY, vT) ./ size(vX, 1);
        valAcc = this.evalAccuracy(vY, vT);
      else
        % Validation set empty: setting error to 0
        valError = 0;
        valAcc = 0;
      end
    end
    
    function makeTrainingStep(this, network, X, T)
      %makeTrainingStep Performs a forward and backward pass, updating
      %network's parameters, on the given batch and batch's labels.
      
      % Forwarding and backpropagating
      network.forward(X);
      [dW, db] = network.backpropagate(X, T);
      % Getting deltas and updating network's parameters
      [deltaW, deltaB] = this.optimizer.evaluateDeltas(dW, db, size(X, 1));
      network.updateParameters(deltaW, deltaB);
    end
    
    function [X, T] = getBatch(this, X, T, b)
      %getBatchs Gets the b-th batch of samples and labels from training 
      %set.
      Xshape = size(X);  % Caching input's size
      Xshape = Xshape(2:end);  % Extracting shape from size
      % Calculating batch starting and ending point
      from = (b - 1) * this.batchSize + 1;
      to = b * this.batchSize;
      % Adjusting indexes if it is the last batch
      if to > size(X, 1)
        to = from + mod(size(X, 1), this.batchSize) - 1;
      end
      % Returning the batch, reshaping samples to their size
      X = reshape(X(from:to, :), [to - from + 1, Xshape]);
      T = T(from:to, :);
    end
    
    function acc = evalAccuracy(~, Y, T)
      %evalAccuracy Evaluates accuracy between outputs and labels.
      %   Evaluates the accuracy of the network, both for numerical labels
      %   and for categorical labels.
      if(size(Y, 2) == 1)  % Numerical labels
        acc = sum(round(Y) == T);  % # of right values...
      else  % Categorical labels
        [~, Yout] = max(Y, [], 2);
        [~, Tout] = max(T, [], 2);
        acc = sum(Yout == Tout);
      end
      acc = acc ./ size(Y, 1);  % ...divided by total number of values
    end
  end
end

