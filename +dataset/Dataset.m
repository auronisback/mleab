classdef Dataset < handle
  %DATASET Defines object used to store datasets
  %   Manages operation on datasets.
  
  properties(SetAccess = private)
    labelNames;  % Descriptive name for labels
    trainSamples;  % Set of training images
    trainLabels;  % Set of training labels
    testSamples;  % Set of test images
    testLabels;  % Set of test labels
    inputShape;  % Size of input
    labelShape;  % Shape of labels
    numClasses;  % Number of classes
  end
  
  methods
    function this = Dataset(trainSamples, trainLabels, ...
        testSamples, testLabels, labelNames)
      %DATASET Construct the dataset
      % Creates the dataset specifying its data.
      % Inputs:
      %   - trainSamples: the set of training samples, where different
      %       images are spread along the first dimension
      %   - trainLabels: an array of numeric labels for training set
      %   - testSamples: the set of test samples, structured as the
      %       trainSamples
      %   - testLabels: an array of numeric labels for test set
      %   - labelNames: an array with labels descriptive names
      this.trainSamples = trainSamples;
      this.trainLabels = trainLabels;
      this.testSamples = testSamples;
      this.testLabels = testLabels;
      this.labelNames = labelNames;
      % Sample size inferred from samples
      Xsize = size(trainSamples);
      this.inputShape = squeeze(Xsize(2:end));
      % Label shape inferred from labels
      Lsize = size(trainLabels);
      this.labelShape = squeeze(Lsize(2:end));
      % Number of classes inferred from label names
      this.numClasses = size(labelNames, 2);      
    end
    
    function num = getTrainingN(this)
      %getTrainingN Gets the number of elements in the training set.
      % Output:
      %   - num: The number of elements in the training set
      num = size(this.trainSamples, 1);
    end
    
    function num = getTestN(this)
      %getTestN Gets the number of elements in the test set.
      % Output:
      %   - num: The number of elements in the test set
      num = size(this.testSamples, 1);
    end
    
    
    function [samples, labels] = getTrainingSet(this)
      %getTrainingSet Gets the training set
      %   Retrieves the training set in the dataset.
      % Outputs:
      %   - samples: samples in the training set
      %   - labels: numeric or categorical labels for training set
      
      samples = this.trainSamples;
      labels = this.trainLabels;
    end
    
    function [samples, labels] = getTestSet(this)
      %getTestSet Gets the test set
      %   Retrieves the test set in the dataset.
      % Outputs:
      %   - samples: the images in the test set
      %   - labels: numeric labels for test set
      samples = this.testSamples;
      labels = this.testLabels;
    end
    
    function shuffle(this)
      %shuffle Shuffles the training set
      %   Applies a random permutation to images and labels in the
      %   training set.
      perm = randperm(this.getTrainingN()); %Creating a permutation
      %Permutating training images and labels
      
      % TODO: correct in order to not mess with data - abstract dimensions
      this.trainSamples = this.trainSamples(perm, :, :, :);
      this.trainLabels = this.trainLabels(perm, :, :, :);
    end
    
    function flatten(this)
      %flatten Flattens training and test samples shape
      %   Converts training and test samples into a linear vector.
      this.trainSamples = squeeze(reshape(this.trainSamples, ...
        [this.getTrainingN(), prod(this.inputShape, 'all')]));
      this.testSamples = squeeze(reshape(this.testSamples, ...
        [this.getTestN(), prod(this.inputShape, 'all')]));
      % Updating input size
      this.inputShape = prod(this.inputShape, 'all');
    end
    
    function normalize(this)
      %normalize Normalizes the dataset
      %   Normalizes the dataset in order to its samples have values
      %   between zero and one.
      this.trainSamples = this.trainSamples ./ 255;
      this.testSamples = this.testSamples ./ 255;
    end
    
    function resize(this, targetSize)
      %resize Resizes images in the dataset
      %   Resizes training and test samples in the current dataset.
      % Inputs:
      %   - targetSize: size of images, as a 2D array with target values
      %     for rows and columns
      N = this.getTrainingN();
      trResized = zeros([N, targetSize]);
      for n = 1:N
        sample = reshape(squeeze(this.trainSamples(n, :)), this.inputShape);
        resSample = imresize(sample, targetSize);
        trResized(n, :) = resSample(:);
      end
      % Resizing test
      N = this.getTestN();
      testResized = zeros([N, targetSize]);
      for n = 1:N
        sample = reshape(squeeze(this.testSamples(n, :)), this.inputShape);
        resSample = imresize(sample, targetSize);
        testResized(n, :) = resSample(:);
      end
      % Reshaping
      this.inputShape = targetSize;
      this.trainSamples = reshape(trResized, [N, this.inputShape]);
      this.testSamples = reshape(testResized, [N, this.inputShape]);
    end
    
    function toCategoricalLabels(this)
      %toCategoricalLabels Transforms labels in categorical mode
      %   Performs a transformation of labels in order to produce
      %   categorical labels, which are labels with size equal to number of
      %   classes in the classification problem; each label has all zeros
      %   and a single '1' value on the index related to category it
      %   belongs.
      
      % Pre-allocating labels
      catTrainLabels = zeros(this.getTrainingN(), this.numClasses);
      catTestLabels = zeros(this.getTestN(), this.numClasses);
      % Setting right category
      for n = 1:this.getTrainingN()
        catTrainLabels(n, this.trainLabels(n) + 1) = 1;
      end
      for n = 1:this.getTestN()
        catTestLabels(n, this.testLabels(n) + 1) = 1;
      end
      % Updating labels and label size
      this.trainLabels = catTrainLabels;
      this.testLabels = catTestLabels;
      this.labelShape = this.numClasses;
    end
  end
end

