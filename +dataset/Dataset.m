classdef Dataset < handle
  %DATASET Defines object used to store datasets
  %   Manages operation on datasets.
  
  properties(Access=private)
    label_names;  % Descriptive name for labels
    train_images;  % Set of training images
    train_labels;  % Set of training labels
    test_images;  % Set of test images
    test_labels;  % Set of test labels
    inputSize;  % Size of input
  end
  
  methods
    function this = Dataset(train_images, train_labels, ...
        test_images, test_labels, label_names)
      %DATASET Construct the dataset
      % Creates the dataset specifying its data.
      % Inputs:
      %   - train_images: the set of training images, where different
      %       images are spread along the first dimension
      %   - train_labels: an array of numeric labels for training set
      %   - test_images: the set of test images, structured as the
      %       train_images
      %   - test_labels: an array of numeric labels for test set
      %   - label_names: an array with labels descriptive names
      this.train_images = train_images;
      this.train_labels = train_labels;
      this.test_images = test_images;
      this.test_labels = test_labels;
      this.label_names = label_names;
      Xsize = size(train_images);
      this.inputSize = squeeze(Xsize(2:end));
    end
    
    function num = getTrainingN(this)
      %getTrainingN Gets the number of elements in the training set.
      % Output:
      %   - num: The number of elements in the training set
      num = size(this.train_images, 1);
    end
    
    function num = getTestN(this)
      %getTestN Gets the number of elements in the test set.
      % Output:
      %   - num: The number of elements in the test set
      num = size(this.test_images(), 1);
    end
    
    
    function [images, labels] = getTrainingSet(this, from, to)
      %getTrainingSet Gets the training set
      %   Retrieves the training set in the dataset specifying the samples
      %   which can be retrieved. If omitted, the whole training set is
      %   returned.
      % Inputs:
      %   - from: starting index from which extract training samples
      %   - to: ending index up to which extract training samples
      % Outputs:
      %   - images: the images in the training set
      %   - labels: numeric labels for training set
      if nargin < 3
        to = this.getTrainingN();
      end
      if nargin < 2
        from = 1;
      end
      dims = size(this.train_images);  % Caching actual size (if flattened)
      images = reshape(this.train_images(from:to, :), ...
        [to - from + 1, dims(2:end)]);
      labels = this.train_labels(from:to);
    end
    
    function [images, labels] = getTestSet(this, from, to)
      %getTestSet Gets the test set
      %   Retrieves the test set in the dataset, specifying
      % Inputs:
      %   - from: the first sample which is retrieved
      %   - to: the last samples which is retrieved
      % Outputs:
      %   - images: the images in the test set
      %   - labels: numeric labels for test set
      if nargin < 3
        to = this.getTestN();
      end
      if nargin < 2
        from = 1;
      end
      dims = size(this.test_images);  % Caching actual size (if flattened)
      images = reshape(this.test_images(from:to, :), ...
        [to - from + 1, dims(2:end)]);
      labels = this.test_labels(from:to);
    end
    
    function label_names = getLabelNames(this)
      %getLabelNames Retrieves the descriptive labels related to the
      %   dataset images.
      % Outputs:
      %   - label_names: An array of strings containing descriptive
      %       labels
      label_names = this.label_names;
    end
    
    function dim = getDimensions(this)
      %getDimensions Gets the dimensions of training and test elements.
      % Output:
      %   - dim: an array with the sizes of elements in the dataset
      dim = this.inputSize;
    end
    
    function shuffle(this)
      %shuffle Shuffles the training set
      %   Applies a random permutation to images and labels in the
      %   training set.
      perm = randperm(this.getTrainingN()); %Creating a permutation
      %Permutating training images and labels
      this.train_images = this.train_images(perm, :, :, :);
      this.train_labels = this.train_labels(perm, :, :, :);
    end
    
    function flatten(this)
      %flatten Flattens training and test samples shape
      %   Converts training and test samples into a linear vector.
      this.train_images = squeeze(reshape(this.train_images, ...
        [this.getTrainingN(), prod(this.getDimensions(), 'all')]));
      this.test_images = squeeze(reshape(this.test_images, ...
        [this.getTestN(), prod(this.getDimensions(), 'all')]));
      % Updating input size
      this.inputSize = prod(this.inputSize, 'all');
    end
    
    function normalize(this)
      %normalize Normalizes the dataset
      %   Normalizes the dataset in order to its samples have values
      %   between zero and one.
      this.train_images = this.train_images ./ 255;
      this.test_images = this.test_images ./ 255;
    end
  end
end

