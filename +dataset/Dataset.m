%Author: Francesco Altiero
%Date: 07/12/2018

classdef Dataset < handle
  %DATASET Defines dataset objects
  %   This class manages datasets of training or test data used in neural
  %   network training/validation/testing.
  
  properties (SetAccess = private)
    num %Number of patterns/labels in the dataset
    patterns %Patterns in the dataset
    labels %Labels for each pattern in the dataset
    patternSizes %Array with size for each dimension of pattern data
    labelSizes %Array with size for each dimension of labels
  end
  
  methods
    function this = Dataset(patternSizes, labelSizes)
      %DATASET Construct an instance of this class
      %   Creates a dataset object specifying dimensions and sizes of
      %   patterns, along with dimension of labels.
      %   Inputs:
      %     - patternSizes: an array whose elements are the sizes of each
      %         pattern dimension. The number of dimensions of patterns is
      %         given by the length of this array
      %     - labelSizes: an array in which each value represents the size
      %         of the respective dimension in labels. Total dimensions of
      %         labels is given by the array's length
      %   Output:
      %     - this: the Dataset object created, with empty patterns
      
      %Checking patternSizes and labelSizes
      this.checkSizes(patternSizes);
      this.checkSizes(labelSizes);
      %Ok, initializing properties
      this.patternSizes = patternSizes;
      this.labelSizes = labelSizes;
    end

    function setPatternsAndLabels(this, patterns, labels)
      %setPatternsAndLabels Adds patterns and labels to the dataset
      %   Sets the patterns and the labels in the dataset, checking
      %   dimensions.
      %   
      %   Inputs:
      %     - patterns: patterns in the dataset
      %     - labels: labels in the dataset
      
      %Asserting labels and patterns have the same cardinality
      assert(size(patterns, 1) == size(labels, 1), 'Dataset:invalidSize', ...
        sprintf('Patterns and labels have different cardinality: %d vs %d', ...
          size(patterns, 1), size(labels, 1)));
      %Ok, setting
      this.num = size(patterns, 1);
      this.patterns = patterns;
      this.labels = labels;
    end

    function shuffle(this)
      %shuffle Shuffles the dataset
      %   Performs a shuffle of the dataset.
      
      %Creating a permutation
      perm = randperm(this.num);
      %Permutating patterns and labels
      this.patterns = this.patterns(perm, :);
      this.labels = this.labels(perm, :);
    end
  end
  
  methods(Access=private)
    function checkSizes(~, sizes)
      % checkSizes Checks for validity of pattern or label size array. A
      %   size array is valid if its length is greater than 0 and all its
      %   dimensions are positive integers. It will raise an exception if
      %   any size is invalid.
      %
      %   Inputs:
      %     - sizes: a size array
      
      %Checking length
      assert(isvector(sizes), 'Dataset:invalidSize', ...
          'Given size is not a 1xn or nx1 matrix');
      %Checking each value
      assert(all(sizes(:) > 0), 'Dataset:invalidDimensions', ...
          'Given sizes has 1 or more non-positive values');
    end
  end
end

