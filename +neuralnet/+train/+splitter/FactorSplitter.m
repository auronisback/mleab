classdef FactorSplitter < neuralnet.train.splitter.DatasetSplitter
  %FACTORSPLITTER Splits the dataset givin a factor of total for the
  %validation set
  %   Manages the proportional splitting of the training set, specifying
  %   which factor of the total has to be put in the validation set. If the
  %   dataset has N elements and a factor of f is given then the validation
  %   has floor(N*f) and training set has N - floor(N*f).
  
  properties
    factor %The factor in ]0, 1[ used to calculate validation set size
    training %Training set, after the split
    validation %Validation set, after the split
  end
  
  methods
    function this = FactorSplitter(factor)
      %FactorSplitter Creates a factor splitter, specifying factor
      %   Constructs the factor splitter object, specifying the factor,
      %   that is the proportion of validation set size on the whole
      %   dataset size.
      %
      %   - Inputs:
      %     - factor: the factor of the proportion, in ]0, 1[
      
      %Checking the factor
      assert(factor > 0 && factor < 1, 'FactorSplitter:invalidFactor', ...
        sprintf('Invalid factor: %f', factor));
      this.factor = factor;
    end
    
    function split(this, ds)
      %split Preliminary splits the dataset into train and validation
      %   Splits the given dataset using the proportional factor, in order
      %   to obtain training set and validation set.
      %
      %   Inputs:
      %     - ds: the Dataset object that has to be split
      
      %Calculating sizes of training and validation
      vsSize = ceil(ds.num * this.factor);
      tsSize = ds.num - vsSize;
      %Creating datasets
      ds.shuffle(); %Shuffling the dataset
      this.training = dataset.Dataset(ds.patternSizes, ds.labelSizes);
      this.training.setPatternsAndLabels(ds.patterns(1:tsSize, :), ...
        ds.labels(1:tsSize, :));
      this.validation = dataset.Dataset(ds.patternSizes, ds.labelSizes);
      this.validation.setPatternsAndLabels(ds.patterns(tsSize + 1:end, :), ...
        ds.labels(tsSize + 1:end, :));
    end
      
    function ts = getTrainingSet(this, ~)
      %getTrainingSet Gets the training set
      %   Gets the training set after the split. With this method, training
      %   and validation sets are independent from train status and so the
      %   argument will be ignored.
      %  
      %   Outputs:
      %     - ts: the training set
      ts = this.training;
    end
      
    function vs = getValidationSet(this, ~)
      %getValidationSet Gets the validation set
      %   Gets the validation set after the split. With this method, training
      %   and validation sets are independent from train status and so the
      %   argument will be ignored.
      %  
      %   Outputs:
      %     - vs: the validation set
      vs = this.validation;
    end
  end
end

