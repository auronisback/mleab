%Author: Francesco Altiero
%Date: 10/12/2018

classdef DatasetSplitter < handle
  %DATASETSPLITTER Interface for splitter object
  %   Interface that a splitter must implement in order to be used in the
  %   training of a neural network. It provides the split method used to
  %   initially split the dataset, and methods to obtain training and
  %   validation sets.
  
  methods (Abstract)
    split(this, dataset);
      %split Preliminary splits the dataset into train and validation
      %   Splits the given dataset in order to be used when needed in the
      %   training algorithm.
      %
      %   Inputs:
      %     - dataset: the Dataset object that has to be split
      
    ts = getTrainingSet(this, trainObj);
      %getTrainingSet Gets the training set
      %   Gets the training set after the split. The training object is
      %   used in order to return a training set that depends on some
      %   training properties (such as epochs in K-fold validation).
      %  
      %   Inputs:
      %     - trainObj: an AbstractTraining concrete object used to obtain
      %         additional parameters in order to return training set
      %   Outputs:
      %     - ts: the training set
      
    vs = getValidationSet(this, trainObj);
      %getValidationSet Gets the validation set
      %   Gets the validation set after the split. The training object is
      %   used in order to return a validation set that depends on some
      %   training properties (such as epochs in K-fold validation).
      %  
      %   Inputs:
      %     - trainObj: an AbstractTraining concrete object used to obtain
      %         additional parameters in order to return validation set
      %   Outputs:
      %     - ts: the validation set
  end
end

