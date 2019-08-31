%Author: Francesco Altiero
%Date: 14/12/2018

classdef WeightUpdateStrategy < handle
  %WEIGHTUPDATESTRATEGY Interface for updating weight in a neural net
  %   Defines methods a concrete object has to implement in order to
  %   perform the weight update of the net. This interface provides a
  %   method used to recover delta values for weights and biases of any
  %   net's level, as a cell array whose length is the depth of the net.
  
  methods (Abstract)
    update(this, net, derW, derB);
      %update Updates the weight in the net using derivatives
      %   Calculates differential in the weights and the biases of the
      %   given neural net and updates them. Any implementor specifies the
      %   strategy used to update weights.
      %
      %   Inputs:
      %     - net: the neural net object
      %     - derW: a cell array with the derivatives of each net's layer
      %         with respect to weights
      %     - derB: a cell array with the derivatives of each net's layer
      %         with respect to biases
      
  end
end

