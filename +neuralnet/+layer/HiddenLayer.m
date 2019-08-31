classdef HiddenLayer < neuralnet.layer.NetLayer
  %HIDDENLAYER Defines an hidden layer in the neural network
  %   This class manages network layers that are hidden in the net, that is
  %   they are not output layers. For those layers, the backward
  %   propagation uses values calculated in subsequential layers, 
  %   independenty from the error function the net is trained with.
  
  methods
    function this = HiddenLayer(dimensions)
      %Calling parent's constructor
      this@neuralnet.layer.NetLayer(dimensions);
    end
  end
  
  methods (Abstract)
    dhid = backward(this, delta, Wnext);
      %backward Performs the backward propagation in order to tune weights.
      %   In concrete implementor, this method should perform the back
      %   propagation of input data given deltas obtained by next layer.
      %
      %   Inputs:
      %     - delta: delta from the next layer if this layer is not an output
      %         layer
      %     - errFun: the error function used to calculate deltas if this
      %         layer is an output layer
      %   Outputs:
      %     - d: the delta value for nodes in this layer
  end
end

