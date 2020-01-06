classdef Optimizer < handle
  %OPTIMIZER Abstract class for optimizers
  %   Manages common properties of neural network optimizers, which are
  %   used to manage the update of weights and biases in network's layers.
  
  methods(Abstract)
    
    [deltaW, deltab] = calculateDeltas(this, dW, db, N, position);
      %calculateDeltas Performs the delta for weights and biases of a layer
      %   Use derivatives of layer's weights and biases in order to
      %   calculate delta values which will be added to the actual layer's
      %   weights in the training process.
      % Inputs:
      %   - dW: derivatives of error function w.r.t. weights
      %   - db: derivatives of error function w.r.t. db
      %   - N: number of training samples in the batch
      %   - position: layer's position in the network, used for caching
      %     operations when needed
      % Outputs:
      %   - deltaW: delta values for weights
      %   - deltab: delta values for biases
      
  end
end

