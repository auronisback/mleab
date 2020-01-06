classdef FcLayer < ann.layers.Layer
  %FCLAYER Summary of this class goes here
  %   Detailed explanation goes here
  
  properties(SetAccess = private)
    nodeNum;
    W;
    b;
    A;  % Cached activation values
  end
  
  methods
    function this = FcLayer(inputShape, nodeNumber, activation)
      %FCLAYER Construct an instance of this class
      %   Detailed explanation goes here
      this.inputShape = prod(inputShape, 'all');
      this.activation = activation;
      this.activation.setLayer(this);
      this.nodeNum = nodeNumber;
      this.initializeWeightsAndBiases();
    end
    
    function Z = predict(this, X)
      Z = this.activation.eval(...
        X * this.W.' + repmat(this.b, size(X, 1), 1));
    end
    
    function Z = forward(this, X)
      this.A = (X * this.W.') + repmat(this.b, size(X, 1), 1);
      this.Z = this.activation.eval(this.A);
      Z = this.Z;
    end
    
    function [dhid, dW, db] = backward(this, delta, X, Wnext)
      % X: layer's input
      % Wnext: weights of subsequent layer
      % delta: delta as in Bishop
      % errorFunction: error if it is last layer
      % dhid: hidden deltas
      dhid = (delta * Wnext) .* this.activation.derive();
      dW = dhid.' * X;
      db = sum(dhid);
    end
    
    function [dout, dW, db] = outputBackward(this, errorFun, X, T)
      % Used if the layer is last layer. Output is cached into Z
      dout = errorFun.derive(this.Z, T) .* this.activation.derive();
      dW = dout.' * X;
      db = sum(dout);
    end
    
    function updateParameters(this, deltaW, deltaB)
      this.W = this.W + deltaW;
      this.b = this.b + deltaB;
    end
  end
  
  methods(Access = private)
    function initializeWeightsAndBiases(this)
      %Random uniform in [-1, 1]
      this.W = 1 - 2 * rand(this.nodeNum, this.inputShape);
      this.b = 1 - 2 * rand(1, this.nodeNum);
    end
  end
end

