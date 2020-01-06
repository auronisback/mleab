classdef Layer < handle
  %LAYER Summary of this class goes here
  %   Detailed explanation goes here
  
  properties(SetAccess = protected)
    name;
    inputShape;
    activation;
    Z;  % Cached outputs
  end
  
  methods(Abstract)
    Z = predict(this, X);
    Z = forward(this, X);
    [dhid, dW, db] = backward(this, delta, X, Wnext);
    [dout, dW, db] = outputBackward(this, errorFun, X, T);
    updateParameters(this, deltaW, deltaB);
  end
end

