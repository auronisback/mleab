classdef ActivationFunction < handle
  %ACTIVATIONFUNCTION Summary of this class goes here
  %   Detailed explanation goes here
  
  properties(SetAccess = private)
    layer;
  end
  
  methods
    function setLayer(this, layer)
      this.layer = layer;
    end
  end
  
  methods(Abstract)
    Z = eval(this, A);
    dZ = derive(this);
  end
end

