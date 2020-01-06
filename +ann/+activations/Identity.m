classdef Identity < ann.activations.ActivationFunction
  %IDENTITY Summary of this class goes here
  %   Detailed explanation goes here
  
  methods
    function Z = eval(~, A)
      Z = A;
    end
      
    function dZ = derive(this)
      dZ = ones(size(this.layer.A));
    end
  end
end

