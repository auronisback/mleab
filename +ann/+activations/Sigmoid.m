classdef Sigmoid < ann.activations.ActivationFunction
  %SIGMOID Summary of this class goes here
  %   Detailed explanation goes here
  
  methods
    function Z = eval(~, A)
      Z = 1 ./ (1 + exp(-A));
    end
    
    function dZ = derive(this)
      dZ = this.layer.Z .* (1 - this.layer.Z);
    end
  end
end

