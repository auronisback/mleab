classdef Relu < ann.activations.ActivationFunction
  %RELU Rectified Linear Unit activation function
  %   Rectified Linear Unit activation function, which calculates nework's
  %   output using:
  %     z_j = max(0, a_j)
  %   Its derivatives are calculated simply setting to 1 all outputs which
  %   were non-negative.
  
  methods
    function Z = eval(~, A)
      %eval Evaluates ReLU activation function
      %   Evaluates the ReLU activation function on layer's activation
      %   values.
      % Inputs:
      %   - A: layer's activation values
      % Outputs:
      %   - Z: a matrix with same shape of A in which negative values are
      %     suppressed
      Z = max(0, A);
    end
    
    function dA = derive(this, dZ)
      %derive Derives the ReLU activation function
      %   Calculates derivatives of activation function w.r.t. activation
      %   values cached in the layer and derivatives of error function
      %   w.r.t. layer's output.
      % Inputs:
      %   - dZ: derivatives of error function w.r.t. layer's output
      % Output:
      %   - dA: derivatives of error function w.r.t. the layer's activation
      %     values 
      dA = dZ .* (this.layer.A >= 0);
    end
  end
end

