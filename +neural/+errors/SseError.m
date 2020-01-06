classdef SseError < neural.errors.ErrorFunction
  %SSEERROR Sum of Squared error function
  %   Manages SSE error function, intended as:
  %     E(Y, T) = sum((T - Y)^2) / 2
  
  properties(Constant)
    NAME_SSE = 'SSE';  % Name of error function
  end
  
  methods
    function this = SseError()
      %SSEERROR Creates a new instance of sum of squared error function.
      this = this@neural.errors.ErrorFunction(...
        neural.errors.SseError.NAME_SSE);
    end
    
    function e = evaluate(~, Y, T)
      %evaluate Evaluates the SSE function
      %   Evaluates the SSE function on given outputs and labels.
      % Inputs:
      %   - Y: outputs of the network
      %   - T: ground truth labels of samples
      % Outputs:
      %   - e: a measure of error between Y and T
      e = T - Y;
      e = e .* e;
      e = .5 * sum(e, 'all');
    end
    
    function dE = derive(~, Y, T)
      %derive Derives the SSE function w.r.t. outputs
      %   Evaluates the derivative of the SSE function given the
      %   network's output and labels of samples which were fed to the
      %   network.
      % Inputs:
      %   - Y: outputs of the network
      %   - T: fround truth labels of samples
      % Outputs:
      %   - dE: derivatives of the error function w.r.t. network's outputs
      dE = Y - T;
    end
  end
end

