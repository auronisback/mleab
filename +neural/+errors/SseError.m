classdef SseError < neural.errors.ErrorFunction
  %SSEERROR Sum of Squared error function
  %   Manages SSE error function, intended as:
  %     E(Y, T) = sum((Y - T)^2) / 2
  
  methods
    function e = evaluate(~, Y, T)
      %evaluate Evaluates the SSE function
      %   Evaluates the SSE function on given outputs and labels.
      % Inputs:
      %   - Y: outputs of the network
      %   - T: ground truth labels of samples
      % Outputs:
      %   - e: a measure of error between Y and T
      e = Y - T;
      e = e .* e / 2;
      e = sum(e, 'all');
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

