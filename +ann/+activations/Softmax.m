classdef Softmax < ann.activations.ActivationFunction
  %SOFTMAX Softmax activation function
  %   Defines and performs operations used to evaluate and derive a softmax
  %   activation function. The activation is evaluated as:
  % 
  %              e^(a_i - max_j(a_j))
  % s(a_i) = ------------------------
  %            C
  %           ---
  %           \
  %           /    e^(a_k - max_j(a_j))
  %           ---
  %           k=1
  % where k ranges over all C activation values for the layer. Subtraction
  % of maximum is added for numerical stability and follows from the
  % relation s(a) = s(a + c) where c is a constant term.
  %
  % The derivative is equal to:
  %           C
  %         -----
  %   dE    \       dE
  %  ---- = /      ---- * (delta_ik * z_i - z_i * z_k)
  %  da_i   -----  dz_k
  %          k=1
  
  methods
    function Z = eval(~, A)
      %eval Evaluates softmax on all activations
      %   Evaluates the softmax activation function.
      % Inputs:
      %   - A: activation values of the layer
      % Output:
      %   - Z: output values
      expA = exp(A - max(A, [], 2));
      expSum = repmat(sum(expA, 2), 1, size(A, 2));
      Z = expA ./ expSum;
    end
    
    function dA = derive(this, dZ)
      %derive Derives the softmax function
      %   Derives softmax function using layer's activation values.
      % Inputs:
      %   - dZ: derivatives of layer's output
      % Output:
      %   - dA: derivatives of softmax output with respect to activations
      deltaZ = dZ .* this.layer.Z;
      S = repmat(sum(deltaZ, 2), 1, size(dZ, 2));
      dA = deltaZ - this.layer.Z .* S;
    end
  end
end

