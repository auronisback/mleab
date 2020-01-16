classdef CrossEntropy < ann.errors.ErrorFunction
  %CROSSENTROPY Cross-entropy error function
  %   Crossentropy error function used to train the network. The error
  %   function will be:
  %              ---
  %              \
  %   E(t,y) = - /    t_i * ln(y_i)
  %              ---
  %               i
  %   with i which ranges over all classes.
  %   In order to calculate derivative w.r.t. outputs, used formula will
  %   be:
  %    dE       t_i
  %   ---- = - -----
  %   dy_i      y_i
  %
  %   A small constant is used if results are NaN.
  
  properties(Constant)
    EPSILON = 1e-06;  % Constant used to prevent NaNs
  end
  
  methods
    function E = eval(this, Y, T)
      %eval Evaluates the cross entropy error
      %   Evaluates cross-entropy error between output and truth values.
      % Inputs:
      %   - Y: network's output
      %   - T: true output values
      % Output:
      %   - E: measure of cross-entropy error
      
      % Error is taken to be maximum of Y or the epsilon value to prevent
      % NaNs
      E = reallog(max(Y, this.EPSILON));
      E = - sum(T .* E, 'all');
    end
    
    function dE = derive(~, Y, T)
      %derive Derives the cross-entropy error function.
      %   Derives cross-entropy error function w.r.t. all output values.
      % Inputs:
      %   - Y: network's output
      %   - T: truth values for the ouptut
      % Output:
      %   - dE: derivatives of the cross-entropy function w.r.t. outputs
      dE = - T ./ Y;
    end
  end
end

