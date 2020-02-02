classdef SsError < ann.errors.ErrorFunction
  %SSERROR Sum of Squares error function
  %   Defines the Sum of Squares error function used to train a neural
  %   network.
  
  methods
    function E = eval(~, Y, T)
      %eval Evaluates the Sum of Squares error function
      %   Evaluates the Sum of Squares error between net's output and
      %   target values.
      % Inputs:
      %   - Y: output of network
      %   - T: ground truth values
      % Output:
      %   - E: evaluation of SoS error function between Y and T
      E = T - Y;
      E = E .* E;
      E = .5 * sum(E);
    end
    
    function dE = derive(~, Y, T)
      %derive Calculate derivatives of SoS  function w.r.t. outputs
      %   Derives the Sum of Squares error between network's output and 
      %   target values.
      % Inputs:
      %   - Y: network's output values
      %   - T: target values
      % Output:
      %   - dE: derivative of Sum of Squared error function with respect to
      %       outputs; it is
      dE = Y - T;
    end
    
    function s = toString(~)
      %toString Gets the string representation of the object
      %   Converts the error function into its string representation.
      % Outputs:
      %   - s: string representation of the Sum of Squared Error
      s = 'Sum of Squared Error';
    end
  end
end

