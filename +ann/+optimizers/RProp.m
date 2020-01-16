classdef RProp < ann.optimizers.Optimizer
  %RPropUpdate Updates a neural network using the Resilient backPROPagation
  %method when updating weights.
  
  properties(Constant)
    DEFAULT_DELTA_MIN = 1e-09; % Default value for delta min
    DEFAULT_DELTA_MAX = 50; % Default value for delta max
    DEFAULT_DELTA_ZERO = .125;  % Default delta value before training
  end
  
  properties(SetAccess = private)
    etaPlus;  % Learning rate for same sign derivatives
    etaMinus;  % Learning rate for different sign derivatives
    deltaZero;  % Initial (scalar) delta value
    deltaMin;  % Minimum delta value in calculating update terms
    deltaMax;  % Maximum delta value in calculating update terms
    prevDeltaW; %Delta weights value for previous epoch
    prevDeltaB; %Delta biases value for previous epoch
    prevGradW;  % Previous derivatives for weights
    prevGradB;  % Previous derivatives for biases
  end
  
  methods
    function this = RProp(etaMinus, etaPlus, deltaZero, deltaMin, deltaMax)
      %RProp Creates the RProp object specifying eta+, eta- and 
      %deltaZero values. Optional arguments are deltaMin and deltaMax 
      %values used to set a minimum and a maximum update for each step.
      %
      %Inputs:
      % - etaPlus: the scalar value for eta applied when actual and
      %            previous derivatives match
      % - etaMinus: scalar value for eta applied when current and previous
      %             derivatives mismatch in sign
      % - deltaZero: optional, scalar with the starting update value for 
      %              the first training epoch. If none given, first epoch
      %              derivative will be used
      % - deltaMin: optional, is the scalar minimum value for an update
      %             step. Must be supplied with deltaMax value.
      % - deltaMax: optional, is the scalar maximum value for an update
      %             step. Must be supplied with deltaMin value.
      
      % Checking consistency of etas
      assert(etaPlus >= 1, 'Rprop:invalidEtaPlus',...
        'Invalid eta+ value: %f', etaPlus);
      assert(0 < etaMinus && etaMinus < 1, 'Rprop:invalidEtaMinus', ...
        'Invalid eta- value: %f', etaMinus);
      this.etaMinus = etaMinus;
      this.etaPlus = etaPlus;
      % Setting delta zero if given
      if(~exist('deltaZero', 'var'))
        this.deltaZero = ann.optimizers.RProp.DEFAULT_DELTA_ZERO;
      else
        this.deltaZero = deltaZero;
      end
      % If deltaMin and deltaMax are given, they are initialized
      if(exist('deltaMin', 'var') && exist('deltaMax', 'var'))
        % Checking consistency of delta min and delta max
        assert(deltaMin < deltaMax, 'RProp:invalidDeltaMinMax', ...
            'Minimum delta is greater than maximum delta value');
        this.deltaMin = deltaMin;
        this.deltaMax = deltaMax;
      else  % If not given, they're set at a default values
        this.deltaMin = ann.optimizers.RProp.DEFAULT_DELTA_MIN;
        this.deltaMax = ann.optimizers.RProp.DEFAULT_DELTA_MAX;
      end
    end

    function [deltaW, deltaB] = evaluateDeltas(this, dW, db, ~)
      %evaluateDeltas Evaluates delta using RProp
      %   Evaluates the values of delta for weights and biases in the
      %   neural network using actual derivatives of error function.
      % Inputs:
      %   - dW: the derivatives of error function for weights
      %   - dB: the derivatives of error function for biases
      % Outputs:
      %   - deltaW: cell-array with delta values for weights
      %   - deltaB: cell-array with delta values for biases
      
      L = size(dW, 2);  % Caching numbel of layers
      % Pre-allocating outputs
      deltaW = cell(1, L);
      deltaB = cell(1, L);
      % Setting delta for epoch before the first
      if(isempty(this.prevDeltaW))
        for l = 1:L
          deltaW{l} = repmat(this.deltaZero, size(dW{l}));
          deltaB{l} = repmat(this.deltaZero, size(db{l}));
        end
      else
        for l = 1:L
          % Getting signs
          signW = sign(dW{l} .* this.prevGradW{l});
          signB = sign(db{l} .* this.prevGradB{l});
          % Calculating deltas
          deltaW{l} = ... 
            min(this.deltaMax, this.prevDeltaW{l} * this.etaPlus) .* (signW > 0) + ...
            max(this.deltaMin, this.prevDeltaW{l} * this.etaMinus) .* (signW < 0) + ...
            this.prevDeltaW{l} .* (signW == 0);
          deltaB{l} = ... 
            min(this.deltaMax, this.prevDeltaB{l} * this.etaPlus) .* (signB > 0) + ...
            max(this.deltaMin, this.prevDeltaB{l} * this.etaMinus) .* (signB < 0) + ...
            this.prevDeltaB{l} .* (signB == 0);
        end
      end
      % Caching previous deltas
      this.prevDeltaW = deltaW;
      this.prevDeltaB = deltaB;
      % Applying sign to output deltas
      for l = 1:L
        deltaW{l} = - sign(dW{l}) .* deltaW{l};
        deltaB{l} = - sign(db{l}) .* deltaB{l};
      end
      this.prevGradW = dW;
      this.prevGradB = db;
    end
    
    function clear(this)
      %clear Clears this object, resetting its values
      %   Clears the object, resetting its values. It is used after the
      %   whole training has been done. This method resets the previous 
      %   derivatives' to empty array.
      this.prevDeltaW = [];
      this.prevDeltaB = [];
    end
  end
  
end

