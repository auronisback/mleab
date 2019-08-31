% Author: Francesco Altiero
% Date: 27/12/2018
classdef RProp < neuralnet.train.update.WeightUpdateStrategy
  %RPropUpdate Updates a neural network using the Resilient backPROPagation
  %method when updating weights.
  
  properties(Constant)
    DEFAULT_DELTA_MIN = 1e-6; %Default value for delta min
    DEFAULT_DELTA_MAX = 50; %Default value for delta max
    DEFAULT_DELTA_ZERO_LRATE = 0.001; %Default value for learning rate used
                                      %when first deltas are set to
                                      %derivative
  end
  
  properties
    etaPlus %Eta value for same sign derivatives
    etaMinus %Eta value for different sign derivatives
    deltaMin %Minimum delta value in calculating update terms
    deltaMax %Maximum delta value in calculating update terms
    deltaZero %Initial value for deltas
  end
  
  properties(SetAccess = private)
    prevDeltaW %Delta weights value for previous epoch
    prevDeltaB %Delta biases value for previous epoch
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
      
      %Checking consistency of etas
      if(etaMinus > etaPlus)
        throw(MException('RPropUpdate:etaMinusGreaterThanEtaPlus', ...
            'eta minus is greater than eta plus'));
      end
      this.etaMinus = etaMinus;
      this.etaPlus = etaPlus;
      %Setting delta zero value
      if(exist('deltaZero', 'var'))
        this.deltaZero = deltaZero;
      end
      %If deltaMin and deltaMax are given, they are initialized
      if(exist('deltaMin', 'var') && exist('deltaMax', 'var'))
        %Checking consistency of delta min and delta max
        if(deltaMin > deltaMax)
          throw(MException('RPropUpdate:deltaMinGreaterThanDeltaMax', ...
            'minimum delta is greater than maximum delta value'));
        end
        this.deltaMin = deltaMin;
        this.deltaMax = deltaMax;
      else
        %If not given, they're set at a default values
        this.deltaMin = neuralnet.train.update.RProp.DEFAULT_DELTA_MIN;
        this.deltaMax = neuralnet.train.update.RProp.DEFAULT_DELTA_MAX;
      end
    end
    
    function update(this, net, derW, derB)
      %update Updates the weight of the net with Resilient BackPROPagation,
      %using the derivatives of weights and biases in each layer.
      %
      %Input:
      % - net: the neural net whose weights have to be updated
      % - derW: the derivatives of error function for weights
      % - derB: the derivatives of error function for biases
            
      %Initializing previous weights and biases if one of them is empty
      if(isempty(this.prevDeltaW) || isempty(this.prevDeltaB) ...
          || isempty(this.prevDerSignW) || isempty(this.prevDerSignB))
        %Checking if a delta zero value has been given
        if(isempty(this.deltaZero))
          this.initDerivativesToFirst(derW, derB);
        else
          this.initDerivativesToDeltaZero(net);
        end
      end
      %Initializing current delta for weights and biases
      deltaW = cell(1, net.depth + 1);
      deltaB = cell(1, net.depth + 1);
      %Updating each layer of the net
      for l = 1:net.depth
        
      end
      %Updating the output layer
      
      %Setting the new deltas as previous ones for next epoch
      this.prevDeltaW = deltaW;
      this.prevDeltaB = deltaB;
      %Updating previus derivatives' sign matrix as current
      this.prevDerSignW = derW;
      this.prevDerSignB = derB;
    end
    
    function clear(this)
      %clear Clears this object, resetting its values. It is used after the
      %whole training has been done. This method resets the previous 
      %derivatives' sign to empty array.
      this.prevDerSignW = [];
      this.prevDerSignB = [];
      this.prevDeltaW = [];
      this.prevDeltaB = [];
    end
  end
  
  methods(Access=private)
    function initDerivativesToDeltaZero(this, net)
     %initDerivativesToDeltaZero Initializes the derivatives for the first
     %training epoch when delta zero has been given
     %  Initializes the parameter used in the RProp when in the first 
     %  training epoch if a fixed delta zero values has been given.
     %
     %  Inputs:
     %    - net: the neural net whose weights have to be updated, used to
     %           obtain weights' size
     
     %Initializing cell arrays for speed
     this.prevDerSignW = cell(1, net.depth + 1);
     this.prevDerSignB = cell(1, net.depth + 1);
     this.prevDeltaW = cell(1, net.depth + 1);
     this.prevDeltaB = cell(1, net.depth + 1);
     %Creating default values for hidden layers
     for l = 1:net.depth
       %Caching layer
       layer = net.hiddenLayers{l};
       %Setting initial signs to zero
       this.prevDerSignW{l} = zeros(size(layer.weights));
       this.prevDerSignB{l} = zeros(size(layer.biases, 2));
       %Setting initial delta zero value
       this.prevDeltaW{l}(1:size(layer.weights, 1), ...
         1:size(layer.weights, 2)) = this.deltaZero;
       this.prevDeltaB{l}(1:size(layer.biases, 2)) = this.deltaZero;
     end
     %Default values for output layer
     layer = net.outputLayer;
     l = net.depth + 1;
     %Setting initial signs to zero
     this.prevDerSignW{l} = zeros(size(layer.weights));
     this.prevDerSignB{l} = zeros(size(layer.biases, 2));
     %Setting initial delta zero value
     this.prevDeltaW{l}(1:size(layer.weights, 1), ...
       1:size(layer.weights, 2)) = this.deltaZero;
     this.prevDeltaB{l}(1:size(layer.biases, 2)) = this.deltaZero;
    end
   
    function initDerivativesToFirst(this, derW, derB)
      %initDerivativesToFirst Initializes delta value for the first epoch
      %when no fixed delta zero has been given
      %   This method initializes derivatives for weights and biases in the
      %   first epoch (no previous delta is present) to the derivatives of
      %   weights and biases.
      %
      %   Inputs:
      %     - derW: derivatives w.r.t. weights
      %     - derB: derivatives w.r.t. biases
      
      this.prevDeltaW = - derW;
      this.prevDeltaB = - derB;
    end
    
    function [deltaW, deltaB] = calculateLayerDeltas(this, derSignW, ... 
        derSignB, l)
      %calculateLayerDeltas Calculate deltas for weights and biases
      %   Performs the operations in order to calculate the values of the
      %   delta used to tune weights and biases.
      %
      %   Inputs:
      %     - derSignW: the sign of derivatives for weights in the layer
      %     - derSignB: the sign of derivatives for biases in the layer
      %     - l: the layer index in the net's layers
      %   Outputs:
      %     - deltaW: the delta value for the weights in the layer
      %     - deltaB: the delta value for the biases in the layer
      
      %Initializing eta matrix for weights and biases
      HW = ones(size(derSignW));
      HB = ones(size(derSignB));
      %Caching matrix product
      prodW = derSignW .* this.prevDerSignW{l};
      prodB = derSignB .* this.prevDerSignB{l};
      %Conditionally initialize eta matrices
      HW(prodW > 0) = this.etaPlus;
      HW(prodW < 0) = this.etaMinus;
      HB(prodB > 0) = this.etaPlus;
      HB(prodB < 0) = this.etaMinus;
      %Calculating deltas as a point-wise multiplication
      deltaW = HW .* this.prevDeltaW{l};
      deltaB = HB .* this.prevDeltaB{l};
    end
  end
end

