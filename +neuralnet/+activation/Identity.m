%Author: Francesco Altiero
%Date: 09/12/2018

classdef Identity < neuralnet.activation.ActivationFunction
  %IDENTITY Identity activation function
  %   Activation function that does no transformation on the input and
  %   implements the identity function, that is
  %     id(x) = x
  
  properties (Constant)
    NAME_IDENTITY = 'identity' %Name of this activation function
  end
  
  methods
    function this = Identity(layer)
      %IDENTITY Construct an instance of this class
      %   Creates a new identity activation function related to the given
      %   net's layer given.
      %
      %   Inputs:
      %     - layer: the NetLayer object for which the object is the
      %         activation function
      
      %Calling parent's constructor with this class name
      this@neuralnet.activation.ActivationFunction(...
        neuralnet.activation.Identity.NAME_IDENTITY, layer);
    end
    
    function Z = eval(this, A)
      %eval Evaluates the identity function on input A or on the layer's
      %activations.
      %   Calculates the identity function to the network's activations
      %   values. If not given explicitely, the network layer is used to
      %   obtain activations.
      %
      %   Inputs:
      %   - A: the activation value of neuron in the layer. If not given,
      %       the layer activations will be taken (it is mandatory that the
      %       activation values were cached)
      %   Outputs:
      %     - Z: output fired for each neuron in the net
      
      %Checking if calculate values directly or take them from layer
      if ~exist('A', 'var')
        A = this.layer.activations;
      end
      Z = A;
    end
    
    function df = derivative(this, A)
      %derivative Derive the identity function w.r.t. activations.
      %   Returns the derivative value of the identity function with
      %   respect to activations. If activations are not given explicitely,
      %   then the values are taken directly from the linked layer.
      % 
      %   Inputs:
      %     - A: the activation value of neuron in the layer. If not given,
      %       the layer activations will be taken (it is mandatory that the
      %       activation values were cached)
      %   Outputs:
      %     - df: the derivative value of the sigmoidal evaluated in the
      %       activation values
      
      if ~exist('A', 'var')
        A = this.layer.activations;
      end
      %Calculating derivative: a one matrix with same size as A
      df = ones(size(A));
    end
  end
end

