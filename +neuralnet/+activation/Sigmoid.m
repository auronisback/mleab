%Author: Francesco Altiero
%Date: 09/12/2018

classdef Sigmoid < neuralnet.activation.ActivationFunction
  %SIGMOID Logistic sigmoid activation function
  %   Calculates values and derivatives for a logistic sigmoid activation
  %   function s, that is:
  %     s(x) = 1 / (1 + e^(-x))
  
  properties (Constant)
    NAME_SIGMOID = 'sigmoid' %Name of this activation function
  end
  
  methods
    function this = Sigmoid(layer)
      %SIGMOID Construct an instance of this class
      %   Creates a new sigmoid activation function related to the given
      %   net's layer given.
      %
      %   Inputs:
      %     - layer: the NetLayer object for which the object is the
      %         activation function
      
      %Calling parent's constructor with this class name
      this@neuralnet.activation.ActivationFunction(...
        neuralnet.activation.Sigmoid.NAME_SIGMOID, layer);
    end
    
    function Z = eval(this, A)
      %eval Evaluates the sigmoid function on input A or on the layer's
      %activations.
      %   Calculates the sigmoidal function to the network's activations
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
      Z = 1 ./ (1 + exp(-A));
    end
    
    function df = derivative(this, A)
      %derivative Derive the sigmoidal function w.r.t. activations.
      %   Returns the derivative value of the sigmoidal function with respect
      %   to activations. If activations are not given explicitely, then the
      %   values are taken directly from the linked layer.
      % 
      %   Inputs:
      %     - A: the activation value of neuron in the layer. If not given,
      %       the layer activations will be taken (it is mandatory that the
      %       activation values were cached)
      %   Outputs:
      %     - df: the derivative value of the sigmoidal evaluated in the
      %       activation values
      
      if ~exist('A', 'var')
        %Calculating derivatives by layer's outputs
        Z = this.layer.outputs;
      else
        %Calculating derivatives on input
        Z = this.eval(A);
      end
      %Calculating derivative
      df = Z .* (1 - Z);
    end
  end
end

