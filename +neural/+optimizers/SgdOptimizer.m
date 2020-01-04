classdef SgdOptimizer < neural.optimizers.Optimizer
  %SGDOPTIMIZER Stochastic Gradient Descent optimizer
  %   Optimize weights using Stochastic Gradient Descent algorithm,
  %   calculating delta as -\eta * \nabla W and -\eta * \nabla b
  
  properties
    learningRate;  % The learning rate
  end
  
  methods
    function this = SgdOptimizer(learningRate)
      %SGDOPTIMIZER Construct an instance of SGD optimizer
      %   Creates an SGD oprimizer with specified learning rate.
      this.learningRate = learningRate;
    end
    
    function [deltaW, deltab] = calculateDeltas(this, dW, db, ~)
      %calculateDeltas Calculates deltas using SGD algorithm
      %   Applies the stochastic gradient descent using the learning rate
      %   which was specified when the object was created.
      % Inputs:
      %   - dW: derivatives of the layer w.r.t. weights
      %   - db: derivatives of the layer w.r.t. biases
      deltaW = - this.learningRate * dW;
      deltab = - this.learningRate * db;
    end
  end
end

