classdef FcConvEquivLayer < ann.layers.Layer
  %FCCONVEQUIVLAYER Fully-connected layer equivalent to a convolutional one
  %   Implements a Fully-Connected layer which is equivalen to a
  %   convolutional layer. The implementation is faster because
  %   computations are done using simple optimized matrix multiplications
  %   in opposition to convolutions.
  %   Objects of this class use an hidden fully-connected layer object in
  %   order to do all calculations.
  
  properties(SetAccess = private)
    filterNum;  % Number of filters in the layer
    filterShape;  % Shape of each filter
    stride = [1, 1];  % Vertical and horizontal stride in convolutions
    padding = [0, 0];  % Vertical and horizontal padding in convolutions
    fcLayer;  % A fully connected layer which performs calculations
  end
  
  methods
    function this = FcConvEquivLayer(inputShape, filterNum, filterShape, ...
        activation, stride, padding)
      %FcConvEquivLayer Construct an instance of this class
      %   Creates a fully-connected layer which performs 2D convolutions
      %   using matrix multiplications.
      % Inputs:
      %   - inputShape: dimension of convolution inputs
      %   - filterNum: the number of filters involved in convolution
      %   - filterShape: the size of each filter for convolution
      %   - activation: layer's activation function
      %   - stride: optional parameter for the stride size, as a scalar or
      %       a vector. In the first case, the stride will be equal to both
      %       horizontal and vertical dimensions; in the latter case, the
      %       vertical stride is set to the first element in the array
      %       while the horizontal stride is set to the second one.
      %       Defaults to [1, 1] if not given
      %   - padding: optional argument for vertical and horizontal padding.
      %       If a scalar is given, the padding will be equal to the given
      %       value; if an array is given, vertical and horizontal padding
      %       will be set similarly as seen for stride; if the 'valid'
      %       string is given, no padding will be applied; if 'same' string
      %       is given, then the padding will be set to a value which
      %       produces an output image with the same size of the input. It
      %       defaults to [0, 0]
      
      % Adjusting channels input shape if 2-D
      if size(inputShape, 2) == 2
        inputShape = [inputShape, 1];
      end
      % Adjusting channels filter shape if 2-D
      if size(filterShape, 2) == 2
        filterShape = [filterShape, 1];
      end
      % Checking arguments
      this.checkMandatoryArguments(inputShape, filterNum, filterShape);
      this.filterNum = filterNum;
      this.inputShape = inputShape;
      this.filterShape = filterShape;
      this.activation = activation;
      % Initializing stride if needed
      if exist('stride', 'var')
        this.initializeStride(stride);
      end
      % Initializing padding if needed
      if exist('padding', 'var')
        this.initializePadding(padding);
      end
      % Initializing output size
      this.initializeOutputShape();
      % Initializing inner FC layer
      this.initializeFcLayer();
      this.name = 'Fc-Conv-Equiv';
    end
    
    function [W, b] = getParameters(this)
      %getParameters Gets weights and biases of the layer
      %   Getter for weights and biases of the layer.
      % Outputs:
      %   - W: weights
      %   - b: biases
      
      % Getting inner layer's parameters
      [W, b] = this.fcLayer.getParameters();
      % Reshaping weights
      W = reshape(W, this.filterNum, this.filterShape(2), ...
        this.filterShape(1), this.filterShape(3));
      W = permute(W, [1, 3, 2, 4]);
    end
    
    function setParameters(this, W, b)
      %setParameters Sets layer's parameters
      %   Setter for weights and biases of the layer.
      % Inputs:
      %   - W: weights, which should have same size of layer's weights
      %   - b: biases, which should have same size of layer's biases
      
      % Reshaping parameters in order to pass them to FC-layer
      W = permute(W, [1, 3, 2, 4]);
      W = reshape(W, [this.filterNum, prod(this.filterShape)]);
      this.fcLayer.setParameters(W, b);
    end
    
    function Z = predict(this, X)
      %predict Evaluates the convolutional layer on given input
      %   Performs convolutions between layer's filters and given set of
      %   images in order to produce layer's output.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Z: output value for given input
      
      % Creating patches used padded input
      X = padarray(X, [0, this.padding, 0], 0, 'both');
      lX = this.extractPatches(X, ...
        this.outputShape(1), this.outputShape(2), ...
        this.filterShape(1), this.filterShape(2), ...
        this.stride(1), this.stride(2));
      % Forwarding to FC layer
      Z = this.fcLayer.predict(lX);
      % Reshaping and permuting Z to its output size
      Z = this.reconstructShape(Z);
    end
    
    function Z = forward(this, X)
      %forward Evaluates convolutional layer on given input
      %   Performs a forward pass of the Conv layer and caches output and
      %   activation values in order to use them in training.
      % Inputs:
      %   - X: layer's input
      % Outputs:
      %   - Z: output for given input
      
      % Creating patches used padded input
      X = padarray(X, [0, this.padding, 0], 0, 'both');
      lX = this.extractPatches(X, ...
        this.outputShape(1), this.outputShape(2), ...
        this.filterShape(1), this.filterShape(2), ...
        this.stride(1), this.stride(2));
      % Forwarding to FC layer
      Z = this.fcLayer.forward(lX);
      % Reshaping and permuting Z to its output size
      Z = this.reconstructShape(Z);
      this.Z = Z;  % Caching outputs
      % Reshaping activations to its output size
      this.A = this.reconstructShape(this.fcLayer.A);
    end
    
    function [dX, dW, db] = backward(this, dZ, X)
      %backward Performs a backward pass in the layer
      %   Manages backpropagation on the layer in order to calculate
      %   derivatives w.r.t. weights and biases in the layer. It also
      %   calculates derivatives w.r.t. input used for backpropagation in
      %   previous layers.
      % Inputs:
      %   - dZ: derivatives of error w.r.t. layer's output
      %   - X: layer's input values
      % Outputs:
      %   - dX: derivatives w.r.t. layer's input
      %   - dW: derivatives w.r.t. layer's weights
      %   - db: derivatives w.r.t. layer's biases
      
      % Creating input patches used padded input
      X = padarray(X, [0, this.padding, 0], 0, 'both');
      lX = this.extractPatches(X, ...
        this.outputShape(1), this.outputShape(2), ...
        this.filterShape(1), this.filterShape(2), ...
        this.stride(1), this.stride(2));
      % Creating dZ patches
      ldZ = this.linearizeDZ(dZ);
      % Backwarding for weights and biases derivatives only
      [dX, dW, db] = this.fcLayer.backward(ldZ, lX);
      % Reshaping weights
      dW = reshape(dW.', this.filterShape(2), ...
        this.filterShape(1), this.inputShape(3), []);
      dW = permute(dW, [4, 2, 1, 3]);
      % Reshaping biases
      db = db.';
      % Obtaining derivatives w.r.t inputs
      dX = this.calculateInputDerivatives(dX);
    end
    
    function [dX, dW, db] = outputBackward(this, errorFun, X, T)
      %outputBackward Backpropagates errors if this is an output layer
      %   Manages backpropagation if the FC-layer is the last layer in the
      %   network, using error function, target values and layer's input.
      % Inputs:
      %   - errorFun: error function used to derive error with respect to
      %     network's output
      %   - X: layer's input
      %   - T: target values for the input
      % Outputs:
      %   - dX: derivatives w.r.t. this layer's inputs, used by previous
      %     layer
      %   - dW: derivatives w.r.t. layer's weights
      %   - db: derivatives w.r.t. layer's biases
      
      % Creating input patches used padded input
      X = padarray(X, [0, this.padding, 0], 0, 'both');
      lX = this.extractPatches(X, ...
        this.outputShape(1), this.outputShape(2), ...
        this.filterShape(1), this.filterShape(2), ...
        this.stride(1), this.stride(2));
      % Backwarding for weights and biases derivatives only
      [dX, dW, db] = this.fcLayer.outputBackward(errorFun, lX, T);
      % Reshaping weights
      dW = reshape(dW.', this.filterShape(2), ...
        this.filterShape(1), this.inputShape(3), []);
      dW = permute(dW, [4, 2, 1, 3]);
      % Reshaping biases
      db = db.';
      % Obtaining derivatives w.r.t inputs
      dX = this.calculateInputDerivatives(dX);
    end
    
    function [dW, db] = inputBackward(this, dZ, X)
      %inputBackward Evaluates layer's derivatives if it is the first layer
      %   Calculates derivatives for the layer when the layer itself is the
      %   first layer in the network. Derivatives w.r.t. inputs are not
      %   evaluated due to the fact that nothing has to be propagated
      %   backward.
      % Inputs:
      %   - dZ: derivatives w.r.t. layer's output
      %   - X: network's input
      % Outputs:
      %   - dW: derivatives w.r.t. filter's weights
      %   - db: derivatives w.r.t. biases
      
      % Creating input patches used padded input
      X = padarray(X, [0, this.padding, 0], 0, 'both');
      lX = this.extractPatches(X, ...
        this.outputShape(1), this.outputShape(2), ...
        this.filterShape(1), this.filterShape(2), ...
        this.stride(1), this.stride(2));
      % Creating dZ patches
      ldZ = this.linearizeDZ(dZ);
      % Backwarding for weights and biases derivatives only
      [dW, db] = this.fcLayer.inputBackward(ldZ, lX);
      % Reshaping weights
      dW = reshape(dW.', this.filterShape(2), ...
        this.filterShape(1), this.inputShape(3), []);
      dW = permute(dW, [4, 2, 1, 3]);
      % Reshaping biases
      db = db.';
    end
    
    function updateParameters(this, deltaW, deltaB)
      %updateParameters Updates this FC-layer's parameters
      %   Updates parameters of the layer using delta values calculated by
      %   an optimizer.
      % Inputs:
      %   - deltaW: delta values for weights
      %   - deltaB: delta values for biases
      
      % Reshaping deltas
      deltaW = reshape(deltaW, [this.filterNum, prod(this.filterShape)]);
      this.fcLayer.updateParameters(deltaW, deltaB.');
    end
    
    function reinitialize(this)
      %reinitialize Re-initializes weights and biases for the layer.
      %   Reinitializes all parameters of the hidden FC layer.
      this.fcLayer.reinitialize();
    end
    
    function s = toString(this)
      %toString Gets a string representation of this layer
      %   Converts the layer into its string representation.
      % Output:
      %   - s: string representation of the convolutional layer
      s = [this.name, sprintf('(#f: %d, shape: %dx%dx%d)', ...
        this.filterNum, this.filterShape)];
    end
    
  end
  
  methods(Access = private)
    
    function checkMandatoryArguments(~, inputShape, ...
        numFilters, filterShape)
      %checkMandatoryArguments Checks input given when the layer is created
      
      % Invalid type for number of filters
      if ~isscalar(numFilters)
        error('Conv2DLayer:InvalidNumFilters', ...
          'numFilters must be a scalar value');
      end
      % Invalid number of channel between input dimensions and filter size
      if any(inputShape(3:end) ~= filterShape(3:end))
        error('Conv2DLayer:DifferentChannelNumber', ...
          'Input and filters have different channels: %d vs %d', ...
          inputShape(3:end), filterShape(3:end))
      end
    end
    
    function initializeStride(this, stride)
      %initializeStride Initializes the stride hyperparameter
      if isscalar(stride)
        this.stride = repmat(stride, [1, 2]);
      else
        this.stride = stride(1:2);
      end
    end
    
    function initializePadding(this, padding)
      %initializePadding Initialize the padding hyper-parameter
      
      if ischar(padding)  % This is a character array
        switch padding
          case 'valid'
            this.padding = [0, 0];
          case 'same'
            % Same padding allowed only for odd filters
            assert(all(mod(this.filterShape(1:2), 2) == 1), ...
              'FcConvEquivLayer:samePaddingEvenFilter', ...
              '"same" padding allowed only for odd filters');
            this.padding = ceil(((this.inputShape(1:2) - 1) .* this.stride ...
              + this.filterShape(1:2) - this.inputShape(1:2))/2);
          otherwise
            error('Conv2DLayer:InvalidPadding', ...
              'Given padding is not valid: %s', padding);
        end
      elseif isscalar(padding)
        this.padding = repmat(padding, [1, 2]);
      else
        this.padding = padding(1:2);
      end
    end
    
    function initializeOutputShape(this)
      %initializeoutputShape Calculates shape of the output using layer's
      %parameters
      this.outputShape = floor((this.inputShape(1:2) - this.filterShape(1:2)...
        + 2 * this.padding) ./ this.stride) + 1;
      % Adjusting size for odd filter sizes
      assert(all(this.outputShape(1:2) > 0), 'ConvLayer:invalidShape', ...
        'Invalid output shape resulting from parameters'); 
      this.outputShape(3) = this.filterNum;
    end
    
    function initializeFcLayer(this)
      %initializeFcLayer Initializes the inner FC layer
      this.fcLayer = ann.layers.FcLayer(this.filterShape, this.filterNum, ...
        this.activation);
    end
    
    function P = extractPatches(~, X, oH, oW, fH, fW, sH, sW)
      %extractPatches Creates a 2D array from a 4D tensor in which on rows
      %will be stored all convolution patches of input
      % Inputs:
      %   - X: 4D tensor, whose 2nd and 3rd dimensions give patches
      %   - oH: output height
      %   - oW: output width
      %   - fH: filter height
      %   - fW: filter width
      %   - sH: stride on height
      %   - sW: stride on width
      [N, ~, ~, C] = size(X);  % Extracting channels and # of samples
      ppi = oH * oW;  % Patches per image
      % Initializing patches matrix
      P = zeros(N * oH * oW, fH * fW * C);
      % Extracting patches
      for h = 1:oH
        for w = 1:oW
          % Calculating patch ranges
          startH = (h - 1) * sH + 1;
          startW = (w - 1) * sW + 1;
          % Extracting the patch from all images in the batch
          patch = X(:, startH:startH+fH-1, startW:startW+fW-1, :);
          % Permuting patch to get correct values
          patch = permute(patch, [1, 3, 2, 4]);
          % Linearizing patch
          P((h-1) * oW + w:ppi:end, :) = patch(:, :);
        end
      end
    end
    
    function ldZ = linearizeDZ(~, dZ)
      %linearizeDZ Linearizes derivatives w.r.t. outputs
      ldZ = permute(dZ, [4, 3, 2, 1]);
      ldZ = ldZ(:, :).';
    end
    
    function Mout = reconstructShape(this, M)
      %reconstructOutput Recreates shape of CONV layer starting from
      %outputs or activations of inner fully-connected layer
      % Inputs:
      %   - M: output or activation matrix of FC layer
      % Outputs:
      %   - Mout: matrix M reshaped in order to have the right size
      
      % Caching values
      oH = this.outputShape(1); oW = this.outputShape(2);
      fN = this.filterNum;
      % Reshaping and permuting
      Mout = reshape(M.', fN, oW, oH, []);
      Mout = permute(Mout, [4, 3, 2, 1]);
    end
    
    function dX = calculateInputDerivatives(this, ldX)
      %calculateInputDerivatives Calculates derivatives of input using
      %derivatives w.r.t. activations of the layer
      %   Manages calculation of input derivatives for the layer starting
      %   with derivatives produced by the inner FC layer.
      % Inputs:
      %   - ldX: linearized derivatives w.r.t. input given by FC layer
      % Output:
      %   - dX: derivatives of the layer w.r.t. input
      
      % Caching values
      oH = this.outputShape(1); oW = this.outputShape(2); 
      fH = this.filterShape(1); fW = this.filterShape(2);
      sH = this.stride(1); sW = this.stride(2);
      pH = this.padding(1); pW = this.padding(2);
      xH = this.inputShape(1); xW = this.inputShape(2);
      C = this.inputShape(3); ppi = oH * oW;
      % Reshaping ldX in a 4D tensor
      ldX = reshape(ldX, [], fW, fH, C);
      ldX = permute(ldX, [1, 3, 2, 4]);
      % Obtaining real number of samples
      N = size(ldX, 1);
      N = N / ppi;
      % Initializing dX to zero (including padding)
      dX = zeros(N, xH + 2 * pH, xW + 2 * pW, C);
      % Summing patches
      for h = 1:oH
        for w = 1:oW
          % Calculating patch ranges
          startH = (h - 1) * sH + 1; startW = (w - 1) * sW + 1;
          rangeH = startH:startH+fH-1; rangeW = startW:startW+fW-1;
          % Adding the patch to derivative values
          dX(:, rangeH, rangeW, :) = dX(:, rangeH, rangeW, :) + ...
            ldX((h - 1) * oW + w:ppi:end, :, :, :);
        end
      end
      % Unpadding dX to recover its original shape
      dX = dX(:, 1+pH:end-pH, 1+pW:end-pW, :);
    end
  end
  
end

