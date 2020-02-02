classdef ConvLayer < ann.layers.Layer
  %CONVLAYER Convolutional layer used in an ANN
  %   Manages operations and properties for a convolutional layer in a
  %   neural network, allowing to set parameters as stride, padding, filter
  %   number and filter size.
  
  properties(SetAccess = private)
    filterNum;  % Number of filters in the layer
    filterShape;  % Shape of each filter
    stride = [1, 1];  % Vertical and horizontal stride in convolutions
    padding = [0, 0];  % Vertical and horizontal padding in convolutions
    F;  % Filters in the layer
    b;  % Biases array for each filter
  end
  
  methods
    function this = ConvLayer(inputShape, filterNum, filterShape, ...
        activation, stride, padding)
      %ConvLayer Construct an instance of this class
      %   Creates an object which performs 2-dimensional convolutions
      %   specifying convolution parameters.
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
      this.activation.setLayer(this);
      this.initializeFilters();
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
      this.name = 'Conv';
    end
    
    function [W, b] = getParameters(this)
      %getParameters Gets weights and biases of the layer
      %   Getter for weights and biases of the layer.
      % Outputs:
      %   - W: weights
      %   - b: biases
      W = this.F;
      b = this.b;
    end
    
    function setParameters(this, W, b)
      %setParameters Sets layer's parameters
      %   Setter for weights and biases of the layer.
      % Inputs:
      %   - W: weights, which should have same size of layer's weights
      %   - b: biases, which should have same size of layer's biases
      assert(all(size(W) == size(this.F)), 'FcLayer:invalidWeights', ...
        'Invalid weight size');
      assert(all(size(b) == size(this.b)), 'FcLayer:invalidBiases', ...
        'Invalid bias size');
      this.F = W;
      this.b = b;
    end
    
    
    function Z = predict(this, X)
      %predict Evaluates the convolutional layer on given input
      %   Performs convolutions between layer's filters and given set of
      %   images in order to produce layer's output.
      % Inputs:
      %   - X: layer's input
      % Output:
      %   - Z: output value for given input
      
      % Padding input
      padX = padarray(X, [0, this.padding, 0], 0, 'both');
      % Convoluting tensor
      A = this.convolve(padX, this.F, this.outputShape, this.stride);
      % Applying bias
      for k = 1:this.filterNum
        A(:, :, :, k) = A(:, :, :, k) + this.b(k);
      end
      Z = this.activation.eval(A);
    end
    
    function Z = forward(this, X)
      %forward Evaluates convolutional layer on given input
      %   Performs a forward pass of the Conv layer and caches output and
      %   activation values in order to use them in training.
      % Inputs:
      %   - X: layer's input
      % Outputs:
      %   - Z: output for given input
      
      % Padding input
      padX = padarray(X, [0, this.padding, 0], 0, 'both');
      % Convoluting tensor
      this.A = this.convolve(padX, this.F, this.outputShape, this.stride);
      % Applying bias
      for k = 1:this.filterNum
        this.A(k, :) = this.A(k, :) + this.b(k);
      end
      this.Z = this.activation.eval(this.A);
      % Assigning to output
      Z = this.Z;
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
      
      % Calculating derivatives w.r.t. activations
      dA = this.activation.derive(dZ);
      % Invoking backward method
      [dW, db] = this.backwardFromActivation(dA, X);
      % Calculating derivatives w.r.t. input
      dX = this.calculateInputDerivatives(dA, X);
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
      
      % Calculating derivatives w.r.t. activation
      if isa(errorFun, 'ann.errors.CrossEntropy') && ...
          (isa(this.activation, 'ann.activations.Sigmoid') || ...
           isa(this.activation, 'ann.activations.Softmax'))
         % Softmax or sigmoid layer with cross-entropy
        dA = this.Z - T;
      else
        % General calculation of derivative w.r.t. activation
        dY = errorFun.derive(this.Z, T);
        dA = this.activation.derive(dY);
      end
      % Invoking backpropagation method
      [dW, db] = this.backwardFromActivation(dA, X);
      % Calculating derivatives w.r.t. input
      dX = this.calculateInputDerivatives(dA, X);
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
      % Calculating derivatives w.r.t. activations
      dA = this.activation.derive(dZ);
      % Invoking backward method
      [dW, db] = this.backwardFromActivation(dA, X);
    end
    
    function updateParameters(this, deltaW, deltaB)
      %updateParameters Updates this FC-layer's parameters
      %   Updates parameters of the layer using delta values calculated by
      %   an optimizer.
      % Inputs:
      %   - deltaW: delta values for weights
      %   - deltaB: delta values for biases
      this.F = this.F + deltaW;
      this.b = this.b + deltaB;
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
    
    function initializeFilters(this)
      %initializeFilters Initializes weights and biases of all filters in
      %the layer, in order to make them uniform random in [-1, 1]
      this.F = 1 - 2 * rand([this.filterNum, this.filterShape]);
      this.b = 1 - 2 * rand(1, this.filterNum);
    end
    
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
              'ConvLayer:samePaddingEvenFilter', ...
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
      %initializeoutputShape Calculate the shape of the output using layer's
      %parameters
      this.outputShape = floor((this.inputShape(1:2) - this.filterShape(1:2)...
        + 2 * this.padding) ./ this.stride) + 1;
      assert(all(this.outputShape(1:2) > 0), 'ConvLayer:invalidShape', ...
        'Invalid output shape resulting from parameters'); 
      this.outputShape(3) = this.filterNum;
    end
   
    function [dW, db] = backwardFromActivation(this, dA, X)
      %backwardFromActivation Backpropagates once activation derivatives
      %have been calculated.
      
      % Caching useful values
      N = size(X, 1); fN = this.filterNum;
      sH = this.stride(1); sW = this.stride(2);
      pH = this.padding(1); pW = this.padding(2);
      oH = this.outputShape(1); oW = this.outputShape(2);
      fH = this.filterShape(1); fW = this.filterShape(2); C = size(X, 3);
      % Padding and transforming X in order to obtain convolution input
      padX = padarray(X, [0, this.padding, 0], 0, 'both');
      padX = permute(padX, [4, 2, 3, 1]);
      %TODO: remove commented lines
      %fprintf('N: %d, fN: %d, C: %d\n', N, fN, C);
      %size(padX)
      % Transforming dA for deconvolution
      %size(dA)
      decdA = zeros(N, (oH - 1) * sH + 1, (oW - 1) * sW + 1, fN);
      decdA(:, 1:sH:end, 1:sW:end, :) = dA;
      decdA = permute(decdA, [4, 2, 3, 1]);
      %size(decdA)
      dW = this.convolve(padX, decdA, [fH, fW, fN], [1, 1]);
      % Permuting dW in order to obtain its right dimension
      dW = permute(dW, [4, 2, 3, 1]);
      % Calculating biases
      db = squeeze(sum(dA, [1, 2, 3]));
      
    end
    
    function dX = calculateInputDerivatives(this, dA, X)
      %calculateInputDerivatives Calculates derivatives w.r.t. input
      dX = zeros(size(X));
      N = size(X, 1);
      H = size(dA, 2);  % Row elements of derivative
      W = size(dA, 3);  % Column elements of derivative
      fH = this.filterShape(1);
      fW = this.filterShape(2);
      for n = 1:N
        % Creating inputs using padding values
        dXi = padarray(zeros(this.inputShape), this.padding, 0, 'both');
        % Looping through derivative's shape
        for i = 1:H
          for j = 1:W
            for k = 1:this.filterNum
              % Extracting patch from filter
              Delta_ij = zeros(size(dXi));
              h = (i - 1) * this.stride(1) + 1;
              w = (j - 1) * this.stride(2) + 1;
              Delta_ij(h:h+fH-1, w:w+fW-1, :) = ...
                dA(n, i, j) .* reshape(this.F(k, :), this.filterShape);
              dXi = dXi + Delta_ij;
            end
%             fprintf('h: %d, w: %d, dXi:\n', h, w);
%             squeeze(dXi)
          end
        end
        % Unpadding
        dXi = dXi(1 + this.padding(1):end - this.padding(1), ...
          1 + this.padding(2):end - this.padding(2), :);
        dX(n, :) = dXi(:);
      end
      reshape(dX, size(X));
    end
    
    function H = convolve(~, X, F, outShape, stride)
      %convolve Convolves a 4D tensor with a 4D filter on 2nd to 4th
      %dimensions
      %   Given two 4D tensor, convolves input data with given filters.
      %   Input data should be a 4D tensor, whose first dimension
      %   represents the number of samples, while other dimensions are
      %   rows, columns and channels of the image. If any padding is used
      %   on input, it shall be already applied before calling this method.
      %   In a same manner, given filters should be a 4D tensor whose first
      %   dimension represents different filters and the others are the 
      %   shape of each filter.
      %   Result of convolution is a 4D tensor with N output samples. All 
      %   dimensions except the first have shape equal to the output shape
      %   given.
      % Inputs:
      %   - X: input which has to be convolved, as a 4d-tensot
      %   - F: filters used to convolve image, as a 4d-tensor
      %   - outShape: output shape, used to initialize output matrix and
      %     should be consistent with convolution output's size
      %   - stride: stride for the convolution
      % Output:
      %   - H: a 4D tensor with N samles on the first dimension; each
      %     sample has shape equal to the output shape given as input
      
      % Caching input and filter sizes
      N = size(X, 1);
      [~, fH, fW, C] = size(F);
      sH = stride(1); sW = stride(2);
      % Pre-allocating output
      H = zeros([N, outShape]);
      % Extracting patches across all images
      for h = 1:outShape(1)
        for w = 1:outShape(2)
          % Calculating patch ranges
          startH = (h - 1) * sH + 1;
          startW = (w - 1) * sW + 1;
          rangeH = startH:startH+fH - 1;
          rangeW = startW:startW+fW - 1;
          % Extracting patches for (h, w) in all samples
          patches = reshape(X(:, rangeH, rangeW, :), [], fH, fW, C);
          % Convolving patches with filters
          H(:, h, w, :) = patches(:, :) * F(:, :).';
        end
      end
    end
    
  end
  
end

