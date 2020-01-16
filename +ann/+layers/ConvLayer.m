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
      this.name = sprintf('Conv (%d)', this.filterNum);
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
      
      N = size(X, 1);
      % Initializing activation values
      A = zeros([N, this.outputShape]);
      % Convolution over all images
      for n = 1:N
        padImg = padarray(reshape(X(n, :), this.inputShape), ...
          this.padding, 0, 'both');
        % Convolution over all filters
        for k = 1:this.filterNum
          filter = reshape(this.F(k, :), this.filterShape);
          H = this.convolve(padImg, filter, this.outputShape(1:end - 1));
          A(n, :, :, k) = H;
        end
      end
      A = reshape(A, [N, this.outputShape]);
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
      N = size(X, 1);
      % Initializing activation values
      this.A = zeros([N, this.outputShape]);
      % Convolution over all images
      for n = 1:N
        padImg = padarray(reshape(X(n, :), this.inputShape), ...
          this.padding, 0, 'both');
        % Convolution over all filters
        for k = 1:this.filterNum
          filter = reshape(this.F(k, :), this.filterShape);
          H = this.convolve(padImg, filter, this.outputShape(1:end - 1));
          this.A(n, :, :, k) = H + this.b(k);
        end
      end
      this.A = reshape(this.A, [N, this.outputShape]);
      this.Z = this.activation.eval(this.A);
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
        + 2 * this.padding) ./ this.stride);
      % Adjusting size for odd filter sizes
      this.outputShape(mod(this.filterShape(1:2), 2) == 1) = ...
        this.outputShape(mod(this.filterShape(1:2), 2) == 1) + 1;
      assert(all(this.outputShape(1:2) > 0), 'ConvLayer:invalidShape', ...
        'Invalid output shape resulting from parameters'); 
      this.outputShape(3) = this.filterNum;
    end
    
    function H = convolve(this, img, filter, shape)
      %convolve Convolves a single image with a given filter
      %   Performs a convolution operation of a single input image with a
      %   given filter.
      % Inputs:
      %   - img: the image which has to be convolved
      %   - filter: filter used to convolve image
      %   - shape: output shape, used for initializing output matrix
      % Output:
      %   - H: the image convolved with all filters in the layer
      H = zeros(shape);
      fShape = size(filter);
      for h = 1:this.stride(1):shape(1)
        for w = 1:this.stride(2):shape(2)
          iSlice = img(h : h + fShape(1) - 1, ...
            w : w + fShape(2) - 1, :);
          H(h, w) = sum(filter .* iSlice, 'all');
        end
      end
    end
    
    function [dW, db] = backwardFromActivation(this, dA, X)
      %backwardFromActivation Backpropagates once activation derivatives
      %have been calculated.
      % Caching size
      N = size(dA, 1);
      % Calculating derivatives for weights and biases
      dW = zeros([this.filterNum, this.filterShape]);
      db = zeros([1, this.filterNum]);
      for n = 1:N
        % Extracting input and applying padding
        padImg = padarray(reshape(X(n, :), this.inputShape), ...
          this.padding, 0, 'both');
        % Convolution over all filters
        for k = 1:this.filterNum
          % Extracting derivatives of current filter (repeating matrix on
          % 3rd dimension)
          df = reshape(dA(n, :, :, k), this.outputShape(1:2));
          % Convolving each channel
          for c = 1:this.filterShape(3)
            H = this.convolve(padImg(:, :, c), df, this.filterShape(1:2));
            % Convolving
            dW(k, :, :, c) = squeeze(dW(k, :, :, c)) + H;
          end
        end
        % Calculating biases
        db = db + sum(dA(n, :));
      end
      % Reshaping derivatives w.r.t. weights
      dW = reshape(dW, [this.filterNum, this.filterShape]);
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
              % minus 1
              Delta_ij(h:h+fH-1, w:w+fW-1, :) = ...
                dA(i, j) .* reshape(this.F(k, :), this.filterShape);
              dXi = dXi + Delta_ij;
            end
          end
        end
        % Unpadding
        dXi = dXi(1 + this.padding(1):end - this.padding(1), ...
          1 + this.padding(2):end - this.padding(2), :);
        dX(n, :) = dXi(:);
      end
      reshape(dX, size(X));
    end
    
  end
  
end

