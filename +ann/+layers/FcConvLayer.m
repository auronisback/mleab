classdef FcConvLayer < ann.layers.ConvLayer
  %FCCONVLAYER Convolutional equivalent layer which performs convolutions
  %using matrix multiplication, as it was a FC layer
  %   Convolutional layer which performs convolution operations using the
  %   fully-connected equivalent approach defined in [Ma et Lu, 2017].
  
  methods
    function this = FcConvLayer(inputShape, filterNum, filterShape, ...
        activation, stride, padding)
      %FcConvLayer Construct an instance of this class
      %   Creates an object which performs 2-dimensional convolutions
      %   on 4D tensors, using a Fully-Connected approach.
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
      
      % Calling parent's constructor
      if ~exist('stride', 'var')
        stride = [1, 1];
      end
      if ~exist('padding', 'var')
        padding = [0, 0];
      end
      this = this@ann.layers.ConvLayer(inputShape, filterNum, ...
        filterShape, activation, stride, padding);
      % Renaming
      this.name = 'FcConv';
    end
  end
  
  methods(Access = protected)
    function H = convolve(this, X, F, stride)
      %convolve Convolves a 4D tensor with a 4D filter on 2nd to 4th
      %dimensions, using fully-connected approach.
      %   Given two 4D tensor, convolves input data with given filters.
      %   Input data should be a 4D tensor, whose first dimension
      %   represents the number of samples, while other dimensions are
      %   rows, columns and channels of the image. If any padding is used
      %   on input, it shall be already applied before calling this method.
      %   In a same manner, given filters should be a 4D tensor whose first
      %   dimension represents different filters and the others are the 
      %   shape of each filter.
      %   Result of convolution is a 4D tensor with N output samples.
      %   The convolution is implemented converting input and filter in a
      %   2D matrix and calculates outputs by matrix multiplication.
      %   Results will then be reshaped to their actual 4D shape.
      % Inputs:
      %   - X: input which has to be convolved, as a 4D-tensot
      %   - F: filters used to convolve image, as a 4D-tensor
      %   - stride: stride for the convolution
      % Output:
      %   - H: a 4D tensor with N samles on the first dimension; each
      %     sample has shape equal to the output shape given as input
      
      % Caching input and filter sizes
      [N, xH, xW, C] = size(X);
      [fN, fH, fW, ~] = size(F);
      sH = stride(1); sW = stride(2);
      % Asking for the output shape (padding is 0 as it was already been
      % applied to the input)
      [oH, oW, ~] = ann.layers.ConvLayer.getOutputShape([xH, xW, C], ...
        [fH, fW, C], fN, [sH, sW], [0, 0]);
      % Stretching input image
      X = this.stretchInput(X, oH, oW, fH, fW, sH, sW);
      % Stretching filters
      F = this.stretchFilters(F);
      % Executing a matrix multiplication
      H = X * F.';
      % Reshaping result
      H = permute(reshape(H.', fN, oW, oH, N), [4, 3, 2, 1]);
    end
  end
  
  methods(Access = private)
    function lX = stretchInput(~, X, oH, oW, fH, fW, sH, sW)
      %stretchInput Extract all patch from the input and creates a 2D
      %matrix used to perform convolution with matrix multiplication.
      % Inputs:
      %   - X: convolution input
      %   - oH: output height
      %   - oW: output width
      %   - fH: filters' height
      %   - fW: filters' width
      %   - sH: stride on height
      %   - sW: stride on width
      % Output:
      %   - lX: stretched input
      
      [N, ~, ~, C] = size(X);  % Extracting channels and # of samples
      ppi = oH * oW;  % Patches-per-image
      % Initialize patches
      lX = zeros(N * oH * oW, fH * fW * C);
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
          lX((h-1) * oW + w:ppi:end, :) = patch(:, :);
        end
      end
    end
    
    function F = stretchFilters(~, F)
      %stretchFilters Stretch a 4D tensor in a 2D matrix used to calculate
      %convolutions with a matrix multiplication
      % Input:
      %   - F: convolution filters
      % Output:
      %   - F: stretched filters
      
      % In this case, just a permutation is used
      F = permute(F, [1, 3, 2, 4]);
      F = F(:, :);
    end
  end
end

