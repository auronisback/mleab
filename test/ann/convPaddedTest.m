% Padding test: 1 channel, 1 stride, 1 padding
fN = 15;
N = 10;
inputShape = [3, 3, 1];
filterShape = [2, 2, 1];
outputShape = [4, 4, fN];
stride = [1, 1];
padding = [1, 1];

fprintf('Testing 1 channel convolutional layer:\n');
fprintf(' - N: %d, #filters: %d\n', N, fN);
fprintf(' - input shape = [ %s]\n', sprintf('%d ', inputShape));
fprintf(' - filter shape = [ %s]\n', sprintf('%d ', filterShape));
fprintf(' - output shape = [ %s]\n', sprintf('%d ', outputShape));
fprintf(' - stride = [ %s]\n', sprintf('%d ', stride));
fprintf(' - padding = [ %s]\n', sprintf('%d ', padding));

X = zeros([N, inputShape]);
for n = 1:N
  X(n, :) = repmat(1:prod(inputShape(1:2)), 1, inputShape(3));
end
X = permute(X, [1, 3, 2, 4]);

l = ann.layers.ConvLayer(inputShape, fN, filterShape, ...
  ann.activations.Identity(), stride, padding);
W = zeros([fN, filterShape]);
b = zeros([1, fN]);
for f = 1:fN
  W(f, :) = 1:prod(filterShape);
  b(f) = 1;
end
W = permute(W, [1, 3, 2, 4]);
l.setParameters(W, b);

fprintf('Predicting and checking results... ');
expY = zeros([N, outputShape]);
% Using built-in convolution for prediction
for n = 1:N
  for k = 1:fN
    for c = 1:inputShape(3)
      inX = padarray(X, [0, padding, 0], 0, 'both');
      inX = squeeze(inX(n, :, :, c));
      % Rotating weights (in order to have a real convolution)
      inW = squeeze(W(k, :, :, c));
      lW = inW(:);
      inW(:) = lW(end:-1:1);
      res = conv2(inX, inW, 'valid');
      expY(n, :, :, k) = squeeze(expY(n, :, :, k)) + res;
    end
    expY(n, :, :, k) = expY(n, :, :, k) + b(k);
  end
end
Y = l.predict(X);
assert(all(abs(expY - Y) < 1e-10, 'all'), ...
  'ERROR: results from convolution are not equal to expected ones');
fprintf('OK\n');

fprintf('Backpropagating:\n');
dZ = zeros([N, outputShape(2), outputShape(1), outputShape(3)]);
for n = 1:N
  dZ(n, :) = repmat(1:prod(outputShape(1:2)), 1, fN);
end
dZ = permute(dZ, [1, 3, 2, 4]);
[dX, dW, db] = l.backward(dZ, X);

expdW = zeros([fN, filterShape]);
for f = 1:fN
  expdW(f, :) = [573, 393, 528, 348] .* N;
end
expdb = sum(dZ, [1, 2, 3]);

expdX = zeros([N, inputShape]);
for f = 1:N
  expdX(f, :) = [26, 36, 46, 66, 76, 86, 106, 116, 126] .* fN;
end
expdX = permute(expdX, [1, 3, 2, 4]);

fprintf('Checking dW... ');
assert(all(abs(expdW - dW) < 1e-10, 'all'), ...
  'ERROR: invalid obtained dW');
fprintf('Ok\n');
fprintf('Checking db... ');
assert(all(abs(db - expdb) < 1e-10, 'all'), ...
  'ERROR: invalid obtained db');
fprintf('Ok\n');
fprintf('Checking dX... ');
assert(all(abs(expdX - dX) < 1e-10, 'all'), ...
  'ERROR: invalid obtained dX');
fprintf('Ok\n');
clear
