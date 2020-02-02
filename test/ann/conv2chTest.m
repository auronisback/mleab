% Second test: 2 channels, stride = 1, padding = 0
fprintf('Testing 2 channels convolutional layer:\n');
fprintf(' - input shape = [3, 3, 2]\n');
fprintf(' - filter shape = [2, 2, 2]\n');
fprintf(' - stride = [1, 1]\n');
fprintf(' - padding = [0, 0] (valid)\n');

fN = 133;
N = 153;
inputShape = [3, 3, 2];
filterShape = [2, 2, 2];
outputShape = [2, 2, fN];
stride = [1, 1];
padding = [0, 0];

X = zeros([N, inputShape]);
for n = 1:N
  X(n, :) = 1:prod(inputShape);
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

expY = zeros([N, outputShape]);
for n = 1:N
  expY(n, :) = repmat([357, 465, 393, 501], 1, fN);  % 2 channels
  %expY(n, :) = repmat([38, 68, 48, 78], 1, fN);  % 1 channel
  %expY(n, :) = repmat([52, 152, 72, 172], 1, fN);  % 1 ch s = [2, 2]
  %expY(n, :) = repmat([52, 102, 152, 202, 72, 122, 172, 222], 1, fN);  % 1ch s = [1, 2]
end

fprintf('Checking results... ');
Y = l.predict(X);
assert(all(abs(expY - Y) < 1e-10, 'all'), ...
  'ERROR: results from convolution are not equal to expected ones');
fprintf('OK\n');

dZ = zeros([N, outputShape]);
for f = 1:N
  dZ(f, :) = repmat([1, 3, 2, 4], 1, fN);
end
[dX, dW, db] = l.backward(dZ, X);

expdW = zeros([fN, filterShape]);
for f = 1:fN
  expdW(f, :) = [37, 67, 47, 77, 127, 157, 137, 167] .* N;
end
expdb = sum(dZ, [1, 2, 3]);

expdX = zeros([N, inputShape]);
for f = 1:N
  expdX(f, :) = [1, 4, 4, 6, 20, 16, 9, 24, 16, 5, 16, 12, 22, 60, 40, 21, 52, 32] .* fN;
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