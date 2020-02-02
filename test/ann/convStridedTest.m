
% Third test: 1 channel, stride = [1, 2], padding = 0
fprintf('Third test:\n');
fprintf(' - input shape = [5, 5, 1]\n');
fprintf(' - filter shape = [2, 2, 1]\n');
fprintf(' - stride = [1, 2]\n');
fprintf(' - padding = [0, 0] (valid)\n');

fN = 5;
N = 5;
inputShape = [5, 5, 1];
filterShape = [2, 2, 1];
outputShape = [4, 2, fN];
stride = [1, 2];
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

expY = zeros([fN, outputShape]);
for n = 1:N
  expY(n, :) = repmat([52, 102, 152, 202, 72, 122, 172, 222], 1, fN);  % 1ch s = [1, 2]
end

fprintf('Checking results... ');
Y = l.predict(X);
assert(all(abs(expY - Y) < 1e-10, 'all'), ...
  'ERROR: results from convolution are not equal to expected ones');
fprintf('OK\n');
clear