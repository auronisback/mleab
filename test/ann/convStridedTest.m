% Convolutional layer test with stride > 1

fN = 1;
N = 1;
inputShape = [5, 5, 1];
filterShape = [3, 3, 1];
outputShape = [2, 3, fN];
stride = [2, 1];
padding = [0, 0];

fprintf('Testing 1 channel convolutional layer:\n');
fprintf(' - N: %d, #filters: %d\n', N, fN);
fprintf(' - input shape = [ %s]\n', sprintf('%d ', inputShape));
fprintf(' - filter shape = [ %s]\n', sprintf('%d ', filterShape));
fprintf(' - output shape = [ %s]\n', sprintf('%d ', outputShape));
fprintf(' - stride = [ %s]\n', sprintf('%d ', stride));
fprintf(' - padding = [ %s]\n', sprintf('%d ', padding));

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
  expY(n, :) = repmat([412, 862, 457, 907, 502, 952], 1, fN);
end

fprintf('Checking results... ');
Y = l.predict(X);
assert(all(abs(expY - Y) < 1e-10, 'all'), ...
  'ERROR: results from convolution are not equal to expected ones');
fprintf('OK\n');

fprintf('Backpropagating:\n');
dZ = zeros([N, outputShape]);
for f = 1:N
  dZ(f, :) = repmat([1, 4, 2, 5, 3, 6], 1, fN);
end
[dX, dW, db] = l.backward(dZ, X);

expdW = zeros([fN, filterShape]);
for f = 1:fN
  expdW(f, :) = [196, 301, 406, 217, 322, 427, 238, 343, 448] .* N;
end
expdb = sum(dZ, [1, 2, 3]);

expdX = zeros([N, inputShape]);
for f = 1:N
  expdX(f, :) = [1, 4, 11, 16, 28, 4, 13, 35, 40, 67, ...
    10, 28, 74, 73, 118, 12, 27, 69, 60, 93, 9, 18, 45, 36, 54] .* fN;
end

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