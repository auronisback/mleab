inputShape = [3, 3, 5];
outputShape = prod(inputShape, 'all');
N = 10;

l = ann.layers.FlattenLayer(inputShape);

X = rand([N, inputShape]);
expZ = X(:, :);
% Checking right output size
assert(all(l.outputShape == outputShape), 'ERROR: invalid output shape');

Z = l.forward(X);
assert(all(Z == expZ, 'all'), 'ERROR: invalid layer output');
[dX, dW, db] = l.backward(Z, X);
assert(all(dX == X, 'all'), 'ERROR: invalid dX');