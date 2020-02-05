inputShape = [3, 3, 1];
numFilters = 1;  % Problem with more filters on dX
filterShape = [2, 2, 1];
stride = [1, 1];
padding = [0, 0];
N = 1;
activation = ann.activations.Identity();
fcConvL = ann.layers.FcConvEquivLayer(inputShape, numFilters, filterShape, ...
  activation, stride, padding);
convL = ann.layers.ConvLayer(inputShape, numFilters, filterShape, ...
  activation, stride, padding);
flatten = ann.layers.FlattenLayer(fcConvL.outputShape);
lout = ann.layers.FcLayer(fcConvL.outputShape, 1, ann.activations.Identity());
errorFun = ann.errors.SsError();

[fW, fb] = fcConvL.getParameters();
convL.setParameters(fW, fb);
[cW, cb] = convL.getParameters();
fcConvL.setParameters(cW, cb);
% Checking equivalence
[cW, cb] = fcConvL.getParameters();
[fW, fb] = convL.getParameters();
assert(all(cW == fW, 'all'), 'ERROR: Weights are not equal!');
assert(all(cb == fb, 'all'), 'ERROR: Biases are not equal!');

X = rand([N, inputShape]);
%X(2, :) = X(1, :);  % TODO: Remove after
T = rand([N, 1]);
fcZ = fcConvL.predict(X);
convZ = convL.predict(X);

fprintf('Checking same size: ');
assert(all(size(fcZ) == size(convZ), 'all'), 'ERROR: different result size!');
fprintf('Ok\n');
fprintf('Checking same activations: ');
assert(all(abs(fcConvL.A - convL.A) < 1e-10, 'all'), 'ERROR: different cached activation');
fprintf('Ok\n');
fprintf('Checking same outputs: ');
% Checking distance is lesser han a treshold
assert(all(abs(fcZ - convZ) < 1e-10 , 'all'), 'ERROR: different results!');
fprintf('Ok\n');

% Checking backpropagation
dZ = rand([N, convL.outputShape]);
[cdX, cdW, cdb] = convL.backward(dZ, X);
[fcdX, fcdW, fcdb] = fcConvL.backward(dZ, X);

fprintf('Checking same dWs: ');
assert(all(abs(cdW - fcdW) < 1e-10, 'all'), 'ERROR: different dWs');
fprintf('Ok\n');
fprintf('Checking same dbs: ');
assert(all(abs(cdb - fcdb) < 1e-10, 'all'), 'ERROR: different dbs');
fprintf('Ok\n');
fprintf('Checking same dXs: ');
assert(all(abs(cdX - fcdX) < 1e-10, 'all'), 'ERROR: different dXs');
fprintf('Ok\n');

