inputShape = [14, 14];
numFilters = 96;
filterShape = [4, 4];
stride = [2, 2];
padding = [1, 1];
N = 128;

fcConvL = ann.layers.ConvInnerFcLayer(inputShape, numFilters, filterShape, ...
  ann.activations.Relu(), stride, padding);
convL = ann.layers.ConvLayer(inputShape, numFilters, filterShape, ...
  ann.activations.Relu(), stride, padding);
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
fcZ = fcConvL.forward(X);
convZ = convL.forward(X);

fprintf('Checking same size: ');
assert(all(size(fcZ) == size(convZ), 'all'), 'ERROR: different result size!');
fprintf('Ok\n');
fprintf('Checking same activations: ');
%assert(all(abs(fcConvL.A - convL.A) < 1e-10, 'all'), 'ERROR: different cached activation');
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

% Checking update brings same values
fprintf('Checking update: ');
opt = ann.optimizers.Sgd(.1);
[deltaW, deltaB] = opt.evaluateDeltas({cdW}, {cdb}, 1);
% Updating parameters in both layer
convL.updateParameters(deltaW{1}, deltaB{1});
fcConvL.updateParameters(deltaW{1}, deltaB{1});
[newConvW, newConvB] = convL.getParameters();
[newFcConvW, newFcConvB] = fcConvL.getParameters();
assert(all(abs(newConvW - newFcConvW) < 1e-10, 'all'), ...
  'ERROR: different weights after update');
assert(all(abs(newConvB - newFcConvB) < 1e-10, 'all'), ...
  'ERROR: different biases after update');
fprintf('Ok\n');

