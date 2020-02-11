% Shows the equivalence in training and outputs of Fully-Connected
% convolutional equivalent layers with canonicl Convolutional layers.
% Experiments will be executed only once, in order to show plots of
% training with training and validation accuracy and loss.

TRAIN_NUM = 5000;
TEST_NUM = 1000;
BATCH_SIZE = 128;
VALIDATION_SPLIT = 0.2;
ETA = 0.01;
EPOCHS = 200;

fprintf('Creating MNIST dataset with %d training samples and %d test samples...\n', ...
  TRAIN_NUM, TEST_NUM);
ds = mnist.MnistFactory.createDataset(TRAIN_NUM, TEST_NUM);
ds.normalize();
ds.resize([14, 14]);
ds.shuffle();
ds.toCategoricalLabels();

errorFun = ann.errors.CrossEntropy();
optimizer = ann.optimizers.Sgd(ETA);

nF = 64;
fShape = [4, 4];
padding = [0, 0];
stride = [1, 1];
convL = ann.layers.ConvLayer(ds.inputShape, nF, fShape, ...
  ann.activations.Relu(), stride, padding);
fcL = ann.layers.FcConvEquivLayer(ds.inputShape, nF, fShape, ...
  ann.activations.Relu(), stride, padding);

% Creating Convolutional and FC equivalent with same parameters
convNet = ann.NeuralNetwork({convL, ...
  ann.layers.FlattenLayer(convL.outputShape), ...
  ann.layers.FcLayer(convL.outputShape, 200, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(200, ds.labelShape, ann.activations.Softmax)
}, errorFun);
fcNet = ann.NeuralNetwork({fcL, ...
  ann.layers.FlattenLayer(fcL.outputShape), ...
  ann.layers.FcLayer(fcL.outputShape, 200, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(200, ds.labelShape, ann.activations.Softmax)
}, errorFun);

% Equalizing both network's parameters
[W, b] = convNet.getParameters();
fcNet.setParameters(W, b);

% Training of Conv network
fprintf('Training Convolutional Network:\n');
convNet.print();
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
[errors, bestEpoch] = training.train(EPOCHS, convNet, ds);
[testErr, testAcc] = training.evaluateOnTestSet(convNet, ds);
fprintf('Test error: %.2f\nTest Accuracy: %.2f\n', ...
  testErr, testAcc * 100);
plotErrors(errors, bestEpoch, 'Convolutional Training');

% Training of FC equiv network
fprintf('Training Fully-Connected Equivalent Network:\n');
fcNet.print();
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
[errors, bestEpoch] = training.train(EPOCHS, fcNet, ds);
[testErr, testAcc] = training.evaluateOnTestSet(fcNet, ds);
fprintf('Test error: %.2f\nTest Accuracy: %.2f\n', ...
  testErr, testAcc * 100);
plotErrors(errors, bestEpoch, 'FC-equivalent Training');