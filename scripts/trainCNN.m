% Training script for a convolutional neural network, with one
% convolutional layer and a fully-connected softmax output layer. The
% training uses Cross-entropy as error function and RProp as optimizer.

TRAIN_NUM = 1000;
TEST_NUM = 200;
BATCH_SIZE = 128;
VALIDATION_SPLIT = 0.2;
EPOCHS = 500;

errorFun = ann.errors.CrossEntropy;
optimizer = ann.optimizers.RProp(.5, 1.2, .00125);

fprintf('Creating MNIST dataset with %d training samples and %d test samples...\n', ...
  TRAIN_NUM, TEST_NUM);
ds = mnist.MnistFactory.createDataset(TRAIN_NUM, TEST_NUM);
% Halving image size
ds.resize([14, 14]);
ds.normalize();
ds.shuffle();
ds.toCategoricalLabels();

fprintf('Creating a neural network:\n');
convLayer = ann.layers.ConvLayer(ds.inputShape, 16, [3, 3, 1], ...
  ann.activations.Relu());
flattenLayer = ann.layers.FlattenLayer(convLayer.outputShape);
fcLayer = ann.layers.FcLayer(flattenLayer.outputShape, ds.labelShape, ...
  ann.activations.Softmax);
net = ann.NeuralNetwork({ convLayer, flattenLayer, fcLayer }, errorFun);
net.print();

fprintf('Training:\n');
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
% Starts training
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
[errors, bestEpoch] = training.train(EPOCHS, net, ds);
[testErr, testAcc] = training.evaluateOnTestSet(net, ds);
% Printing and plotting errors
fprintf('Test error: %.2f\nTest Accuracy: %.2f\n', ...
  testErr, testAcc * 100);
plotErrors(errors, bestEpoch);