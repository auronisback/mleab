% Trains a fully-connected neural network with 2 hidden layer and one
% output layer, using RProp as optimizer and Cross Entropy error function.

TRAIN_NUM = 5000;
TEST_NUM = 1000;
BATCH_SIZE = 128;
VALIDATION_SPLIT = 0.2;
EPOCHS = 100;

fprintf('Creating MNIST dataset with %d training samples and %d test samples...\n', ...
  TRAIN_NUM, TEST_NUM);
ds = mnist.MnistFactory.createDataset(TRAIN_NUM, TEST_NUM);
ds.normalize();
ds.flatten();
ds.shuffle();
ds.toCategoricalLabels();

errorFun = ann.errors.CrossEntropy;
optimizer = ann.optimizers.RProp(.5, 1.2, 0.0125);

fprintf('Creating a neural network:\n');
net = ann.NeuralNetwork({...
  ann.layers.FcLayer(ds.inputShape, 200, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(200, ds.labelShape, ann.activations.Softmax)
  }, errorFun ...
);
net.print();

fprintf('Training:\n');
fprintf(' - error: Cross-Entropy\n');
fprintf(' - optimizer: RProp (eta_min: 0.5, eta_max: 1.2, delta_0 = 0.0125)\n');
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
% Starts training
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
[errors, bestEpoch] = training.train(EPOCHS, net, ds);
fprintf('Best epoch: %d\n', bestEpoch);
[testErr, testAcc] = training.evaluateOnTestSet(net, ds);
fprintf('Test error: %.2f\nTest Accuracy: %.2f\n', ...
  testErr, testAcc * 100);
plotErrors(errors, bestEpoch);