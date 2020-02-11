
%INIT VARIABLES
TRAIN_NUM = 2000;
TEST_NUM = 100;
BATCH_SIZE = 1600;
VALIDATION_SPLIT = 0.2;
ETA = .01;
EPOCHS = 50;

fprintf('Creating MNIST dataset with %d training samples and %d test samples...\n', ...
  TRAIN_NUM, TEST_NUM);
ds = mnist.MnistFactory.createDataset(TRAIN_NUM, TEST_NUM);
ds.normalize();
ds.flatten();
ds.shuffle();
ds.toCategoricalLabels();  % Converting labels in categorical form

errorFun = ann.errors.CrossEntropy();
optimizer = ann.optimizers.RProp(0.5, 1.2, 0.0125);

% 2 hidden layers
fprintf('Creating a neural network with 2 hidden layer:\n');
net = ann.NeuralNetwork({...
  ann.layers.FcLayer(ds.inputShape, 200, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(200, 100, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(100, ds.labelShape, ann.activations.Softmax)
}, errorFun);
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
fprintf('Best epoch: %d\n', bestEpoch);
fprintf('Test error: %.2f\nTest Accuracy: %.2f\n', ...
  testErr, testAcc * 100);
plotErrors(errors, bestEpoch, 'Two hidden Layers');