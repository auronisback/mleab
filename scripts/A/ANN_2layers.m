
%INIT VARIABLES
TRAIN_NUM = 5000;
TEST_NUM = 1000;
BATCH_SIZE = 128;
VALIDATION_SPLIT = 0.2;
ETA = .01;
EPOCHS = 500;

fprintf('Creating MNIST dataset with %d training samples and %d test samples...\n', ...
  TRAIN_NUM, TEST_NUM);
ds = mnist.MnistFactory.createDataset(TRAIN_NUM, TEST_NUM);
ds.normalize();
ds.flatten();
ds.shuffle();

errorFun = ann.errors.SsError();
optimizer = ann.optimizers.Sgd(ETA);

fprintf('Creating a neural network with 1 hidden layer:\n');
net = ann.NeuralNetwork({...
  ann.layers.FcLayer(ds.inputShape, 200, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(200, ds.labelShape, ann.activations.Identity)
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
fprintf('Test error: %.2f\nTest Accuracy: %.2f\n', ...
  testErr, testAcc * 100);
fprintf('Best epoch: %d\n', bestEpoch);
plotErrors(errors, bestEpoch, 'One hidden Layer');
