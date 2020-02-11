
% Statistical constants
NUM_REPETITIONS = 10;
FIRST_LAYER_NODES = 200;
SECOND_LAYER_NODES = FIRST_LAYER_NODES / 2;
THIRD_LAYER_NODES = SECOND_LAYER_NODES / 2;
OUTPUT_LAYER_NODES = THIRD_LAYER_NODES;

%INIT VARIABLES
TRAIN_NUM = 5000;
TEST_NUM = 1000;
VALIDATION_SPLIT = 0.2;
ETA = .01;
EPOCHS = 500;
BATCH_SIZE = TRAIN_NUM * (1 - VALIDATION_SPLIT);  % Full-batch

%DataSet Definition
fprintf('Creating MNIST dataset with %d training samples and %d test samples...\n', ...
  TRAIN_NUM, TEST_NUM);
ds = mnist.MnistFactory.createDataset(TRAIN_NUM, TEST_NUM);
ds.normalize();
ds.flatten();
ds.shuffle();
ds.toCategoricalLabels();  % Converting labels in categorical form

%Error / Optimizer Definition
errorFun = ann.errors.CrossEntropy();
optimizer = ann.optimizers.RProp(0.5, 1.2, 0.0125);

%Training Object Definition
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);

%Neural Network Definition: 
fprintf('Creating a neural network with 1 hidden layer and Sigmoid:\n');
fprintf('Training:\n');
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
net = ann.NeuralNetwork({...
  ann.layers.FcLayer(ds.inputShape, FIRST_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(FIRST_LAYER_NODES, ds.labelShape, ann.activations.Softmax)
}, errorFun);
net.print();
% Starts training
repeatTraining(net, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B7/ANN/SGD_SMAX_1layers.xls');

%Neural Network Definition: 
fprintf('Creating a neural network with 2 hidden layer and Sigmoid:\n');
fprintf('Training:\n');
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
net = ann.NeuralNetwork({...
  ann.layers.FcLayer(ds.inputShape, FIRST_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(FIRST_LAYER_NODES, SECOND_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(SECOND_LAYER_NODES, ds.labelShape, ann.activations.Softmax)
}, errorFun);
net.print();
% Starts training
repeatTraining(net, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B7/ANN/SGD_SMAX_2layers.xls');

%Neural Network Definition: 
fprintf('Creating a neural network with 3 hidden layer and Sigmoid:\n');
fprintf('Training:\n');
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
net = ann.NeuralNetwork({...
  ann.layers.FcLayer(ds.inputShape, FIRST_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(FIRST_LAYER_NODES, SECOND_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(SECOND_LAYER_NODES, THIRD_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(OUTPUT_LAYER_NODES, ds.labelShape, ann.activations.Softmax)
}, errorFun);
net.print();
% Starts training
repeatTraining(net, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B7/ANN/SGD_SMAX_3layers.xls');


%{
[errors, bestEpoch] = training.train(EPOCHS, net, ds);
[testErr, testAcc] = training.evaluateOnTestSet(net, ds);
fprintf('Best epoch: %d\n', bestEpoch);
fprintf('Test error: %.2f\nTest Accuracy: %.2f\n', ...
  testErr, testAcc * 100);
plotErrors(errors, bestEpoch, 'Two hidden Layers');
%}