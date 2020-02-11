% Training of an ANN with 1, 2 and 3 hidden layers with ReLU activation
% function. Output is a softmax layer and cross-entropy is used as error.
% Experiments are repeated in order to obtain statistical data on training
% and produced models.

% Statistical constants
NUM_REPETITIONS = 10;

% Layers hyperparameters
FIRST_NODENUM = 300;  % Number of nodes in the first layer
SECOND_NODENUM = 150;  % Number of nodes in the second layer
THIRD_NODENUM = 75;  % Number of nodes in the third layer

% Dataset's number of samples
NUM_TRAINING = 5000;
NUM_TEST = 1000;

% Training hyper-parameters
EPOCHS = 50;
ETA_PLUS = 1.2;
ETA_MINUS = 0.5;
DELTA_ZERO = 0.0125;
VALIDATION_SPLIT = 0.2;
BATCH_SIZE = NUM_TRAINING * (1 - VALIDATION_SPLIT);  % Full-batch

fprintf('Loading %d training and %d test samples from MNIST... ', ...
  NUM_TRAINING, NUM_TEST);
ds = mnist.MnistFactory.createDataset(NUM_TRAINING, NUM_TEST);
% Normalizing and shuffling dataset
ds.flatten();
ds.normalize();
ds.shuffle();
ds.toCategoricalLabels();
% Caching samples and labels shapes
sampleShape = ds.inputShape;
labelShape = ds.labelShape;
fprintf('Ok\n');

% Creating training object
optimizer = ann.optimizers.RProp(ETA_MINUS, ETA_PLUS, DELTA_ZERO);
errorFun = ann.errors.CrossEntropy();
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);

fprintf('Training a network with 1 hidden layer and ReLU:\n');
fprintf(' - epochs: %d\n', EPOCHS);
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.2f\n', VALIDATION_SPLIT);
net = ann.NeuralNetwork({
  ann.layers.FcLayer(sampleShape, FIRST_NODENUM, ann.activations.Relu()),...
  ann.layers.FcLayer(FIRST_NODENUM, labelShape, ann.activations.Softmax())
}, errorFun);
repeatTraining(net, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B7/ReLU_1Layer.xls');

fprintf('Training a network with 2 hidden layer and ReLU:\n');
fprintf(' - epochs: %d\n', EPOCHS);
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.2f\n', VALIDATION_SPLIT);
net = ann.NeuralNetwork({
  ann.layers.FcLayer(sampleShape, FIRST_NODENUM, ann.activations.Relu()),...
  ann.layers.FcLayer(FIRST_NODENUM, SECOND_NODENUM, ann.activations.Relu()), ...
  ann.layers.FcLayer(SECOND_NODENUM, labelShape, ann.activations.Softmax())
}, errorFun);
repeatTraining(net, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B7/ReLU_2Layer.xls');

fprintf('Training a network with 3 hidden layer and ReLU:\n');
fprintf(' - epochs: %d\n', EPOCHS);
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.2f\n', VALIDATION_SPLIT);
net = ann.NeuralNetwork({
  ann.layers.FcLayer(sampleShape, FIRST_NODENUM, ann.activations.Relu()),...
  ann.layers.FcLayer(FIRST_NODENUM, SECOND_NODENUM, ann.activations.Relu()), ...
  ann.layers.FcLayer(SECOND_NODENUM, THIRD_NODENUM, ann.activations.Relu()), ...
  ann.layers.FcLayer(THIRD_NODENUM, labelShape, ann.activations.Softmax())
}, errorFun);
repeatTraining(net, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B7/ReLU_3Layer.xls');
