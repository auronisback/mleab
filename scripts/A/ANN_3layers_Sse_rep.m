
% Statistical constants
NUM_REPETITIONS = 10;
FIRST_LAYER_NODES = 200;
SECOND_LAYER_NODES = FIRST_LAYER_NODES / 2;
OUTPUT_LAYER_NODES = SECOND_LAYER_NODES;

%INIT VARIABLES
TRAIN_NUM = 5000;
TEST_NUM = 1000;
VALIDATION_SPLIT = 0.2;
ETA = .01;
EPOCHS = 500;
BATCH_SIZE = 128;

%DataSet Definition
fprintf('Creating MNIST dataset with %d training samples and %d test samples...\n', ...
  TRAIN_NUM, TEST_NUM);
ds = mnist.MnistFactory.createDataset(TRAIN_NUM, TEST_NUM);
ds.normalize();
ds.flatten();
ds.shuffle();

%Error / Optimizer Definition
errorFun = ann.errors.SsError();
optimizer = ann.optimizers.Sgd(ETA);

%Neural Network Definition
fprintf('Creating a neural network with 2 hidden layer:\n');
net = ann.NeuralNetwork({...
  ann.layers.FcLayer(ds.inputShape, FIRST_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(FIRST_LAYER_NODES, SECOND_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(OUTPUT_LAYER_NODES, ds.labelShape, ann.activations.Identity)
}, errorFun);
net.print();

fprintf('Training:\n');
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);

% Starts repeated training
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
repeatTraining(net, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/A/ANN_3layers_Sse.xls');


% Doubling the number of nodes

fprintf('Creating a neural network with 2 hidden layer (doubled nodes):\n');
net = ann.NeuralNetwork({...
  ann.layers.FcLayer(ds.inputShape, 2 * FIRST_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(2 * FIRST_LAYER_NODES, 2 * SECOND_LAYER_NODES, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(2 * OUTPUT_LAYER_NODES, ds.labelShape, ann.activations.Identity)
}, errorFun);
net.print();

fprintf('Training:\n');
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);

% Starts repeated training
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
repeatTraining(net, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/A/ANN_3layers_Sse_doubled.xls');
