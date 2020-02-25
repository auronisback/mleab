% Shows the equivalence in training and outputs of Fully-Connected
% convolutional equivalent layers with canonicl Convolutional layers.
% Experiments will be repeated and output spreadsheet will be produced in
% order to evaluate statistics on both kind of networks.

% Number of repetitions
NUM_REPETITIONS = 10;

% Training and dataset parameters
TRAIN_NUM = 2500;
TEST_NUM = 500;
VALIDATION_SPLIT = 0.2;
BATCH_SIZE = TRAIN_NUM * (1 - VALIDATION_SPLIT);
EPOCHS = 50;

fprintf('Creating MNIST dataset with %d training samples and %d test samples...\n', ...
  TRAIN_NUM, TEST_NUM);
ds = mnist.MnistFactory.createDataset(TRAIN_NUM, TEST_NUM);
ds.normalize();
ds.resize([14, 14]);
ds.shuffle();
ds.toCategoricalLabels();

errorFun = ann.errors.CrossEntropy();
optimizer = ann.optimizers.RProp(.5, 1.2, .00125);

nF = 64;
fShape = [3, 3];
padding = [0, 0];
stride = [1, 1];
convL = ann.layers.ConvLayer(ds.inputShape, nF, fShape, ...
  ann.activations.Relu(), stride, padding);
fcConvL = ann.layers.FcConvLayer(ds.inputShape, nF, fShape, ...
  ann.activations.Relu(), stride, padding);
fcInnerL = ann.layers.ConvInnerFcLayer(ds.inputShape, nF, fShape, ...
  ann.activations.Relu(), stride, padding);

% Creating Convolutional and FC equivalent with same parameters
convNet = ann.NeuralNetwork({convL, ...
  ann.layers.FlattenLayer(convL.outputShape), ...
  ann.layers.FcLayer(convL.outputShape, 200, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(200, ds.labelShape, ann.activations.Softmax)
}, errorFun);
fcConvNet = ann.NeuralNetwork({fcConvL, ...
  ann.layers.FlattenLayer(fcInnerL.outputShape), ...
  ann.layers.FcLayer(fcInnerL.outputShape, 200, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(200, ds.labelShape, ann.activations.Softmax)
}, errorFun);
fcInnerNet = ann.NeuralNetwork({fcInnerL, ...
  ann.layers.FlattenLayer(fcInnerL.outputShape), ...
  ann.layers.FcLayer(fcInnerL.outputShape, 200, ann.activations.Sigmoid), ...
  ann.layers.FcLayer(200, ds.labelShape, ann.activations.Softmax)
}, errorFun);

% Training of Conv network
fprintf('Training Convolutional Network:\n');
convNet.print();
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
repeatTraining(convNet, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B9/Conv_64.xls');

% Training of Fc Conv equivalent network
% Training of Conv network
fprintf('Training FC Equivalent Network:\n');
fcConvNet.print();
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
repeatTraining(fcConvNet, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B9/FcConv_64.xls');

% Training of FC Inner equivalent network
fprintf('Training FC Equivalent Network (with inner FC layer):\n');
fcInnerNet.print();
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
repeatTraining(fcInnerNet, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B9/InnerConv_64.xls');