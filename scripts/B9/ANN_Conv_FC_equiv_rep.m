% Shows the equivalence in training and outputs of Fully-Connected
% convolutional equivalent layers with canonicl Convolutional layers.
% Experiments will be repeated and output spreadsheet will be produced in
% order to evaluate statistics on both kind of networks.

% Number of repetitions
NUM_REPETITIONS = 10;

% Training and dataset parameters
TRAIN_NUM = 2500;
TEST_NUM = 500;
BATCH_SIZE = 256;
VALIDATION_SPLIT = 0.2;
ETA = 0.01;
EPOCHS = 100;

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

% Training of FC equiv network
fprintf('Training Fully-Connected Equivalent Network:\n');
fcNet.print();
fprintf(' - error: %s\n', errorFun.toString());
fprintf(' - optimizer: %s\n', optimizer.toString());
fprintf(' - bacth size: %d\n', BATCH_SIZE);
fprintf(' - validation split factor: %.3f\n', VALIDATION_SPLIT);
fprintf('Training for %d epochs:\n', EPOCHS);
training = ann.Training(optimizer, BATCH_SIZE, VALIDATION_SPLIT);
repeatTraining(fcNet, ds, training, EPOCHS, NUM_REPETITIONS, ...
  'experiments/B9/FC_Conv_64.xls');