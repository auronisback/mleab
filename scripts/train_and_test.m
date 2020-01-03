% Trains and tests the network on the MNIST dataset.
%
% Author: Francesco Altiero
% Date: 26/12/2018

% Dataset constants and filenames
TRAINING_NUMBER = 500;  % Number of training examples
TEST_NUMBER = 100;  % Number of test examples

% Training constants
MAX_EPOCHS = 300; % Epochs in training

% Error function for training
TRAIN_ERROR_FUNCTION = neuralnet.train.error.CrossEntropy();

% Weight update strategy
ETA = 0.01;           % Learning rate
WEIGHT_STRATEGY = neuralnet.train.update.DeltaRule(ETA);

% Stop criterion
STOP_CRITERION = neuralnet.train.criteria.NonStopCriterion();

% Dataset splitter
SPLIT_FACTOR = 0.25;  % Factor between size of TS and VS
DS_SPLITTER = neuralnet.train.splitter.FactorSplitter(SPLIT_FACTOR);

% Error function for validation 
ERROR_FUNCTION = TRAIN_ERROR_FUNCTION; %Same as the training

% Net parameters
HIDDEN_LAYER_NODES = 300; % Nodes for hidden layer
OUTPUT_LAYER_NODES = 10; % Output dimensionality

%Loading datasets
fprintf('Loading MNIST dataset... ');
mnistDs = mnist.MnistFactory.createDataset(TRAINING_NUMBER, TEST_NUMBER);
fprintf('Ok\n');

% Training method
fprintf('Creating a batch learning method... ');
training = neuralnet.train.BatchTraining(...
  MAX_EPOCHS, TRAIN_ERROR_FUNCTION, WEIGHT_STRATEGY, ...
  STOP_CRITERION, DS_SPLITTER, ERROR_FUNCTION ...
);
fprintf('Ok\n');

% Neural network
fprintf(['Creating a full-connected neural net with one hidden layer '...
  '(sigmoidal) and softmax output layer...']);
net = neuralnet.NeuralNet(prod(trainingSet.patternSizes));
hidden = neuralnet.layer.GenericLayer(prod(trainingSet.patternSizes), ...
  HIDDEN_LAYER_NODES);
hidden.actFun = neuralnet.activation.Sigmoid(hidden);
output = neuralnet.layer.SoftmaxLayer(HIDDEN_LAYER_NODES, OUTPUT_LAYER_NODES);
net.addHiddenLayer(hidden);
net.setOutputLayer(output);
fprintf('Ok\n');

% Training the net
fprintf('Training the net on %d examples... ', TRAINING_NUMBER);
errors = training.train(net, trainingSet);
fprintf('Done\nPlotting errors...\n');
plotErrors(errors);

% Creating the MNIST classifier object
fprintf('Showing the net on the test set... ');
classifier = mnist.MnistClassifier(net);
viewer = mnist.MnistViewer(true, 25);
viewer.showClassifierOutputs(classifier, testSet);
fprintf('Done.\n');

% Clearing previous constants
clear DS_SPLITTER ERROR_FUNCTION ETA MAX_EPOCHS SPLIT_FACTOR ...
  STOP_CRITERION TEST_LABELS_FNAME TEST_PATTERNS_FNAME ...
  TRAINING_LABELS_FNAME TRAINING_PATTERNS_FNAME WEIGHT_STRATEGY ...
  TRAIN_ERROR_FUNCTION TRAINING_NUMBER TEST_NUMBER HIDDEN_LAYER_NODES ...
  OUTPUT_LAYER_NODES;


