%Training on MNIST dataset with cross-entropy and softmax
%Author: Francesco Altiero
%Date: 11/12/2018

%Some constants
MAX_EPOCHS = 100;
ETA_MINUS = 0.5;
ETA_PLUS = 1.2;
MNIST_TS_PATTERNS = 'MNIST_dataset/train-images-idx3-ubyte';
MNIST_TS_LABELS = 'MNIST_dataset/train-labels-idx1-ubyte';
MNIST_TS_NUMBER = 200;

%Loading MNIST dataset
mnistDs = mnist.MnistFactory.loadFromFile(...
  MNIST_TS_PATTERNS, MNIST_TS_LABELS, MNIST_TS_NUMBER, 'classification');

%Initializing the net
patternSize = 28 * 28;
net = neuralnet.NeuralNet(patternSize);

hidden = neuralnet.layer.GenericLayer(patternSize, 100);
hidden.actFun = neuralnet.activation.Sigmoid(hidden);

output = neuralnet.layer.SoftmaxLayer(100, 10);

net.addHiddenLayer(hidden);
net.setOutputLayer(output);

clear hidden output;

%Initializing the training
train = neuralnet.train.BatchTraining(MAX_EPOCHS, ...
  neuralnet.train.error.CrossEntropy(), ...
  neuralnet.train.update.RProp(ETA_MINUS, ETA_PLUS) ...
);

fprintf('Training the net using batch training...\n');
errors = train.train(net, mnistDs);

fprintf('Plotting errors...\n');

size(errors.training)

plot(errors.training);
hold on;
plot(errors.validation);
legend({'Training Set', 'Validation Set'});
hold off;