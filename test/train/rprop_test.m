%Tests training with cross-entropy
%Author: Francesco Altiero
%Date: 11/12/2018

%Some constants
MAX_EPOCHS = 100;
ETA_MIN = 0.5;
ETA_PLUS = 1.2;

%Stub dataset
X = rand(5, 3);
T = 1 - round(rand(5, 2));
ds = dataset.Dataset(size(X, 2), size(T, 2));
ds.setPatternsAndLabels(X, T);

%Initializing the net
net = neuralnet.NeuralNet(size(X, 1));

hidden = neuralnet.layer.GenericLayer(3, 5);
hidden.actFun = neuralnet.activation.Sigmoid(hidden);

output = neuralnet.layer.SoftmaxLayer(5, 2);

net.addHiddenLayer(hidden);
net.setOutputLayer(output);

clear hidden output;

%Initializing the training
train = neuralnet.train.BatchTraining(MAX_EPOCHS, ...
  neuralnet.train.error.CrossEntropy(), ...
  neuralnet.train.update.RProp(ETA_MIN, ETA_PLUS) ...
);

fprintf('Training the net using batch training...\n');
errors = train.train(net, ds);

fprintf('Plotting errors...\n');

plot(errors.training);
hold on;
plot(errors.validation);
legend({'Training Set', 'Validation Set'});