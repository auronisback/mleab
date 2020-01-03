%Tests the NeuralNet class
%Author: Francesco Altiero
%Date: 08/12/2018

X = rand(20, 10);

net = neuralnet.NeuralNet(10);

hidden = neuralnet.GenericLayer(10, 5);
hidden.actFun = neuralnet.activation.Sigmoid(hidden);
output = neuralnet.GenericLayer(5, 10);
net.addHiddenLayer(hidden);
net.setOutputLayer(output);

net.addHiddenLayer(neuralnet.GenericLayer(5, 5), 2);
net.addHiddenLayer(neuralnet.GenericLayer(20, 10), 1);
net.addHiddenLayer(neuralnet.GenericLayer(10, 4));

net.removeHiddenLayer(1);
net.removeHiddenLayer(2);
net.removeHiddenLayer(2);

net.forward(X)