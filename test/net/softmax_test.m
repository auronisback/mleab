%Tests Softmax layers
%Author: Francesco Altiero
%Date: 22/12/2018

X = rand(20, 10);

net = neuralnet.NeuralNet(10);

hidden = neuralnet.layer.GenericLayer(10, 5);
hidden.actFun = neuralnet.activation.Sigmoid(hidden);
output = neuralnet.layer.SoftmaxLayer(5, 10);
net.addHiddenLayer(hidden);
net.setOutputLayer(output);

net.forward(X)