%Tests generic layers
%Author: Francesco Altiero
%Date: 08/12/2018

X = ones(5, 3);

layer = neuralnet.GenericLayer(3, 4);

layer.forward(X);
