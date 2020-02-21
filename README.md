# mleab
Machine Learning course assignment for Università degli Studi di Napoli Federico II.

This repository contains packages and scripts developed in order to fulfill exam's assignment for the Machine Learning (mod.B) courses. It is implemented in Matlab language, using OOP paradigm. The code provides packages and scripts in order to create and train feed-forward layered _Artificial Neural Networks_ as specified in the assignment body.

It contains the following packages:
- dataset: defines classes used to model datasets on which a network is trained and evaluated.
- ann: defines a neural network class and a training class used to train _ANNs_. It contains four sub-packages:
   + layers: defines an abstract layer class and some layer typology, as _fully-connected_ and _convolutional_ layers.
   + activations: provides common activation function for layers, as _Sigmoid_ and _ReLU_.
   + errors: defines some error function used to evaluate and train a network.
   + optimizers: includes classes which manages the _optimization_ of network's parameters during training, defining rules of update for weights and biases.
- mnist: provides a factory class used to load _MNIST_ dataset of handwritten digits.

In _script_ folder are placed scripts used to repeat experiments realized by authors and commented in the report. In _test_ folder there are some test scripts used to ensure layers were correctly implemented.
