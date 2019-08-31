%Tests the MnistViewer class.
%Author: Francesco Altiero
%Date: 07/12/2018

MNIST_TS_PATTERNS = '../../MNIST_dataset/train-images-idx3-ubyte';
MNIST_TS_LABELS = '../../MNIST_dataset/train-labels-idx1-ubyte';
MNIST_TS_NUMBER = 200;

MNIST_VIEWER_NUMBER = 25;

mnistDs = mnist.MnistFactory.loadFromFile(...
  MNIST_TS_PATTERNS, MNIST_TS_LABELS, MNIST_TS_NUMBER);

clear MNIST_TS_PATTERNS MNIST_TS_LABELS MNIST_TS_NUMBER;

viewer = mnist.MnistViewer(true, MNIST_VIEWER_NUMBER);
viewer.show(mnistDs);