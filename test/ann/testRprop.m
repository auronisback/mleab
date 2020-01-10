% Trains a fully-connected neural network with 2 hidden layer and one
% output layer, using SGD as optimizer.

TRAIN_NUM = 2000;
TEST_NUM = 100;
BATCH_SIZE = 64;
VALIDATION_SPLIT = 0;
EPOCHS = 50;

trainX = [1; 2; 4; 3; 6];
trainT = [0, 1; 1, 0; 1, 0; 0, 1; 1, 0];
%trainT = [0; 1; 1; 0; 1];
testX = 5;
%testT = [0, 1];
testT = 1;

ds = dataset.Dataset(trainX, trainT, testX, testT, ['1', '2', '3']);

errorFun = ann.errors.CrossEntropy();

net = ann.NeuralNetwork({...
  ann.layers.FcLayer(ds.inputShape, ds.labelShape, ann.activations.Softmax)
  }, errorFun ...
);

training = ann.Training(ann.optimizers.RProp(.5, 1.2), ...
  BATCH_SIZE, VALIDATION_SPLIT);
errors = training.train(EPOCHS, net, ds);