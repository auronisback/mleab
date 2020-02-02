TRAIN_NUM = 500;
TEST_NUM = 50;
BATCH_SIZE = 128;
VALIDATION_SPLIT = 0.2;
EPOCHS = 100;

ds = mnist.MnistFactory.createDataset(TRAIN_NUM, TEST_NUM);
ds.resize([7, 7]);
ds.normalize();
ds.toCategoricalLabels();
inputShape = ds.inputShape;

l = ann.layers.ConvLayer(inputShape, 5, [3, 3], ann.activations.Relu(), [2, 2], [1, 1]);
l2 = ann.layers.ConvLayer(l.outputShape, 3, [2, 2, 5], ann.activations.Relu());

network = ann.NeuralNetwork({l, l2, ...
  ann.layers.FlattenLayer(l2.outputShape), ...
  ann.layers.FcLayer(prod(l2.outputShape, 'all'), 200, ann.activations.Sigmoid()), ...
  ann.layers.FcLayer(200, ds.labelShape, ann.activations.Softmax())
}, ann.errors.CrossEntropy());

opt = ann.optimizers.RProp(.5, 1.2, .00125);
%opt = ann.optimizers.Sgd(0.01);

training = ann.Training(opt, BATCH_SIZE, .2);
errors = training.train(EPOCHS, network, ds);

% Extracting some elements from test set in order to evaluate
[X, T] = ds.getTrainingSet();
X = reshape(X(1:10, :), [10, ds.inputShape]);
T = reshape(T(1:10, :), [10, ds.labelShape]);
Y = network.predict(X)
[~, Yout] = max(Y);
[~, Tout] = max(T);
Yout == Tout
