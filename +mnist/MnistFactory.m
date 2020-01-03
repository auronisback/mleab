%Author: Francesco Altiero
%Date: 07/10/2018

classdef MnistFactory
  %MNISTFACTORY Creates MNIST dataset from file
  %   Manages the creation of the MNIST dataset as a Dataset object by
  %   reading it from file. Follows a static factory pattern.
  
  properties (Constant)
    FILE_TRAIN_IMAGES = './+mnist/res/train-images-idx3-ubyte';
    FILE_TRAIN_LABELS = './+mnist/res/train-labels-idx1-ubyte';
    FILE_TEST_IMAGES = './+mnist/res/t10k-images-idx3-ubyte';
    FILE_TEST_LABELS = './+mnist/res/t10k-labels-idx1-ubyte';
  end
  
  methods(Static)
    function ds = createDataset(num_train, num_test)
      %createDataset Loads data from MNIST dataset
      %   Manages the loading of a number of the MNIST dataset patterns 
      %   and labels.
      %
      %   Inputs:
      %     - num_train: number of images and labels for the training set
      %     - num_test: number of images and labels for the test set
      %   Output:
      %     - dataset: the MNIST dataset, with the given number of patterns
      %         and labels loaded
      
      %Checking arguments
      assert(num_train >= 1, ['Invalid number of training examples: ', ...
        num2str(num_train)]);
      assert(num_test >= 1, ['Invalid number of test examples: ', ...
        num2str(num_test)]);
      %Loading images and labels for training and test
      train_images = mnist.MnistFactory.loadMnistImages(...
        mnist.MnistFactory.FILE_TRAIN_IMAGES, num_train);
      train_labels = mnist.MnistFactory.loadMnistLabels(...
        mnist.MnistFactory.FILE_TRAIN_LABELS, num_train);
      test_images = mnist.MnistFactory.loadMnistImages(...
        mnist.MnistFactory.FILE_TEST_IMAGES, num_train);
      test_labels = mnist.MnistFactory.loadMnistLabels(...
        mnist.MnistFactory.FILE_TEST_LABELS, num_train);
      ds = dataset.Dataset(train_images, train_labels, test_images, ...
        test_labels, ["0", '1', '2', '3', '4', '5', '6', '7', '8', '9']);
    end
  end
    
  methods (Static, Access = private)
    function images = loadMnistImages(file, number)
      %loadMnistImages Loads the specified number of images from the given
      %MNIST image file.
      %
      % Inputs:
      % - file: the file from where read images
      % - number: the number of images that have to be read
      % Output:
      % - image: the image matrix
  
      fp = fopen(file, 'r', 'b'); %Opening the file in big-endian
      %Checking file
      assert(fp ~= -1, ['Unable to open file "', file, '"']);
      %Reading magic number
      magic = fread(fp, 1, 'int32', 0, 'b');
      assert(magic == 2051, ['Images: Invalid magic number: "', num2str(magic), '"']);
      %Reading total number of images, rows and columns
      total = fread(fp, 1, 'int32', 'b');
      rows = fread(fp, 1, 'int32', 'b');
      cols = fread(fp, 1, 'int32', 'b');
      %Adjusting number of requested images
      number = min(total, number);
      %Reading images
      images = fread(fp, number * rows * cols, 'uchar');
      %Reshaping
      images = reshape(images, [cols, rows, number]);
      images = permute(images, [3, 2, 1]);
      %Closing file
      fclose(fp);
    end

    function labels = loadMnistLabels(file, number)
      %loadMnistLabels Loads the specified number of labels from the given
      %MNIST label file.
      %
      % Inputs:
      % - file: the file from where read images
      % - number: the number of images that have to be read
      % Output:
      % - image: the labels array, in which each element is a number related to
      %     the digit the related image represents
     
      fp = fopen(file, 'r', 'b'); %Opening the file in big-endian
  
      %Checking file
      assert(fp ~= -1, ['Unable to open file "', file, '"']);
      %Reading magic number
      magic = fread(fp, 1, 'int32', 0, 'b');
      assert(magic == 2049, ['Labels: Invalid magic number: "', num2str(magic), '"']);
      %Reading total number of images in the label file
      total = fread(fp, 1, 'int32', 'b');
      %Adjusting number if needed
      number = min(number, total);
      %Creating labels
      labels = fread(fp, number, 'uchar');
    end
    
  end
end

