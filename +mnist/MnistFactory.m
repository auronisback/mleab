%Author: Francesco Altiero
%Date: 07/10/2018

classdef MnistFactory
  %MNISTFACTORY Creates MNIST dataset from file
  %   Manages the creation of the MNIST dataset as a Dataset object by
  %   reading it from file. Follows a static factory pattern.
  
  properties (Constant)
    METHOD_REGRESSION = 'regression' % Labels loaded for regression
    METHOD_CLASSIFICATION = 'classification' % Labels loaded for classification
  end
  
  methods(Static)
    function ds = loadFromFile(patternFile, labelFile, number, method)
      %loadFromFile Loads the MNIST dataset from given files
      %   Manages the loading of a number of the MNIST dataset patterns 
      %   and labels from specified files.
      %
      %   Inputs:
      %     - patternFile: the file in which are stored MNIST patterns
      %     - labelFile: the file in which are stored MNIST labels
      %     - number: the number of patterns and labels that will be loaded
      %     - method: specifies if labels have to be loaded as one single
      %         value (for regression) or in a 1-vs-all fashion (for
      %         classification). If not given, regression mode will be used
      %   Output:
      %     - dataset: the MNIST dataset, with the given number of patterns
      %         and labels loaded
      
      %Checking for patterns and label file existance
      assert(isfile(patternFile) && isfile(labelFile), ...
        'MnistFactory:FileNotFound', 'Pattern or label file not found');
      %Loading images and patterns
      patterns = mnist.MnistFactory.loadMnistImages(patternFile, number);
      labels = mnist.MnistFactory.loadMnistLabels(labelFile, number);
      %Checking if the classification method has been given
      if exist('method', 'var') && ...
          strcmp(method, mnist.MnistFactory.METHOD_CLASSIFICATION)
        labels = mnist.MnistFactory.setClassificationLabels(labels, number);
      end
      %Creating the dataset object
      patternSizes = size(patterns);
      patternSizes = patternSizes(2:end);
      labelSizes = size(labels);
      labelSizes = labelSizes(2:end);
      ds = dataset.Dataset(patternSizes, labelSizes);
      ds.setPatternsAndLabels(patterns, labels);
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
      %Normalizing images
      images = mnist.MnistFactory.normalize(images);
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

    function images = normalize(images)
      %normalizeImages Normalizes images given, obtaining values between 0
      %and 1.
      %
      %Inputs:
      % - images: the 3-dimensional matrix containing images
      %Output:
      % - image: the image matrix in which each element has been normalized
      %     due the division by 255 (max pixel value).
  
      images = double(images) / 255;
    end
    
    function labels = setClassificationLabels(l, number)
      %setClassificationLabels Sets the labels for classification
      %   Transforms labels from a regression format (1 output only) to a
      %   classification format (1 for each class).
      %
      %   Inputs: 
      %     - l: the labels, in a regression fashion
      %     - number: the number of examples
      %   Outputs:
      %     - labels: the labels in a 1-vs-all fashion
      
      labels = zeros(number, 10);
      for n = 1 : number
        labels(n, l(n) + 1) = 1;
      end
    end
    
  end
end

