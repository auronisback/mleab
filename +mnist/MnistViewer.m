classdef MnistViewer
  %MNISTVIEWER Manages to show window in order to view MNIST images
  %   Class that manages the visualization of MNIST dataset pattern images
  %   on a window. It uses a detaset with MNIST patterns and labels and
  %   produces a view in which the desired number of patterns are shown.
  
  properties
    grayscale %True if images are shown in greyscale
    number %Number of images that are going to be shown
  end
  
  methods
    function this = MnistViewer(grayscale, number)
      %MNISTVIEWER Construct an instance of this class
      %   Creates the MNIST viewer object giving user's preferences.
      %
      %   Inputs:
      %     - grayscale: logical value for showing grayscaled or color
      %         patterns
      %     - number: number of patterns to show
      this.grayscale = logical(grayscale);
      this.number = number;
    end
    
    function show(this, dataset)
      %show Visualizes the given dataset
      %   Shows the window in which are renderized images on the dataset,
      %   using this object's preferences.
      %   
      %   Inputs:
      %     - dataset: a Dataset object, whose patterns and labels are
      %         going to be shown
      
      %Showing patterns
      figure('NumberTitle', 'off', 'Name', sprintf('MNIST: %d patterns', this.number));
      if this.grayscale
        colormap(gray);
      end
      perRow = ceil(sqrt(this.number));
      for n = 1:this.number
        subplot(perRow, perRow, n);
        axis off
        digit = reshape(dataset.patterns(n, :), dataset.patternSizes);
        imagesc(digit);
        title(sprintf('%d', dataset.labels(n, :)));
      end
    end
    
    function showClassifierOutputs(this, classifier, dataset)
      %showClassifierOutputs Shows the outputs of the classifier on dataset
      %   Uses the classifier in order to show prediction on the given
      %   dataset.
      %   
      %   Inputs:
      %     - classifier: the MNIST classifier object
      %     - dataset: the dataset used to show prediction
      
      %Showing patterns
      figure('NumberTitle', 'off', 'Name', sprintf('MNIST: %d patterns', this.number));
      if this.grayscale
        colormap(gray);
      end
      perRow = ceil(sqrt(this.number));
      for n = 1:this.number
        ax = subplot(perRow, perRow, n);
        ax.XTick = [];
        ax.YTick = [];
        digit = reshape(dataset.patterns(n, :), dataset.patternSizes);
        imagesc(digit);
        %Getting label and probability
        scores = classifier.classify(dataset.patterns(n, :));
        winner = scores{1};
        truth = dataset.labels(n, :);
        %Creating color
        if winner.label == truth
          color = 'green';
        else
          color = 'red';
        end
        %Printing title
        title(sprintf("T: %d\n\\color{%s}Y: %d (%.2f%%)", truth, ...
          color, winner.label, winner.probability * 100));
      end
    end
    
  end
end

