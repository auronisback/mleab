%Tests the dataset class
%Author: Francesco Altiero
%Date: 07/12/2018

try
  %Failing, empty sizes for patterns
  disp('Test 1: giving empty array for pattern sizes...');
  d = dataset.Dataset([], 1);
  %If we stay here, the test failed
  disp('FAILURE: no error has been thrown');
catch ex
  if strcmp(ex.identifier, 'Dataset:invalidSize')
    disp('SUCCESS: Invalid size correctly thrown');
  else
    fprintf('FAILURE: unexpected error: %s\n%s\n', ...
      ex.identifier, ex.message);
  end
end

try
  %Failing, empty sizes for labels
  disp('Test 2: giving empty array for label sizes...');
  d = dataset.Dataset(1, []);
  %If we stay here, the test has failed
  disp('FAILURE: no error has been thrown');
catch ex
  if strcmp(ex.identifier, 'Dataset:invalidSize')
    disp('SUCCESS: Invalid size correctly thrown');
  else
    fprintf('FAILURE: unexpected error: %s\n%s\n', ...
      ex.identifier, ex.message);
  end
end

try
  %Failing, matrix as patterns
  disp('Test 3: giving a 2x2 matrix for pattern sizes...');
  d = dataset.Dataset([1, 2; 3, 4], 1);
  %If we stay here, the test has failed
  disp('FAILURE: no error has been thrown');
catch ex
  if strcmp(ex.identifier, 'Dataset:invalidSize')
    disp('SUCCESS: Invalid size correctly thrown');
  else
    fprintf('FAILURE: unexpected error: %s\n%s\n', ...
      ex.identifier, ex.message);
  end
end

try
  %Failing, negative sizes
  disp('Test 4: giving negative size in pattern sizes...');
  d = dataset.Dataset([2 -1 2], 1);
  %If we stay here, the test has failed
  disp('FAILURE: no error has been thrown');
catch ex
  if strcmp(ex.identifier, 'Dataset:invalidDimensions')
    disp('SUCCESS: Invalid dimensions correctly thrown');
  else
    fprintf('FAILURE: unexpected error: %s\n%s\n', ...
      ex.identifier, ex.message);
  end
end



