%Author: Francesco Altiero
%Date: 10/12/2018

classdef NonStopCriterion < neuralnet.train.criteria.StopCriterion
  %NONSTOPCRITERION Class that manages no stop criteria at all
  %   Class whose objects are used when no stop criteria have been
  %   specified.
  
  methods
    function stop = isMet(~, ~, ~, ~)
      %isMet Returns always false
      %   Stopping criterion for no stop criteria. Returns always false.
      %
      %   Outputs:
      %     - stop: always false, as this criterion is never met
      stop = false;
    end
  end
end

