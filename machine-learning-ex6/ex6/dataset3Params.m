function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%set multipicative steps as noted in pdf
steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%keep best current error for comparison while looping, set to some large number
best_error= 10000;
%Loop for iteration of all possible combination of values for C and alpha
for i=1:8,
  for j=1:8,
    C_iter = steps(i);
    sigma_iter = steps(j);
     model = svmTrain(X, y, C_iter, @(x1, x2) gaussianKernel(x1, x2, sigma_iter));
     % Use prediction and error instruction above to find error for iteration
     predictions = svmPredict(model, Xval);
     error_iter = mean(double(predictions~=yval));
     %If current iteration is better than any previous iter, change C and sigma
     if error_iter < best_error,
       C=C_iter;
       sigma=sigma_iter;
       % Change best_error to current iteration error. to use as comparison
       best_error=error_iter;
     endif
  endfor
endfor

% At the end, this should return best C and sigma with least error
% =========================================================================

end
