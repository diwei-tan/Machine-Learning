function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Firstly, a(1) = X, plus a(1)0 (ones)
a1 = [ones(m, 1) X];
% z(2) = theta1 * a1
z2 = a1*Theta1';
% a2 = g(z2)
a2 = sigmoid(z2);
% add a(2)0 (ones) to a2
a2 = [ones(m,1) a2];
% z(3) = theta2 * a2
z3 = a2*Theta2';
% a(3) g(z3)
a3 = sigmoid(z3);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% y matrix is m by classifers num_labels. Need to create y matrix
ymatrix = zeros(m,num_labels);
% add the 1s in the position of the y value from orginal y matrix
for i=1:m,
  ymatrix(i,y(i)) = 1;
endfor
% h(x) is a(3)
prediction = a3;
J = (1/m)*sum(sum(((-ymatrix).*log(prediction)) - ((1-ymatrix).*log(1-prediction)),2));

%penalty
penalty = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2,2))+sum(sum(Theta2(:,2:end).^2,2)));
J=J+penalty;

%prediction =  sigmoid(X*theta);
%reg_theta = theta;
%reg_theta(1)=0;
%J = (1/m) * (((-y)'*log(prediction)) - ((1-y)'*log(1 .- prediction))) + lambda/(2*m) .* (reg_theta'*reg_theta) ;
%grad = (X'*(prediction - y) + (lambda .* reg_theta)) ./m;
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Backward propagation
%compute error term
error3 = a3 - ymatrix;
%note:z(2) which is a1(5000X401) * Theta1'(401X25) is (5000X25).
%error3(5000X10) * Theta2(10X26) is (5000X26). need to remove first error(2)0.
error2 = (error3*Theta2).*sigmoidGradient([ones(size(z2),1) z2]);
error2 = error2(:,2:end);

%Calculate backward propagation delta terms
delta_2 = error3'*a2;
delta_1 = error2'*a1;

Theta1_grad = delta_1./m;
Theta2_grad = delta_2/m;

%Regularize: Add lambda/m * Thetas to the Theta gradient, but not the first column (j=0)
%Note: To do so, change all first column in Theta to zeros.
reg_term_theta1 = lambda/m .* [zeros(size(Theta1,1),1) Theta1(:,2:end)];
reg_term_theta2 = lambda/m .* [zeros(size(Theta2,1),1) Theta2(:,2:end)];

Theta1_grad = Theta1_grad + reg_term_theta1;
Theta2_grad = Theta2_grad + reg_term_theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
