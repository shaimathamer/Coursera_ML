function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
h = X*theta; % Hypothesis function, inner product of X and theta;
      er = h-y; % error (difference of hypothesis and actual observation);
      er_sqr = er.^2; % error squared
      J = (1/(2*m))*sum(er_sqr); % mean-squared-error (with a 1/2 factor)
      % Partial derivative of J(theta) with respect to theta
      theta_change = (alpha/m)*(X'*(h-y));
      theta = theta-theta_change; % Update theta vector
      %Book-keeping of errors for plotting
      
      J_history(iter) = J;
     











    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
