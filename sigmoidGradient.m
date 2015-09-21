%Computes the gradient of the sigmoid function evaluated at z (matrix, vector or scalar)
function g = sigmoidGradient(activation)
  g = activation.*(1-activation);
end