%Computes the gradient of the sigmoid function evaluated at z (matrix, vector or scalar)
function g = sigmoidGradient(z)
  g = sigmoid(z).*(1-sigmoid(z));
end