% Computes the sigmoid of z (matrix, vector or scalar)
function g = sigmoid(z)
  g = 1.0 ./ (1.0 + exp(-z));
end