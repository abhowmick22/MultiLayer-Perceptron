%% Lab Course in Deep Learning: 11-785
%% Multilayer Perceptron with BackPropation
%% Author: Abhishek Bhowmick
%% Organization: Carnegie Mellon University
%% Andrew Id: abhowmi1

% This is the main function where we define the perceptron, read the
% inputs, train the model, predict on test, visualize the results and
% report the metrics. Each layers is fully connected to the successive
% layer.

% Define the Perceptron Neural Network
perceptron = MultiLayerPerceptron(2, 2, 0.01, 0.0, 0.1);
X = [2 , 2];
perceptron = perceptron.forwardProp(X);

% Read the training data


% Train the matrix


% Test accuracy
