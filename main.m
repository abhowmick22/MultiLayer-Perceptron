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

% Read the training data
X = [2,2 ; 2,1 ; 1,2 ; 1,1];
Y = [1; 0; 0; 1];
numPoints = size(X, 1);

% Train the perceptron
totalCost = 10.0;
numIters = 0;
% To store activations for all data points
activations = zeros(perceptron.widthLayer, perceptron.nbrLayers+1, numPoints);
while totalCost > 0.1 && numIters < 5
    totalCost = 0.0;   
    for m=1:numPoints
        [activations(:,:,m), cost] = perceptron.forwardProp(X(m, :), Y(m, :));
        totalCost = totalCost + cost;
    end
    totalCost = totalCost + perceptron.regSumOfWeightsSquared();
    [Wders, Bders] = perceptron.computeDerivatives(activations);
    perceptron = perceptron.updateWeights(Wders, Bders);
    numIters = numIters + 1;
end
    
% Test accuracy
