classdef MultiLayerPerceptron
   properties
       nbrLayers     % Depth: Number of layers excluding the input layer (layer 0)
       widthLayer    % Width: NUmber of neurons per layer, excluding bias unit
       weights       % weights matrix, where W(i, j, l) denotes weight 
                     % of connection from neuron i in layer l-1 to neuron j
                     % in layer l
       biases        % biases matrix, where b(i, l) denotes weight of connection
                     % from bias unit to neuron i in layer l
       %activations   % activations matrix, where a(i, l) denotes the activation  
                     % at output of neuron i in layer l
       lambda        % weight decay parameter
       sigma         % normal std. dev. for initial randomization
       alpha         % Learning rate
   end
   methods
       % Constructor to build a configurable perceptron
       function obj = MultiLayerPerceptron(depth, width, l, s, a)
           if nargin > 0
              obj.nbrLayers = depth;
              obj.widthLayer = width;
              obj.lambda = l;
              obj.sigma = s;
              obj.alpha = a;
              obj.weights = normrnd(1.0, obj.sigma, [obj.widthLayer, obj.widthLayer, obj.nbrLayers]);
              obj.biases = normrnd(1.0, obj.sigma, [obj.widthLayer, obj.nbrLayers]);
           end
       end    
    
       % Forward Propagation
       % @param obj: The multilayer perceptron
       % @param X_sample: A data point
       % @output h: Return the matrix of activations for this sample point,
       % alongwith sample cost. Activations at output of layer 0 are just
       % the inputs
       function [activations, cost] = forwardProp(obj, X_sample, Y_sample)
           activations = zeros(obj.widthLayer, obj.nbrLayers+1);
           % first hidden layer output activation are just (sigmoids ?) inputs 
           %activations(:,1) = sigmoid(transpose(X_sample * obj.weights(:, :, 1)) + obj.biases(:, 1));  
           activations(:,1) = transpose(X_sample);
           % Compute the activations for each subsequent layer based on
           % weights and biases
           if obj.nbrLayers > 0         
               for layer=1:obj.nbrLayers+1
                   activations(:, layer) = sigmoid(transpose(transpose(activations(:, layer-1)) ...
                                            * obj.weights(:, :, layer)) + obj.biases(:, layer));
               end
           end           
           cost = 0.5 * sumsqr(Y_sample - activations(1:size(Y_sample), obj.nbrLayers+1));
       end
       
       % Get sum of weights squared regularized
       function sum = regSumOfWeightsSquared(obj)
           pair = sumsqr(obj.weights);
           sum = pair(1) * obj.lambda * 0.5;
       end
       
       % Compute derivatives
       function [Wders, Bders] = computeDerivatives(obj, activations, Y_train)
           Wders = zeros(size(obj.weights));
           Bders = zeros(size(obj.biases));
           numPoints = size(activations, 3);
           for m=1:numPoints
               deltas = computeDeltas(obj, activations, Y_train); 
               %Wders(i,j,l) = Wders(i,j,l) + activations(i,l,m) * deltas(j,l);
               Wders = Wders + activations(:,:,m) * transpose(deltas);
               %Bders(i,l) = Bders(i,l) + deltas(i,l);
               Bders = Bders + deltas;
           end
           Wders = Wders./m + obj.lambda.*obj.weights;
           Bders = Bders./m;
       end
       
       % Compute deltas, for output units that are not input
       function deltas = computeDeltas(obj, activations, Y_train)
           deltas = zeros(obj.widthLayer, obj.nbrLayers);
       end
           
       % Update parameters
       % @param Wders: partial derivative wrt weights
       % @param Bders: partial derivative wrt bias
       % @output obj: perceptron with updated parameters
       function obj = updateWeights(obj, Wders, Bders)
           
       end
       
       % Visualize hidden layers
    
   end 
end