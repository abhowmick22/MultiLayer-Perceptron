classdef MultiLayerPerceptron
   properties
       nbrLayers     % Depth: Number of layers excluding the input layer (layer 0)
       widthLayer    % Width: NUmber of neurons per layer, excluding bias unit
       weights       % weights matrix, where W(i, j, l) denotes weight 
                     % of connection from neuron i in layer l-1 to neuron j
                     % in layer l
       biases        % biases matrix, where b(i, l) denotes weight of connection
                     % from bias unit to neuron i in layer l
       activations   % activations matrix, where a(i, l) denotes the activation  
                     % at output of neuron i in layer l
       cost          % Cost function: Need to minimize this
       lambda        % Regularization parameter
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
              % Pre allocate memory and initialize for all parameters
              obj.weights = normrnd(1.0, obj.sigma, [obj.widthLayer, obj.widthLayer, obj.nbrLayers]);
              obj.biases = normrnd(1.0, obj.sigma, [obj.widthLayer, obj.nbrLayers]);
              obj.activations = zeros(obj.widthLayer, obj.nbrLayers);
           end
       end    
    
       % Forward Propagation
       function obj = forwardProp(obj, X_sample)           
           % first hidden layer output activation computed with inputs 
           obj.activations(:,1) = sigmoid(transpose(X_sample * obj.weights(:, :, 1)) + obj.biases(:, 1));
           
           % Compute the activations for each subsequent layer based on
           % weights and biases, if it is a deep perceptron
           if obj.nbrLayers > 1         
               for layer=2:obj.nbrLayers
                   obj.activations(:, layer) = sigmoid(transpose(transpose(obj.activations(:, layer-1)) ...
                                            * obj.weights(:, :, layer)) + obj.biases(:, layer));
               end
           end
       end
       
       % Cost Function
       
       
       % Train with BackPropagation
       
       
       % Predict
       
       
       % Visualise with heatmap
    
   end 
end