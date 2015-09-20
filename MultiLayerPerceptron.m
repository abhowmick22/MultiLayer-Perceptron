classdef MultiLayerPerceptron
   properties
       nbrLayers     % Depth: Number of layers excluding the input layer (layer 0)
       widthLayer    % Width: NUmber of neurons per layer, excluding bias unit
       weights       % weights matrix, where W(i, j, l) denotes weight 
                     % of connection from neuron i in layer l to neuron j
                     % in layer l+1
       biases        % biases matrix, where b(i, l) denotes weight of connection
                     % from bias unit to neuron i in layer l+1
       activations   % activations matrix, where a(i, l) denotes the activation at output of 
                     % neuron i in layer l+1
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
              obj.weights = normrnd(0.0, obj.sigma, [obj.widthLayer, obj.widthLayer, obj.nbrLayers]);
              obj.biases = normrnd(0.0, obj.sigma, [obj.widthLayer, obj.nbrLayers]);
              obj.activations = zeros(obj.widthLayer, obj.nbrLayers);
           end
       end    
       
       % Initialize weights
    
       % Feedforward pass
       
       
       % Cost Function
       
       
       % Train with BackPropagation
       
       
       % Predict
       
       
       % Visualise with heatmap
    
   end 
end