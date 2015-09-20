classdef MultiLayerPerceptron
   properties
       nbrLayers     % Depth: Number of layers excluding the input layer (layer 0)
       widthLayer    % Width: NUmber of neurons per layer
       weights       % weights matrix, where W(i, j, l) denotes weight 
                     % of connection from neuron i in layer l to neuron j
                     % in layer l+1
       biases        % biases matrix, where b(i, l) denotes weight of connection
                     % from bias unit to neuron i in layer l+1
       activations   % activations matrix, where a(i, l) denotes the activation of 
                     % neuron i in layer l
       cost          % Cost function: Need to minimise this
       lambda        % Regularization parameter
       epsilon       % variance for initial randomization
       alpha         % Learning rate
   end
   methods
       % Constructor to build a configurable perceptron
       function obj = MultiLayerPerceptron(depth, width, l, e, a)
           if nargin > 0
              obj.nbrLayers = depth;
              obj.widthLayer = width;
              obj.lambda = l;
              obj.epsilon = e;
              obj.alpha = a;
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