classdef BackpropLayer_Update < handle
    properties
        inputSize
        hiddenSize
        outputSize
        %layers, each has weights, bias, and transfer function
        hiddenLayer
        outputLayer
        learningRate
        
        % The n before sent to transfer function
        hiddenInput
        finalInput

        % Activation Output for each layer
        hiddenOutput
        finalOutput

        % Current input given 
        inputPattern;

    end

    methods
        %% constructor
        function this = BackpropLayer_Update(inputCount, hiddenCount, outputCount, learnRate)
            % setup size counts
            this.inputSize = inputCount;
            this.outputSize = outputCount;
            this.hiddenSize = hiddenCount;

            % randomize weights
            this.hiddenLayer.weights = randn(this.hiddenSize, this.inputSize);
            this.outputLayer.weights = randn(this.outputSize, this.hiddenSize);

            % randomize bias
            this.hiddenLayer.bias = randn(this.hiddenSize, 1);
            this.outputLayer.bias = randn(this.outputSize, 1);

            % set inital learning rate
            this.learningRate = learnRate;
        end

        %% forward, create the output of the layer passed an input set
        function this = forward(this, input)
            this.inputPattern = input;
            % Compute hidden layer input
            this.hiddenInput = this.hiddenLayer.weights * input + this.hiddenLayer.bias;
            % Hidden layer activation value
            this.hiddenOutput = this.doFunc((this.hiddenInput), this.hiddenLayer.transferFunc);

            % Compute output layer output
            this.finalInput = this.outputLayer.weights * this.hiddenOutput + this.outputLayer.bias;
            % Output layer activation value
            this.finalOutput = this.doFunc((this.finalInput), this.outputLayer.transferFunc);
            output = this.finalOutput;
        end

        %factory function to help forward, send to correct function
        function func = doFunc(this, n, do)
            switch do
                case "hardlim"
                    func = this.hardlim(n);
                case "hardlims"
                    func = this.hardlims(n);
                case "purelin"
                    func = this.purelin(n);
                case "logsig" %log sigmoid
                    func = this.sigmoid(n);
                otherwise
                    error("Transfer function not supported.");
            end
        end

        %hard limit
        function f = hardlim(this, n)
            f = n >= 0;
        end

        %symetrical hard limit
        function f = hardlims(this, n)
            if(n < 0)
                f = -1;
            else % n >=0
                f = 1;
            end
        end

        %linear
        function f = purelin(this, n)
            f = n;
        end

        %log sigmoid
        % We will also be taking the derivative as this
        % is our function for the hidden layer 
        % We have added a boolean parameter to indicate
        % when to return the derived f 
        function f = sigmoid(this, n, deriv)
            if nargin <3
                deriv = false; % Defualt if not provided
            end
            denom = 1 + exp(-n);
            sigmoidVal = 1 ./ denom;
            if deriv
                f = (sigmoidVal ./ (1 - sigmoidVal));
            else
                f = sigmoidVal;
            end
        end

        %% use backprop learning to update the weights
        function this = train(this, input, target)

        end

        %% set functions
        function this = setLearningRate(this, newRate)
            this.learningRate = newRate;
        end

        function this = setHiddenLayerSize(this, newSize)
            this.hiddenSize = newSize;
        end

        % Backward function that will utilize backpropagation
        function this = backward(this, testPattern)

            %% Compute Output Layer Sensitivity
            % Find the error in the output layer
            % outputError = t - a
            outputError = testPattern - this.finalOutput';

            % This will be the derivative of our f(n) function
            derivOutput = this.sigmoid(this.finalInput,true);

            % This computes the sensitivity of the output layer
            % S(m+1) =  -2 * f'(n) * e(t-a)
            outputSensitivity = -2 .* (derivOutput' .* outputError);

            disp("BEFORE UPDATE")
            disp(this.outputLayer.weights);
            
            %% Update Outer Layer Weights
            % We can now use outputSensitivity to update the weight
            % of our output layer: 
            % W(m+1) = W(m+1) - learningRate * S(m+1) * a(m-1)
            val = (outputSensitivity .* this.hiddenOutput);
            finalValue = this.learningRate .* val;
            this.outputLayer.weights = this.outputLayer.weights - finalValue';
            
            disp("AFTER UPDATE")
            disp(this.outputLayer.weights);

            %% Compute Hidden Layer Sensitivity
            % First we need to get the sensitivity of the
            % hidden layer and its precursors
            % S(m) = S(m+1) * W(m+1) * f'(m)
            derivHidden = this.sigmoid(this.hiddenInput, true);
            hiddenSensitivity = (this.outputLayer.weights' * outputSensitivity') .* derivHidden;
            
            %% Update Hidden Layer Weights
            % W(m) = W(m) - learningRate * S(m) * P
            this.hiddenLayer.weights = this.hiddenLayer.weights - this.learningRate * (hiddenSensitivity * this.inputPattern');


            
        end

    end

    methods(Access=private)
        %mean-squared error function
        function error = msError(observed, predicted)
            residual = observed - predicted;
            error = power(residual, 2);
        end
    end

end