classdef BackpropLayer
    properties
        inputSize
        hiddenSize
        outputSize
        %layers, each has weights, bias, and transfer function
        hiddenLayer
        outputLayer
        learningRate
    end

    methods
        %% constructor
        function this = BackpropLayer(inputCount, hiddenCount, outputCount, learnRate)
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
        function output = forward(this, input)
            % Compute hidden layer output
            hiddenOutput = this.doFunc((this.hiddenLayer.weights * input + this.hiddenLayer.bias), this.hiddenLayer.transferFunc);
            % Compute output layer output
            output = this.doFunc((this.outputLayer.weights * hiddenOutput + this.outputLayer.bias), this.outputLayer.transferFunc);
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
                    func = this.logsig(n);
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
        function f = logsig(this, n)
            denom = 1 + exp(-n);
            f = 1. / denom;
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

    end

    methods(Static)
        %mean-squared error function
        function error = msError(observed, predicted)
            residual = observed - predicted;
            error = power(residual, 2);
        end
    end

end