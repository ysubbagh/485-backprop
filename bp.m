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
            this.hiddenLayer.weights = randn(hiddenSize, inputSize);
            this.outputLayer.weights = randn(outputSize, hiddenSize);
            % randomize bias
            this.hiddenLayer.bias = randn(hiddenSize, 1);
            this.outputLayer.bias = randn(outputSize, 1);
            % set inital learning rate
            this.learningRate = learnRate;
        end

        %% forward, create the output of the layer passed an input set
        function output = forward(this, input)

        end

        %factory function to help forward, send to correct function
        function func = doFunc(this, n)
            switch this.transferFunc
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
            if(n < 0)
                f = 0;
            else %if n >= 0
                f = 1;
            end
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