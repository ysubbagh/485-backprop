classdef BackpropLayer
    properties (Access = protected)
        layerCount
        transerFunc
        learningRate
    end

    methods
        %% constructor
        function this = BackpropLayer()

        end

        %% forward, create the output of the layer passed an input set
        function output = forward(this, input)

        end

        %factory function to help forward
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
            denom = 1 + exp(n);
            f = 1 / denom;
        end

        %% use backprop learning to update the weights
        function this = train(this, input, target)

        end

    end

    methods(Static)
        %mean-squared error function
        function error = msError()

        end
    end

end