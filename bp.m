classdef BackpropLayer
    properties
       layerCount

    end

    methods
        %constructor
        function this = BackpropLayer()

        end

        %create the output of the layer passed a input set
        function output = forward(input)

        end

        %use backprop learning to update the weights
        function this = learn()

        end

    end

    methods(Static)
        function error = backError()

        end
    end

end