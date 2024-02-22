import BackpropLayer_Update.*

%% setup layer
% Initialize the network with an input size of 30
% a hidden layer of 10 neurons, and an output layer
% equal to the number of values being identified
network = BackpropLayer_Update(30, 10, 3, 0.001);
network.outputLayer.transferFunc = "logsig";
network.hiddenLayer.transferFunc = "logsig";

%% input patterns
p0 = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
p1 = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1];
p2 = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1];
p = [p0' p1' p2'];

% target patterns
t0 = [1 0 0];
t1 = [0 1 0];
t2 = [0 0 1];
t = [t0' t1' t2'];

%% train layer
% network.forward(p2');
% 
% 
% network.backward(t2');
% network.forward(p2');
% network.backward(t2');
% network.forward(p2');
% network.backward(t2');
% network.forward(p2;

epoch = 1000;
network = network.train(t0,p0', epoch);
network = network.train(t1,p1', epoch);
network = network.train(t2,p2', epoch);


%{
for i = 1:size(p, 2)
    % Get the ith input pattern and target pattern
    inputPattern = p(:, i);
    targetPattern = t(:, i);
    
    % Train the network with the current input and target pattern
    network = network.train(targetPattern', inputPattern, epoch);
end
%}


%% testing
output = network.compute(p0');
disp("output p0");
disp(output);

output = network.compute(p1');
disp("output p1");
disp(output);

output = network.compute(p2');
disp("output p2");
disp(output);

