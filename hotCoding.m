%% setup layer
network = BackpropLayer(30, 10, 3, 0.1);
network.outputLayer.transferFunc = "logsig";
network.hiddenLayer.transferFunc = "hardlim";

%% input patterns
p0 = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1];
p1 = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1];
p2 = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1];
p = [p0' p1' p2'];

% target patterns
t0 = [1 0 0]';
t1 = [0 1 0]';
t2 = [0 0 1]';
t = [t0 t1 t2];

%% train layer
output = network.forward(p0');

disp(output);

