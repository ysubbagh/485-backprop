import BackpropLayer_Update.*
close all;

%% setup layer
% Initialize the network with an input size of 30
% a hidden layer of 10 neurons, and an output layer
% equal to the number of values being identified
network = BackpropLayer_Update(30, 20, 3, 0.0001);
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

%% training the network
epoch = 50000;
%{
network = network.train(t0,p0', epoch);
network = network.train(t1,p1', epoch);
network = network.train(t2,p2', epoch);
%}

for rounds = 1:epoch
    for i = 1:size(p, 2)
        % Get the ith input pattern and target pattern
        inputPattern = p(:, i);
        targetPattern = t(:, i);

        % Train the network with the current input and target pattern
        network = network.train(targetPattern', inputPattern, 1);
    end
end


%% testing
%{
output = network.compute(p0');
disp("output for p0");
disp(output);

output = network.compute(p1');
disp("output for p1");
disp(output);

output = network.compute(p2');
disp("output for p2");
disp(output);
%}

%% noisy testing
% Initialize accuracy matrix
numVersions = 50;
noiseLevels = [0 4 8];
accuracyMatrix = zeros(length(noiseLevels), 1);


for i=1:length(noiseLevels)
    noiseLevel = noiseLevels(i);
    correctCount = 0; 

    for k=1:numVersions

        for j=0:2 %get the patterns
            %make noisy 
            inputPattern = getPattern(j);
            targetVal = getTarget(j);
            noisyInput = addNoise(inputPattern, noiseLevel);
    
            %classify
            output = network.compute(noisyInput);
    
            %check correctness
            if isCorrect(output, targetVal)
                correctCount = correctCount + 1;
            end
   
        end 

    end
    % compute accuracy
    accuracyMatrix(i) = (correctCount / (numVersions * 3)) * 100;

end
% results
disp("accuracy");
disp(accuracyMatrix);

%% Plot the graph
xTicks = [0, 4, 8]; % Define the x-axis ticks
figure;
hold on;
plot(noiseLevels, accuracyMatrix, '-o');
hold off;
xlabel('Number of Pixels Flipped');
ylabel('Classification Accuracy (%)');
title('Network Performance of 3-Layer Backpropagation with Noisy Inputs');
grid on;
xticks(0:8); % Set x-axis ticks to integers from 2 to 6



%% helper functions

%%addNoise to a vector, distort it 
function pvec = addNoise(pvec, num)
    % ADDNOISE Add noise to "binary" vector
    % pvec pattern vector (-1 and 1)
    % num number of elements to flip randomly
    % Handle special case where there's no noise
    if num == 0
        return;
    end
    % first, generate a random permutation of all indices into pvec
    inds = randperm(length(pvec));
    % then, use the first n elements to flip pixels
    pvec(inds(1:num)) = -pvec(inds(1:num));
end 

% check for correctness
function correct = isCorrect(output, target)
    [~, predictedClass] = max(output);
    [~, trueClass] = max(target);
    correct = predictedClass == trueClass;
end

%get pattern
function pattern = getPattern(num)
    switch num
        case 0
            pattern = [-1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1]';
        case 1
            pattern = [-1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1]';
        case 2
            pattern = [1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1]';
        otherwise
            error("Invalid paatern retrival.");
    end
end 

%get target
function target = getTarget(num)
    switch num
        case 0
            target = [1 0 0]';
        case 1
            target = [0 1 0]';
        case 2
            target = [0 0 1]';
        otherwise
            error("Invalid paatern retrival.");
    end
end 
