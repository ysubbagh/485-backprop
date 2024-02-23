import BackpropLayer_Update.*
close all;

%% get data 
% file paths
trainImgFile = 'mnist-data/train-images.idx3-ubyte';
trainLabelFile = 'mnist-data/train-labels.idx1-ubyte';
testImgFile = 'mnist-data/t10k-images.idx3-ubyte';
testLabelFile = 'mnist-data/t10k-labels.idx1-ubyte';

% format data
trainImg = loadMNISTImages(trainImgFile);
trainLabels = loadMNISTLabels(trainLabelFile);
trainLabels = trainLabels'; % format the matrix dimensions 
testImg = loadMNISTImages(testImgFile);
testLabels = loadMNISTLabels(testLabelFile);
testLabels = testLabels';


%% setup network
% returns hot coded value 1-10
network = BackpropLayer_Update(size(trainImg, 1), 20, 10, 0.001);
network.outputLayer.transferFunc = "logsig";
network.hiddenLayer.transferFunc = "logsig";


%% do the training
epoch = 20;

for rounds = 1:epoch
    for i = 1:size(trainImg, 2)
        % Get the ith input pattern and target patterns
        inputPattern = trainImg(:, i);
        targetPattern = trainLabels(:, i);

        % Train the network with the current input and target pattern
        network = network.train(targetPattern', inputPattern, 1);
    end
end


%% validty check
correctCount = 0;
for i = 1:size(testImg, 2)
    % get patterns
    input = testImg(:, i);
    target = testLabels(:, i);
    % classify
    output = network.compute(input);
    % check correctness
    if isCorrect(output, target)
        correctCount = correctCount + 1;
    end
end
% check accuracy
accruacy = (correctCount / size(testImg, 2)) * 100;
%disp("accuracy = " + accruacy + "%");


%% testing for dirty data classification
% setup noise and accuracy
numVersions = 6;
noiseLevels = [50 100 200 300 600];
accuracyMatrix = zeros(length(noiseLevels),1);

% test accuracy at each of the noise levels at same amount at training
for i = 1:length(noiseLevels)
    noiseLevel = noiseLevels(i);
    correctCount = 0;
   
    
    % iter through the images in test data
    for k = 1:size(testImg, 2)
        %get patterns
        input = testImg(:, i);
        noisyInput = addNoise(input ,noiseLevel);
        target = testLabels(:, i);

        %classify
        output = network.compute(noisyInput);

        %check correctness
        if isCorrect(output, target)
            correctCount = correctCount + 1;
        end
    end

    accuracyMatrix(i) = (correctCount / size(testImg, 2)) * 100;
end

%% print graph
disp("accuracy");
disp(accuracyMatrix);

figure;
hold on;
plot(noiseLevels, accuracyMatrix, '-o', 'LineWidth',2);
hold off;
xticks(noiseLevels);
xticklabels({'5', '10', '50', '100', '200'});  % Set custom labels for the x-ticks
grid on;
xlabel('Number Of Pixels Flipped');
ylabel('Classification Accuracy (%)');
title('Network Performance of Backpropogated Multilayer Network With Noisy Inputs');


%% mnist helper functions to parse data
% parse labels
function labels = loadMNISTLabels(filename)
    %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    %the labels for the MNIST images
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', filename, '']);
    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
    labels = fread(fp, inf, 'unsigned char');
    assert(size(labels,1) == numLabels, 'Mismatch in label count');
    fclose(fp);
end

%for the images
function images = loadMNISTImages(filename)
    %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
    %the raw MNIST images
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', filename, '']);
    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images,[2 1 3]);
    fclose(fp);
    % Reshape to #pixels x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    % Convert to double and rescale to [0,1]
    images = double(images) / 255;
end

%% validity/testing helper functions

% check for correctness
function correct = isCorrect(output, target)
    [~, predictedClass] = max(output);
    [~, trueClass] = max(target);
    correct = predictedClass == trueClass;
end

% addNoise to a vector, distort it 
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
