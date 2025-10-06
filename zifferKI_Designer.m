clear all; clc;
%% Daten erfassen
dataFolder = fullfile(toolboxdir('nnet'),'nndemos','nndatasets','DigitDataset');

imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% Daten vorbereiten

% Zeigt zufällige Bilder aus dem Datensatz.
figure
tiledlayout("flow");
perm = randperm(10000,20);
for i = 1:20
    nexttile
    imshow(imds.Files{perm(i)});
end

% Zeigt Anzahl je Ziffer
classNames = categories(imds.Labels);
labelCount = countEachLabel(imds)

% Zeigt die Pixelanzahl der Bilder
img = readimage(imds,1);
pixel = size(img)

% Trainingsdaten und Testdaten
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,"randomize");

%% Trainiertes Modell vom Designer laden

options = trainingOptions("sgdm", ...
    MaxEpochs=4, ...
    ValidationData=imdsValidation, ...
    ValidationFrequency=10, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

net = trainnet(imdsTrain,net_1,"crossentropy",options);
%% Wie genau ist das Modell?

accuracy = testnet(net,imdsValidation,"accuracy")
%% Modell testen

scores = minibatchpredict(net,imdsValidation);
YValidation = scores2label(scores,classNames);

numValidationObservations = numel(imdsValidation.Files);
idx = randi(numValidationObservations,9,1);

figure
tiledlayout("flow")
for i = 1:9
    nexttile
    img = readimage(imdsValidation,idx(i));
    imshow(img)
    title("Predicted Class: " + string(YValidation(idx(i))))
end

%% Meine Handschrift

filename = "five.png";

image = imread(filename);
image = imresize(image, [28 28]);
image = rgb2gray(image);

scores = minibatchpredict(net,image);
YValidation = scores2label(scores,classNames);
image = imresize(image, [200 200]);
figure(3)
    imshow(image)
    title("Prediction: " + string(YValidation))
    ax = gca;
    ax.FontSize = 18;