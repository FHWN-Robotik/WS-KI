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


%% Modell testen

[YPred, probs] = classify(trainedNetwork_1, imdsValidation);

figure
tiledlayout("flow");
perm = randperm(2500,20);
for i = 1:20
    nexttile
    imshow(imdsValidation.Files{perm(i)});
    label = YPred;
    M = max(probs,[],2);
    title(string(label(perm(i))) + ", " + num2str(100*M(perm(i)),3) + "%");
end

%% Meine Handschrift

I = imread("vier.png");
%I = imresize(I, [28 28 1]);
I = rgb2gray(I);

[YPred,probs] = classify(trainedNetwork_1,I);
imshow(I)
label = YPred;
title(string(label) + ", " + num2str(100*max(probs),3) + "%");
