clear all
close all

% dataDir = <path to directory flacData> e.g:
dataDir = "D:\Informatica\2020-2021\SPEECH PROCESSING AND RECOGNITION - 101803\Progetto\flacData";

ads = audioDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'FileExtensions','.flac', ...
    'LabelSource','foldernames');

ads = shuffle(ads);

[adsTrain, adsTest] = splitEachLabel(ads, 0.9);

fs = 16000;
% in this settings the windowLength is set to 30 ms
% and the overlapping window to 25 ms
windowLength = round(0.03 * fs);
overlapLength = round(0.025 * fs);

features = [];
labels = [];

while hasdata(adsTrain)
    [audioIn, dsInfo] = read(adsTrain);
%  MFCC and f0 extraction, the hamming filter is used
    melC = mfcc(audioIn, fs, 'Window', hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
    f0 = pitch(audioIn, fs, 'WindowLength', windowLength,'OverlapLength',overlapLength);
    feat = [melC, f0];
    
    voicedSpeech = isVoicedSpeech(audioIn, fs, windowLength, overlapLength);

    feat(~voicedSpeech, :) = [];
    label = repelem(dsInfo.Label, size(feat, 1));
    
    features = [features; feat];
    labels = [labels, label];
end

% Here the features are normalized on a same scale
Mean = mean(features, 1);
Std_dev = std(features, [], 1);
features = (features - Mean)./Std_dev;

% rng for reproducibility
rng("default");
c = cvpartition(labels, "Holdout", 0.2);
trainingIndices = training(c); % Indices for the training set
testIndices = test(c); % Indices for the test set
X_train = features(trainingIndices,:);
X_test = features(testIndices,:);
cpy_labels = labels';
Y_train = cpy_labels(trainingIndices,:);
Y_test = cpy_labels(testIndices,:);

% Here we create a neural network model and then
% we calculate the accuracy on the test set
Mdl = fitcnet(X_train, Y_train);
testAccuracy = 1 - loss(Mdl, X_test, Y_test, "LossFun", "classiferror")
Y_pred = predict(Mdl, X_test);
cm = confusionchart(Y_test, Y_pred);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Normalization = 'row-normalized';
sortClasses(cm,'descending-diagonal');
cm.Normalization = 'absolute';

