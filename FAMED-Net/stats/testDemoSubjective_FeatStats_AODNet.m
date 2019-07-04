%testDemoSubjective_FeatStats_AODNet
close all; clear; clc;

isDebug = 0;
deviceID = 0;

isWriteImgs = 0;
isSaveResults = 1;

isResizeTest = 1;
maxResizeSize = 360;

rGF = 48;
epsGF = 1e-4;
downScaleFactorGF = 4;

% KMapIdx = 93;

addpath(genpath('./../utils/'))
addpath(genpath('./../fast-guided-filter/'))


imgFormat = 'jpg';
rootData = './stats100/clear/';

%%
isEval = 1;
if isEval
    %------------------------matlab-------------------------------
    addpath('/home/admin007/jingzhang/caffe-master-FPCNet_DH/matlab/');
    caffe.set_mode_gpu();
    caffe.set_device(0);
    %         caffe.set_mode_cpu();
    
    modelPath = './../model/AOD-Net/'; KMapIdx = 20;
    netModel = [modelPath, 'deploy.prototxt'];
    netWeights = [modelPath, 'AOD_Net.caffemodel'];
    
    if ~exist(netWeights, 'file')
        error('There is no model exists');
    end
    
    net = caffe.Net(netModel, netWeights, 'test'); % create net and load weights
end

%%

HazeImagePathList = dir(strcat(rootData,'*.', imgFormat));
HazeImageNum = length(HazeImagePathList);

numBin = 20;
stats = zeros(1,numBin);
center = linspace(0,1,numBin);
timeCost = zeros(1, HazeImageNum);
for hazeImgIter = 1:HazeImageNum
    disp(['currently processing ', num2str(hazeImgIter), 'th img...']);
    
    HazeImageName = HazeImagePathList(hazeImgIter).name;
    HazeImage = im2double(imread([rootData,HazeImageName]));
    
    HazeImageBlob = single(permute(HazeImage,[2,1,3])); %h*w*c -> w*h*c
    
    [wid,hei,c] = size(HazeImageBlob);
    if isResizeTest
        HazeImageBlobR = imresize(HazeImageBlob, [maxResizeSize, maxResizeSize], 'nearest');
    end
    
    net.blobs('data').reshape([maxResizeSize, maxResizeSize, c, 1]);
    net.reshape;
    
    tic;
    im_forward = net.forward({HazeImageBlobR});
    timeCost(hazeImgIter) = toc;
    
    KMapR = net.blob_vec(1, KMapIdx).get_data();
    
    rMinFilter = 3;
    KMapStat = 1 - min(mean(1./(KMapR+eps), 3), 1);
    KMapStat = minFilter2(KMapStat, rMinFilter);
    statsTmp = hist(KMapStat(:), center);
    stats(1,:) = stats(1,:) + statsTmp;

end
timeCostMu = mean(timeCost,2)

%%
stats = stats ./ repmat(sum(stats,2), [1, numBin]);
statsCumsum = cumsum(stats, 2);
figure; subplot(1,2,1);
bar(center, stats'); axis([-0.05 1 0 0.8]);

subplot(1,2,2);
plot(center, statsCumsum(1,:), 'r-*');
axis([0,1,0,1])
%%

if isSaveResults
    saveName = ['stats_aodnet.mat'];
    save(saveName, 'stats');
end