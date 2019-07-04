%testDemoObjectiveEval_ForTestSet_FastGF
close all; clear; clc;

deviceID = 0;

isWriteImgs = 0;
isSaveResults = 1;
isEvalSSIM = 0;
isResizeTest = 1;
maxResizeSize = 360;

rGF = 48;
epsGF = 1e-4;
downScaleFactorGF = 4;

KMapIdx = 93;

addpath(genpath('./utils/'))
addpath(genpath('./fast-guided-filter/'))

modelPath = './model/FAMED-Net/';
rootForData = ['RESIDE_DATASET_ROOT/SOTS/'];
dataSetNameWhole = {'/indoor/', '/outdoor/'};
subFolderName = {'/hazy/', '/clear/'};
imgFormat = {'png', 'jpg'};

%%
isEval = 1;
if isEval
    %------------------------matlab-------------------------------
    addpath('CAFFE_ROOT/matlab/');
    
    caffe.reset_all();
    
    caffe.set_mode_gpu();
    caffe.set_device(deviceID);
    
    netModel = [modelPath, 'deploy.prototxt'];
    modelFileName = 'FAMED-Net';
    netWeights = [modelPath, modelFileName, '.caffemodel'];
    if ~exist(netWeights, 'file')
        error('There is no model exists');
    end
    
    net = caffe.Net(netModel, netWeights, 'test'); % create net and load weights
end


%%
for dataSetIdx = 1:2
    dataSetName = dataSetNameWhole{dataSetIdx};
    disp(['>>> processing ', dataSetName, ' test set...']);

    rootData = [rootForData, dataSetName,subFolderName{1}];
    
    testResultSavePath = ['./results/',dataSetName];
    if ~exist('testResultSavePath', 'dir')
        mkdir(testResultSavePath);
    end
    
    
    %prepare img list
    HazeImagePathList = dir([rootData,'*.', imgFormat{dataSetIdx}]);
    HazeImageNum = length(HazeImagePathList);
    
    timeCost = zeros(1, HazeImageNum);
    index = zeros(2, HazeImageNum);
    for hazeImgIter = 1:HazeImageNum
        if isEvalSSIM
            disp(['currently processing ', num2str(hazeImgIter), 'th img...']);
        end
        HazeImageName = HazeImagePathList(hazeImgIter).name;
        HazeImage = im2double(imread([rootData,HazeImageName]));
        
        HazeImageBlob = single(permute(HazeImage,[2,1,3])); %h*w*c -> w*h*c
        HazeImageBlobV = max(HazeImageBlob, [], 3);
        
        [wid,hei,c] = size(HazeImageBlob);
        if isResizeTest
            ratio = maxResizeSize / max(hei,wid);
            widR = round(wid * ratio);
            heiR = round(hei * ratio);
            HazeImageBlobR = imresize(HazeImageBlob, [widR, heiR]);
        else
            widR = wid;
            heiR = hei;
            HazeImageBlobR = HazeImageBlob;
        end
        
        net.blobs('data').reshape([widR,heiR,3,1]);
        net.reshape();
        
        tic;
        im_forward = net.forward({HazeImageBlobR});
        timeCost(hazeImgIter) = toc;
        
        KMapR = net.blob_vec(1, KMapIdx).get_data();
        KMap = imresize(KMapR, [wid, hei]);
        
        KMap_FastGF = KMap;
        for cc = 1:3
            KMap_FastGF(:,:,cc) = fastguidedfilter(HazeImageBlobV, KMap(:,:,cc), rGF, epsGF, downScaleFactorGF);
        end
        
        DehazedImage = KMap_FastGF .* HazeImageBlob - KMap_FastGF + 1; %transformed hazy imaging model
        DehazedImage = permute(DehazedImage,[2,1,3]);
        %     timeCost(hazeImgIter) = toc; %include gf running time
        
        if isWriteImgs
            imgSaveName = [testResultSavePath, HazeImageName];
            if ~isResizeTest
                imgSaveName = [testResultSavePath, strrep(HazeImageName, ['.', imgFormat{dataSetIdx}], ['_ori.',imgFormat{dataSetIdx}])];
            end
            imwrite(uint8(DehazedImage*255), imgSaveName);
        end
        
        
        pos=find(HazeImageName=='_');
        HazeImageLabelName = [HazeImageName(1:pos(1)-1),'.png'];
        
        gt = imread([strrep(rootData, subFolderName{1}, subFolderName{2}),HazeImageLabelName]);
        if dataSetIdx == 1
            gt = gt(11:end-10, 11:end-10, :);
        end
        index(1, hazeImgIter) = psnr(gt, uint8(DehazedImage*255));
        if isEvalSSIM
            index(2, hazeImgIter) = ssim(gt, uint8(DehazedImage*255));
        end
        
        
    end
    timeCostMu = median(timeCost)
    indexMu = mean(index,2)
    
    %%
    log = struct;
    log.modelFileName = modelFileName;
    log.rootData = rootData;
    log.rGF = rGF;
    log.epsGF = epsGF;
    log.downScaleFactorGF = downScaleFactorGF;
    log.maxSize = maxResizeSize;
    log.KMapIdx = KMapIdx;
    log.timeCostMu = timeCostMu;
    log.index = index;
    
    if isSaveResults
        saveName = [testResultSavePath, modelFileName, '_log.mat'];
        save(saveName, 'log');
    end
    
end