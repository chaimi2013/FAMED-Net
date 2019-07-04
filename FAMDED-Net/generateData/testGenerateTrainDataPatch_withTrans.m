% testGenerateTrainDataPatch_withTrans

clear; clc; close all;

addpath(genpath('./utils/'))

isWithTrans = 1;
isRESIDE_OTS = 0;

if ~isRESIDE_OTS
    h5Num = 128; %128 for ITS; 256 for OTS 
    dataSetName = 'ITS';
    root = 'RESIDE_DATASET_ROOT';
    samplesPerImg = 3; %5 for 32 h5
    isSampling = 1;
    samplingRatio = 0.8; %0.8 for ITS-128
    imgFormat = 'png';
else
    h5Num = 256; %128 for ITS; 256 for OTS 
    dataSetName = 'OTS_ALPHA';
    root = 'RESIDE_DATASET_ROOT';
    samplesPerImg = 2; %5 for 32 h5
    isSampling = 1;
    samplingRatio = 0.7586; %0.8 for ITS-128
    imgFormat = 'jpg';
end

rootData = [root, dataSetName,'/'];
h5SavePath = [root, 'h5Patch_GtHazyTrans/'];
if ~exist(h5SavePath, 'dir')
    mkdir(h5SavePath);
end

if isWithTrans
    TransFilePath = [rootData, '/trans/'];
end

HazeImageFilePath = [rootData, '/hazy/'];
HazeImagePathList = dir([HazeImageFilePath,'*.', imgFormat]);
HazeImageNum = length(HazeImagePathList);
idxRand = randperm(HazeImageNum);
HazeImagePathList = HazeImagePathList(idxRand);
if isSampling
    HazeImageNum = round(HazeImageNum * samplingRatio);
end
HazeImagePathList = HazeImagePathList(1:HazeImageNum);

GTFilePath = [rootData, '/clear/'];

HazeImageNumPerH5 = floor(HazeImageNum / h5Num);

patchHei = 128;
patchWid = patchHei;
channel = 3;

timeCost = zeros(1, h5Num);
for h5NumIter = 1:h5Num
    disp(['currently porocessing ', num2str(h5NumIter), 'th h5 ...']);
    tic;
    
    sampleNumPerH5 = ceil(HazeImageNumPerH5 * samplesPerImg);
    data = zeros(patchHei,patchWid,channel, sampleNumPerH5, 'single');
    if isWithTrans
        label = zeros(patchHei,patchWid,channel+1, sampleNumPerH5, 'single');
    else
        label = zeros(patchHei,patchWid,channel, sampleNumPerH5, 'single');
    end
    HazeImageIndexShift = (h5NumIter-1) * HazeImageNumPerH5 + 1;
    count = 0;
    
    for i = HazeImageIndexShift:HazeImageIndexShift+HazeImageNumPerH5-1
        HazeImageName = HazeImagePathList(i).name;
        HazeImage = im2double(imread(strcat(HazeImageFilePath,HazeImageName)));
        
        pos=find(HazeImageName=='_');
        HazeImageLabelName = [HazeImageName(1:pos(1)-1),'.', imgFormat];
        HazeImageLabel = im2double(imread(strcat(GTFilePath,HazeImageLabelName)));
        
        if isWithTrans
            if isRESIDE_OTS
                TransImageName = HazeImageName;
                TransImage = im2double(imread(strcat(TransFilePath,TransImageName)));
            else
                TransImageName = [HazeImageName(1:pos(2)-1),'.', imgFormat];
                TransImage = im2double(imread(strcat(TransFilePath,TransImageName)));
            end
        end
        
        [hei,wid,c] = size(HazeImage);
        for j = 1:samplesPerImg
            rowS = floor((hei-patchHei-2) * rand)+1;
            colS = floor((wid-patchWid-2) * rand)+1;
            tmpHaze = HazeImage(rowS:rowS+patchHei-1,colS:colS+patchWid-1,:);
            tmpGT = HazeImageLabel(rowS:rowS+patchHei-1,colS:colS+patchWid-1,:);
            tmpTrans = TransImage(rowS:rowS+patchHei-1,colS:colS+patchWid-1);
            count = count+1;
            
            data(:,:,:,count) = tmpHaze;
            if isWithTrans
                label(:,:,:,count) = cat(3, tmpGT, tmpTrans);
            else
                label(:,:,:,count) = tmpGT;
            end
            
        end
    end
    if count ~= sampleNumPerH5
        error('sample Number error');
    end
    
    order = randperm(count);
    data = data(:, :, :, order);
    label = label(:, :, :, order);
    
    
    %-----------writing to HDF5----------
    h5Name = [h5SavePath, dataSetName, '_', num2str(h5NumIter), '.h5'];
    if exist(h5Name, 'file')
        delete(h5Name);
        disp(['delete existing h5 file successfully.'])
    end
    chunksz = 64;
    created_flag = false;
    totalct = 0;
    for batchno = 1:floor(sampleNumPerH5/chunksz)
        last_read = (batchno-1) * chunksz;
        batchdata = data(:,:,:,last_read+1:last_read+chunksz);
        batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
        
        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
        curr_dat_sz = store2hdf5(h5Name, batchdata, batchlabs, ~created_flag, startloc, chunksz);
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(h5Name);
    
    timeCost(h5NumIter) = toc;
    disp(['time cost: ', num2str(timeCost(h5NumIter)), 'senconds...']);
    
end

logInfo = struct;
logInfo.h5Num = h5Num;
logInfo.dataSetName = dataSetName;
logInfo.HazeImageFilePath = HazeImageFilePath;
logInfo.HazeImageNum = HazeImageNum;
logInfo.GTFilePath = GTFilePath;
logInfo.h5SavePath = h5SavePath;
logInfo.samplesPerImg = samplesPerImg;
logInfo.HazeImageNumPerH5 = HazeImageNumPerH5;
logInfo.patchHei = patchHei;
logInfo.patchWid = patchWid;
logInfo.channel = channel;
logInfo.timeCost = timeCost;

save([h5SavePath, dataSetName, '_log.mat'], 'logInfo');