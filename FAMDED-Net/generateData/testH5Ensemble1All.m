%testH5Ensemble1
close all; clear; clc;

dataSetName1 = 'OTS_ALPHA';
rootSrc1 = 'RESIDE_DATASET_ROOT/h5Patch_GtHazyTrans/';
h5FileWhole1 = dir([rootSrc1, dataSetName1, '*.h5']);

dataSetName2 = 'ITS';
rootSrc2 = 'RESIDE_DATASET_ROOT/h5Patch_GtHazyTrans/';
h5FileWhole2 = dir([rootSrc2, dataSetName2, '*.h5']);

rootDst = 'RESIDE_DATASET_ROOT/h5Patch_GtHazyTrans_EnsembleAllSmall/';
if ~exist(rootDst, 'dir')
    mkdir(rootDst);
end


patchHei = 128;
patchWid = patchHei;
channels = 3;
h5Num = length(h5FileWhole1);
count = 1;
for h5Iter = 1:h5Num
    tic;
    disp(['currently processing ', num2str(h5Iter), 'th h5...']);
    
    info1 = h5info([rootSrc1, h5FileWhole1(h5Iter+h5Num).name]);
    data1 = h5read([rootSrc1, h5FileWhole1(h5Iter).name], '/data', ...
        [1, 1, 1, 1], info1.Datasets(1).Dataspace.Size);
    label1 = h5read([rootSrc1, h5FileWhole1(h5Iter).name], '/label', ...
        [1, 1, 1, 1], info1.Datasets(2).Dataspace.Size);
    sampleNum4 = size(data1,4);
    
    info2 = h5info([rootSrc2, h5FileWhole2(h5Iter).name]);
    data2 = h5read([rootSrc2, h5FileWhole2(h5Iter).name], '/data', ...
        [1, 1, 1, 1], info2.Datasets(1).Dataspace.Size);
    label2 = h5read([rootSrc2, h5FileWhole2(h5Iter).name], '/label', ...
        [1, 1, 1, 1], info2.Datasets(2).Dataspace.Size);
    sampleNum3 = size(data2,4);
    
    if sampleNum1 ~= sampleNum2
        sampleNum = min(sampleNum1, sampleNum2);
        data1 = data1(:,:,:,1:sampleNum);
        label1 = label1(:,:,:,1:sampleNum);
        
        data2 = data2(:,:,:,1:sampleNum);
        label2 = label2(:,:,:,1:sampleNum);

    end
    
    data = cat(4, data1, data2,);
    label = cat(4, label1, label2);
    sampleNum = size(data,4);
    idxRand = randperm(sampleNum);
    data = data(:,:,:,idxRand);
    label = label(:,:,:,idxRand);
    
    h5SplitNum = 1;
    for i = 1:h5SplitNum
        %--------------------write h5--------------------
        h5Name = [rootDst, dataSetName1, '_', dataSetName1, '_', dataSetName2, '_', num2str(count), '.h5'];
        if exist(h5Name, 'file')
            delete(h5Name);
            disp(['delete existing h5 file successfully.'])
        end
        
        chunksz = 64;
        created_flag = false;
        totalct = 0;
        chunkPerH5 = floor(sampleNum / chunksz / h5SplitNum)
        batchnoShift = (i-1) * chunkPerH5 + 1;
        for batchno = batchnoShift:batchnoShift+chunkPerH5-1
            last_read = (batchno-1) * chunksz;
            batchdata = data(:,:,:,last_read+1:last_read+chunksz);
            batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
            
            startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
            curr_dat_sz = store2hdf5(h5Name, batchdata, batchlabs, ~created_flag, startloc, chunksz);
            created_flag = true;
            totalct = curr_dat_sz(end);
        end
        
        h5disp(h5Name);
        count = count + 1;
        
    end
    toc;
    
end

