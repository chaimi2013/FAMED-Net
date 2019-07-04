%tesWriteList
close all; clear; clc;

rootH5 = 'RESIDE_DATASET_ROOT/h5Patch_GtHazyTrans/';
file = 'h5List.txt';

fileWhole = dir([rootH5, '*.h5']);
h5Num = length(fileWhole);
% h5Num = 64;

fid = fopen([rootH5, file], 'wt');
sampleNum = zeros(1, h5Num);
for i = 1:h5Num
    fprintf(fid, '%s\n', [rootH5, fileWhole(i).name]);
    info = h5info([rootH5, fileWhole(i).name]);
%     info.Datasets(1).Dataspace.Size
%     sampleNum(i) = inrrfo.Datasets(1).Dataspace.Size(4) / 64
end
fclose(fid);