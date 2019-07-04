function dark = minFilter2(img, r)
% function dark = minFilter(img, r)

[hei,wid,c] = size(img);
dark = zeros(hei,wid);

for i = 1:2*r:hei
    for j = 1:2*r:wid
        patch = img(i:min(i+2*r,hei), j:min(j+2*r,wid), :);
        dark(i:min(i+2*r,hei), j:min(j+2*r,wid))= min(patch(:));
    end
end