function [data] = normalize(data,min_p,max_p)

eps=1*1e-7;
if min_p==0
    blk_min = min(data(:));
else
    blk_min   = prctile(data(:), min_p);
end
if max_p==100
    blk_max   = max(data(:));
else
    blk_max   = prctile(data(:), max_p);
end
data = (data-blk_min)./(blk_max-blk_min+eps);


end

