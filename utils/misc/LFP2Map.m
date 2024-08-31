function [view_map] = LFP2Map(signal,n_num)
    [h,w]=size(signal);
    base_h=floor(h/n_num);
    base_w=floor(w/n_num);
    view_map=zeros(h,w);
    for i=1:n_num
        for j=1:n_num
            temp_view=signal(i: n_num: end, j: n_num: end);
            view_map((i-1)*base_h+1:i*base_h, (j-1)*base_w+1:j*base_w)=temp_view;
        end
    end
end

