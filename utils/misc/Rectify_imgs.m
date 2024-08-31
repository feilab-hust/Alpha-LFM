function [temp_img] = Rectify_imgs(data,Nnum,sr_ratio,padding_slices)
[h,w,sig_len]=size(data);
new_h=  (ceil( (ceil(h/sr_ratio/Nnum)-1)/2)*2+1)*sr_ratio*Nnum;
new_w=  (ceil( (ceil(w/sr_ratio/Nnum)-1)/2)*2+1)*sr_ratio*Nnum;
temp_img=zeros(new_h,new_w,padding_slices,'single');
if sig_len==padding_slices
    temp_img(1:h,1:w,:)=data;
elseif sig_len<padding_slices
    offset=floor((padding_slices-sig_len)/2);
    temp_img(1:h,1:w,offset+1:offset+sig_len)=data;
else
    offset=floor((sig_len-padding_slices)/2);
    temp_img(1:h,1:w,:)=data(:,:,offset+1:offset+padding_slices);
end
 
        
end












