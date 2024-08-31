function [coordi] = get_coordi(data,window_sz,thresh_mean,Nnum,sr_ratio)


%% padding lateral
[H,W,~]= size(data);

%% rectify
base_h_lf= H/sr_ratio/Nnum;
base_w_lf= W/sr_ratio/Nnum;

base_window_sz=window_sz/sr_ratio/Nnum;
base_stride = ceil(base_window_sz/2);

x_boundry=[ceil(base_window_sz/2),base_w_lf];
y_boundry=[ceil(base_window_sz/2),base_h_lf];

data_mean= mean(data(:));
coordi=[];

h_wrap_flag=1;
for y_idx=y_boundry(1):base_stride:y_boundry(2)
    w_wrap_flag=1;
    h_s=y_idx-ceil(base_window_sz/2)+1;
    h_e=h_s+base_window_sz-1;
    glb_h_e= h_e*sr_ratio*Nnum;
    glb_h_s= glb_h_e-base_window_sz*sr_ratio*Nnum+1;

    if glb_h_e>H
        if h_wrap_flag
            glb_h_e=H;
            glb_h_s=H-window_sz+1;
            h_wrap_flag=0;
        else
            break
        end
    end

    for x_idx=x_boundry(1):base_stride:x_boundry(2)
        w_s=x_idx-ceil(base_window_sz/2)+1;
        w_e=w_s+base_window_sz-1;
        
        glb_w_e= w_e*sr_ratio*Nnum;
        glb_w_s= glb_w_e-base_window_sz*sr_ratio*Nnum+1;
        if glb_w_e>W
            if  w_wrap_flag
                glb_w_e=W;
                glb_w_s=W-window_sz+1;
                w_wrap_flag=0;
            else
                break
            end
        end

        ROI_block =  data(glb_h_s:glb_h_e,glb_w_s:glb_w_e,:);
        %write3d(ROI_block,sprintf('%d_%d.tif',y_idx,x_idx),32);
        if mean(ROI_block(:))>=thresh_mean*data_mean
            coordi=[coordi;[glb_h_s,glb_h_e,glb_w_s,glb_w_e]]; %#ok<AGROW> 
        end
    end
end
end












