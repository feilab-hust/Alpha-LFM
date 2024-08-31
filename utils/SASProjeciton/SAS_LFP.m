function [DA_LF,center_LF] = SAS_LFP(img,H,shift_times)

center_point=ceil(shift_times/2);
shift_step=size(H,3)/shift_times;
[ori_h,ori_w,depth]=size(img);
Nnum=size(H,3);

new_h=(1-mod(ori_h/Nnum,2))*Nnum+ori_h;
new_w=(1-mod(ori_w/Nnum,2))*Nnum+ori_w;
pd_wf=zeros(new_h,new_w,depth,'single');
pd_wf(1:ori_h,1:ori_w,:)=img;


img= pd_wf;
[h,w,~]=size(img);

global zeroImageEx;
global exsize;
xsize = [h, w];
msize = [size(H,1), size(H,2)];
mmid = floor(msize/2);
exsize = xsize + mmid;
exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];
zeroImageEx = gpuArray(zeros(exsize, 'single'));
LFP_list=zeros([h,w,shift_times*shift_times],"single");
for v_idx=1:shift_times
    for u_idx=1:shift_times
        x_shift_value=(u_idx-center_point)*shift_step;
        y_shift_value=(v_idx-center_point)*shift_step;
        z_shift_value=0;
        se=imtranslate(img,[y_shift_value x_shift_value  z_shift_value]);
        %         se=se + offset ;
        view_idx=(v_idx-1)*shift_times+u_idx;
        %         tic
        LFP_list(:,:,view_idx)=forwardProjectGPU(H,se);
        %         toc
    end
end

center_LF=LFP_list(:,:,ceil(shift_times*shift_times/2));
center_LF=center_LF(1:ori_h,1:ori_w);
%%
% reoder
Nnum=size(H,3);
base_h=size(LFP_list,1)/Nnum;
base_w=size(LFP_list,2)/Nnum;
temp_indx=1:shift_times;
index1=reshape(repmat(temp_indx,[shift_times,1]),[1,shift_times*shift_times]);
index2=repmat(temp_indx,[1,shift_times]);
shift_4D=zeros( size(LFP_list,1),size(LFP_list,2),shift_times,shift_times,'single');
for view_idx=1:shift_times*shift_times
    shift_4D(:,:,index2(view_idx),index1(view_idx))= LFP_list(:,:,view_idx);
end
% Realignment
shfit_viewStack=zeros(Nnum,Nnum,base_h,base_w,shift_times,shift_times,'single');
for i=1:Nnum
    for j=1:Nnum
        for a=1:base_h
            for b=1:base_w
                shfit_viewStack(i,j,a,b,:,:)=squeeze(shift_4D((a-1)*Nnum+i,(b-1)*Nnum+j,:,:));
            end
        end
    end
end
shfit_viewMap=zeros(  base_h*shift_times,base_w*shift_times,Nnum,Nnum,"single");
for a=1:base_h
    for s_i=1:shift_times
        x=shift_times*a+1-s_i;
        for b=1:base_w
            for s_j=1:shift_times
                y=shift_times*b+1-s_j;
                %                 fprintf('x:%d // y:%d\n',x,y);
                shfit_viewMap(x,y,:,:)=squeeze(shfit_viewStack(:,:,a,b,s_i,s_j));
            end
        end
    end
end

ViewStack=zeros( base_h*shift_times,base_w*shift_times,Nnum*Nnum ,"single");
for ii=1:Nnum
    for jj=1:Nnum
        view_idxxx=(ii-1)*Nnum+jj;
        ViewStack(:,:,view_idxxx)=shfit_viewMap(:,:,ii,jj);
    end
end
DA_LF= Stack2LFP(ViewStack,Nnum);
DA_LF= DA_LF(1:ori_h*shift_times,1:ori_w*shift_times);
end

function TOTALprojection = forwardProjectGPU( H, realspace )

    global zeroImageEx;
    global exsize;

    Nnum = size(H,3);
    zerospace = gpuArray.zeros(  size(realspace,1),   size(realspace,2), 'single');
    TOTALprojection = gpuArray.zeros(  size(realspace,1),   size(realspace,2), 'single');

    for aa=1:Nnum,
        for bb=1:Nnum,
            for cc=1:size(realspace,3),
                Hs = gpuArray(squeeze(H( :,:,aa,bb,cc)));
                tempspace = zerospace;
                tempspace( (aa:Nnum:end), (bb:Nnum:end) ) = realspace( (aa:Nnum:end), (bb:Nnum:end), cc);
                projection = conv2FFT(tempspace, Hs);
                TOTALprojection = TOTALprojection + projection;

            end
        end
    end


end




