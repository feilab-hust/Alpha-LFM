function [center_LF] = singleLFP(img,H)

    global zeroImageEx;
    global exsize;
    [h,w,~]=size(img);
    xsize = [h, w];
    msize = [size(H,1), size(H,2)];
    mmid = floor(msize/2);
    exsize = xsize + mmid;
    exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];
    zeroImageEx = gpuArray(zeros(exsize, 'single'));
    center_LF=gather(forwardProjectGPU(H,img));

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