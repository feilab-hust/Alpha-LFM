function [new_data] = Padding_Lateral(data,lateral_size)
    [H,W,depth]=size(data);
    padding_flag=0;
    if H <lateral_size
        H_p = lateral_size;
        padding_flag=1;
    else
        H_p= H;
    end
    if W <lateral_size
        W_p = lateral_size;
        padding_flag=1;
    else
        W_p = W;
    end
    if padding_flag==1
        new_data = zeros([H_p,W_p,depth],'single');
        new_data(1:H,1:W,:)=data;
    else
        new_data=data;
    end
end