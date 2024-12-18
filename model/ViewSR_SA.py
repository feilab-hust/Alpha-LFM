from .util.utils import *
import tensorlayer as tl
import tensorflow as tf

def conv_block(layer, n_filter, kernel_size,
                is_train=True,
                activation=tf.nn.relu,is_norm=False,
                border_mode="SAME",
                name='conv2d'):
    if is_norm:
        s=conv2d(layer, n_filter=n_filter, filter_size=kernel_size, stride=1, padding=border_mode,name=name+'_conv2d')
        s=batch_norm(s,name=name+'in',is_train=is_train)
        s.outputs=activation(s.outputs)
    else:
        s = conv2d(layer, n_filter=n_filter, filter_size=kernel_size, stride=1, act=activation, padding=border_mode, name=name)
    return s

def Spa2Ang(sp_feature, ang_feature, out_channel, angRes=11,name='Spa2Ang'):
    with tf.variable_scope(name):
        SA1 = conv2d(sp_feature, n_filter=out_channel, filter_size=angRes, stride=angRes, padding='VALID', name='conv1')
        SA1 = concat([ang_feature,SA1],name='SA_concat')
        SA1 = conv2d(SA1,n_filter=out_channel, filter_size=1,padding='VALID', name='conv2')
        SA1 = ReluLayer(SA1,'SA_relu')
        SA1 = merge([ang_feature,SA1],name='SA_merge')
    return SA1
def Ang2Spa(sp_feature, ang_feature, out_channel, angRes=11,name='Ang2Spa'):
    with tf.variable_scope(name):
        AS1 = conv2d(ang_feature, n_filter=angRes*angRes*out_channel, filter_size=1, stride=1, padding='VALID', name='conv1')
        AS1 = SubpixelConv2d(AS1,scale=angRes,name='Upsampling')
        AS1 = concat([sp_feature,AS1],name='AS_concat')
        AS1 = conv2d_dilate(AS1, n_filter=out_channel, filter_size=3, stride=1, dilation=angRes, padding='SAME',name='conv2')
        AS1 = ReluLayer(AS1, 'AS_relu')
        AS1 = merge([sp_feature,AS1],name='AS_merge')
    return AS1

# def Res_path(ngf,length,feature):
#     shorcut=feature
#     shorcut=conv_block(layer=shorcut,n_filter=ngf,activation=tf.identity,kernel_size=1)
def LF2SAI(input_tensor,angRes):
    # LFP realign
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(input_tensor[:, i::angRes, j::angRes, :])
        out.append(tf.concat(out_h, 2))
    out = tf.concat(out, 1)
    # subpixel upscale
    return out

def LF_SA_small(LFP,output_size,angRes=11, sr_factor=7,upscale_mode='one_step',is_train=True, reuse=False, name='unet',**kwargs):
    upscale_factor=sr_factor
    Interact_group_num=2
    Interact_block_num=2
    if 'channels_interp' in kwargs:
      channels_interp = kwargs['channels_interp']
    else:
      channels_interp=64
    if 'normalize_mode' in kwargs:
      normalize_mode = kwargs['normalize_mode']
    else:
      normalize_mode='percentile'
    if 'transform_layer' in kwargs:
        transform_layer=eval(kwargs['transform_layer'])
    else:
        transform_layer = Identity
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('Feature_extra'):
            n = InputLayer(LFP, name='LF')
            n = transform_layer(n,angRes=angRes,name='transform')
            # pre-extract
            ang_feature = conv2d(n, n_filter=channels_interp, filter_size=angRes,stride=angRes,padding='VALID',name='AFE1')             # AFE1
            sp_feature  = conv2d_dilate(n, n_filter=channels_interp, filter_size=3,dilation=angRes,padding='SAME',name='SFE1')          # SFE1
            extract_feature=[]
            extract_feature.append([ang_feature,sp_feature])
            long_skip =sp_feature
            # Interact
            for i in range(Interact_group_num):
                for j in range(Interact_block_num):
                    ang_feature = Spa2Ang(sp_feature=extract_feature[i][1],ang_feature=extract_feature[i][0],out_channel=channels_interp,angRes=angRes,name='S2A_G%d_B%d'%(i,j))
                    sp_feature = Ang2Spa(sp_feature=extract_feature[i][1],ang_feature=extract_feature[i][0], out_channel=channels_interp,angRes=angRes, name='A2S_G%d_B%d'%(i,j))
                    extract_feature.append([ang_feature,sp_feature])

            # Fusion
        with tf.variable_scope(name_or_scope='Bottle_fusion'):

            Ag_list=[extract_feature[i*Interact_block_num+Interact_block_num][0] for i in range(Interact_group_num)]
            Sp_list = [extract_feature[i*Interact_block_num+Interact_block_num][1] for i in range(Interact_group_num)]
            Sp_input = concat(Sp_list, name='sp_in')       #sp feature concat
            Ag_input = concat(Ag_list, name='ag_in')       #ang feature concat

            # angle bottle
            Ag_input = conv2d(Ag_input,n_filter=channels_interp,filter_size=1,padding='VALID',name='conv1')
            Ag_input = ReluLayer(Ag_input, 'relu0')

            #anglur to spatial
            Ag_input = conv2d(Ag_input, n_filter=angRes * angRes * channels_interp, filter_size=1, stride=1, padding='VALID',name='conv2')
            Ag_input = SubpixelConv2d(Ag_input, scale=angRes, name='upscale')

            Sp_input = concat([Ag_input,Sp_input],concat_dim=-1,name='concat')

            Sp_input = conv2d_dilate(Sp_input, n_filter=channels_interp, filter_size=3,dilation=angRes,padding='SAME',name='sfe')
            Sp_out   = ReluLayer(Sp_input, 'relu1')
            n   = merge([Sp_out,long_skip],'add')   # Spout---->batch,h,w,channel_interp

        with tf.variable_scope(name_or_scope='Recon_block'):
            # n = conv2d_dilate(n, n_filter=channels_interp // 2 , filter_size=3, dilation=angRes,padding='SAME', name='SFE_initial')
            if upscale_mode=='multi':
                up_steps = np.int(np.ceil(np.log2(upscale_factor)))
                for idx in range(up_steps):
                    n = conv2d_dilate(n, n_filter=channels_interp * 2 ** 2, filter_size=3, dilation=angRes,padding='SAME', name='SFE_%d'%(idx))
                    n.outputs=tf.depth_to_space(n.outputs,2)       #subpiexl convolution
                    if idx==up_steps-1:
                        channels_interp = 1
                    else:
                        channels_interp = channels_interp // 2
                    n = conv2d(n, filter_size=1, n_filter=channels_interp, name='Reshape_%d'%(idx))
                n.outputs = LF2SAI(n.outputs, angRes=angRes)
            else:
                if sr_factor==2 or sr_factor==4:
                    n=conv2d_dilate(n, n_filter=64, filter_size=3,dilation=angRes,padding='SAME',name='SFE1')          # SFE1
                elif sr_factor == 5:
                    n=conv2d_dilate(n, n_filter=25, filter_size=3,dilation=angRes,padding='SAME',name='SFE1')
                elif sr_factor==3:
                    n=conv2d_dilate(n, n_filter=27, filter_size=3,dilation=angRes,padding='SAME',name='SFE1')
                elif sr_factor==1:
                    n = conv2d_dilate(n, n_filter=16, filter_size=3, dilation=angRes, padding='SAME', name='SFE1')
                n.outputs = LF2SAI(n.outputs, angRes=angRes)
                if sr_factor!=1:
                    n.outputs = tf.depth_to_space(n.outputs,upscale_factor)
                n=conv2d(n,filter_size=1,n_filter=1,padding='VALID',name='final_reshape')

            # if not (n.outputs.get_shape().as_list()[1:3]==output_size).all():
            #     n = UpSampling2dLayer(n, size=output_size, is_scale=False, name='resize_final')

        net_outshape=n.outputs.get_shape().as_list()[1:3]
        assert (net_outshape[0]==output_size[0]) and (net_outshape[1]==output_size[1]),'wrong img size'
        #if normalize_mode=='max':
        n.outputs=tf.tanh(n.outputs)
        return n

