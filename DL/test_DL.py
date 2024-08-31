''''
This file is used to check whether the Deep-learning codes work properly.
'''
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.disable(logging.INFO)
os.environ["CUDNN_LOGINFO_DBG"] = '0'

print(50 * '-')
print('[0] Import tensorflow')
print(50 * '-')
try:
    import tensorflow as tf
except Exception as e:
    # print('------------------------')
    # print('Errors when importing tensorflow')
    # print('------------------------')
    raise e
else:
    print(50*'-')
    print("TensorFlow version:", tf.__version__)
    print('current path:', os.getcwd())
    print(50*'-')
    print('[1] Check the GPU is avaliable or not')
    print(50*'-')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    if tf.test.gpu_device_name():
        from tensorflow.python.client import device_lib
        def get_available_gpus():
            local_device_protos = device_lib.list_local_devices()
            return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
        [print(_s) for _s in get_available_gpus()]
        print('Default GPU: %s'%(tf.test.gpu_device_name()))
    else:
        print("No GPU available")
        print(50 * '-')
        # sys.exit()

    # varibles intialization
    normalize_mode='percentile'
    input_size = [360,360]
    SR_size = [1800,1800]
    Recon_size = [720,720]
    sr_factor =5
    n_slices =161
    ang_res = 15
    img_data = np.asarray(np.random.random([1,*input_size,1]),dtype=np.float32)


    # sr1 recon
    sr_data = np.asarray(np.random.random([1,*SR_size,1]),dtype=np.float32)
    recon_data = np.asarray(np.random.random([1,*Recon_size,n_slices]),dtype=np.float32)

    # build graph
    input_tensor = tf.placeholder('float32',[1,*input_size,1],'input_LF')
    from model import LF_attention_denoise,LF_SA_small,MultiRes_UNet
    ngf1,ngf2,ngf3 = [32,64,128]
    Unetpyrimid_list=[128,256,512,512,512]
    # Unetpyrimid_list=[128,256]

    print(50*'-')
    print('[2] Network building')
    print(50*'-')
    denoise_net = LF_attention_denoise(LFP=input_tensor, output_size=input_size, sr_factor=1, angRes=ang_res,
                                     reuse=False, channels_interp=ngf1, name='Atten_denoise')
    SR_net = LF_SA_small(LFP=denoise_net.outputs, output_size=SR_size, sr_factor=sr_factor,
                           angRes=ang_res, reuse=False, name='SA', channels_interp=ngf2,
                           normalize_mode=normalize_mode, transform_layer='SAI2Macron')
    Recon_net = MultiRes_UNet(lf_extra=SR_net.outputs, n_slices=n_slices, output_size=Recon_size,
                                 is_train=True, reuse=False, name='MultiRes', channels_interp=ngf3,
                                 normalize_mode=normalize_mode, transform='SAI2ViewStack',
                                 pyrimid_list=Unetpyrimid_list)

    # loss_compute_op =

    print(50*'-')
    print('[-] Network has been built')
    print(50*'-')
    # network
    print(50*'-')
    print('[3] Start network inference')
    print(50*'-')
    configProto = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    configProto.gpu_options.allow_growth = True
    try:
        with tf.Session(config=configProto) as sess:
            sess.run(tf.global_variables_initializer())
            test_out = sess.run(Recon_net.outputs,{input_tensor:img_data})
            print(50*'-')
            print('[-] Network Inference Success')
            print(50*'-')
            del test_out
    except Exception as e:
        raise e


    print(50*'-')
    print('[4] Test network training')
    print(50*'-')
    try:
        with tf.Session(config=configProto) as sess:
            sess.run(tf.global_variables_initializer())
            test_out = sess.run(Recon_net.outputs,{input_tensor:img_data})
            print(50*'-')
            print('[-] Network Inference Success')
            print(50*'-')
            del test_out
    except Exception as e:
        raise e