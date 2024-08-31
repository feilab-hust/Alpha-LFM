import numpy as np
import mat73
import os
aa= os.getcwd()
if 'DL' in aa:
    path_str=aa.split('\\')
    path_str=path_str[:-1]
    path_str = ['%s\\'%ii for ii in path_str]
    root_path= ''.join(path_str)
else:
    root_path= aa

train_mat_path=os.path.join(root_path,'./logging/validaiton_settings.mat')
validaiton_settings= mat73.loadmat(train_mat_path)
label = validaiton_settings['validaiton_settings']['prefix']
validation_data_path=validaiton_settings['validaiton_settings']['data_path']


##
presettings = np.load(validaiton_settings['validaiton_settings']['train_config_path'],allow_pickle=True)

config = presettings.item()
config['root_path']=root_path
config['validation_data_path']=validation_data_path
config['label']=label
config['eval_ckpt']=validaiton_settings['validaiton_settings']['eval_ckpt']
config['img_gamma']= validaiton_settings['validaiton_settings']['img_gamma']
# config['eval_ckpt']='20'
pass
# vol_sr_factor=presettings[0]['vol_sr_factor']
# shift_times=presettings[0]['shift_times']
# config.img_setting.img_size = int(training_pairs_setting['traingpairs_settings']['patch_size'])
# config.label = label
# config.root_path=root_path
# config.img_setting.sr_factor = shift_times
# config.img_setting.ReScale_factor = [vol_sr_factor / shift_times, vol_sr_factor / shift_times]
# config.img_setting.Nnum = presettings[1]['Nnum']  # N number of the light field psf
# config.img_setting.n_slices = presettings[1]['depth']
# config.img_setting.data_root_path = h5_data_path
#
#
# # ------------------------------net setting Setting----------------------------------
# config.net_setting.denoise_model = 'LF_attention_denoise'  # SAEPI,LF_attention_denoise
# config.net_setting.SR_model = 'LF_SA_small'  # SA,
# config.net_setting.Recon_model = 'MultiRes_UNet'
# config.net_setting.ngf=[32,64,128]
# config.net_setting.is_bias = False
# config.net_setting.UnetPyrimidNnum=4
#
# # ------------------------------Pretrain Setting----------------------------------
#
# config.Pretrain.loading_pretrain_model=False
# config.Pretrain.ckpt_dir=''
#
# # ------------------------------Preprocess Setting----------------------------------
# LFP_Input_type = 'LFP'
# SynView_Input_type = 'LFP'
# Scan_Input_type = SynView_Input_type
#
# config.preprocess.normalize_mode = 'percentile'  # percentile; constant ; max
# config.preprocess.LFP_type = '1_%s' % (LFP_Input_type)
# config.preprocess.SynView_type = '2_%s' % (SynView_Input_type)
# config.preprocess.discard_view = []  # [1,2,14,15,16,17,29,30,196,197,209,210,211,212,224,225]

