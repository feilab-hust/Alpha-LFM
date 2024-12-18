import numpy as np
import json
from easydict import EasyDict as edict
from misc.utils import add_delimiter

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

print('--------------root_path:%s---------'%root_path)
config_mat_path=os.path.join(root_path,'./logging/fine_tune_settings.mat')
fine_tuning_mat= mat73.loadmat(config_mat_path)


fine_tune_data_path= fine_tuning_mat['fine_tune_settings']['new_data_path']
base_model_settings = fine_tuning_mat['fine_tune_settings']['base_mode_config_path']
base_model_settings = np.load(base_model_settings,allow_pickle=True).item()
new_model_name= fine_tuning_mat['fine_tune_settings']['new_model_name']
gpu_idx = int(fine_tuning_mat['fine_tune_settings']['gpu_idx'])
projection_range=fine_tuning_mat['fine_tune_settings']['projection_range']

## updating the base_model parameters
# update
# %% new_data_path,gpu idx %% pretrain ckpt path
# %% sample ratio %% training epoch %% loss settinngs
config = edict(base_model_settings)

config.new_model_name=new_model_name
config.img_setting.fine_tune_data_path=fine_tune_data_path


config.net_setting.gpu_idx=gpu_idx
config.Pretrain.ckpt_dir = os.path.join(root_path,'DL','checkpoint',config.label)
config.TRAIN.sample_ratio = 0.25

config.TRAIN.test_saving_path = "sample/test/{}/".format(new_model_name)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir = "checkpoint/{}/".format(new_model_name)
config.TRAIN.log_dir = "log/{}/".format(new_model_name)


config.TRAIN.n_epoch = 101

config.Loss.projection_range =projection_range

config.Loss.Ratio = [0.1, 0.2, 0.8, 1]
config.Loss.denoise_loss = {'mse_loss': 1.0}
config.Loss.SR_loss = {'mse_loss': 1.0,
                       'EPI_mse_loss': 0.1
                       }
config.Loss.Recon_loss = {'mse_loss': 1.0,
                          # 'edge_loss': 0.1
                          }
config.Loss.finetune_loss = {'wf_loss_mix': 5.0,
                             }
