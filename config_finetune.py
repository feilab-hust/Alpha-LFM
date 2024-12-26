import os
from config import config as base_model_settings  # The base model configs

root_path= os.getcwd()
fine_tune_data_path= './data/finetuning'
new_model_name= 'finetune_MitoO_to_MitoMatrix'
gpu_idx = 0
projection_range= 30      # The DoF of WF (uint:slices)

config = base_model_settings
config.new_model_name=new_model_name
config.img_setting.fine_tune_data_path=fine_tune_data_path

config.net_setting.gpu_idx=gpu_idx
config.Pretrain.ckpt_dir = os.path.join(root_path,'checkpoint',config.label)
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
config.Loss.finetune_loss = {'Reprojection_loss': 5.0,
                             }
