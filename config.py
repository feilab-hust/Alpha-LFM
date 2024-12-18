from easydict import EasyDict as edict
import mat73
import os
from misc.utils import add_delimiter
import json

config = edict()
config.img_setting = edict()
config.preprocess = edict()
config.net_setting = edict()
config.Pretrain = edict()
config.TRAIN = edict()
config.Loss = edict()

root_path= os.getcwd()
print('--------------root_path:%s---------'%root_path)

# ------------------------------net setting Setting----------------------------------
label = 'Example_training_debug'  # The name of folder that contains the trained weights
vol_sr_factor=2
shift_times=5
config.img_setting.img_size = 360
config.label = label
config.root_path=root_path
config.img_setting.sr_factor = shift_times
config.img_setting.ReScale_factor = [vol_sr_factor / shift_times, vol_sr_factor / shift_times]
config.img_setting.Nnum = 15          # N number of the light field psf
config.img_setting.n_slices = 161     # Z-slices of 3D target

config.img_setting.data_root_path = r'J:\YCQ_TEMP\LF\NC_LFM\rab_trainingpair_Data\rab_base_data_720\S01_TrainingData'   # The training data directory
config.img_setting.save_hdf5 = False      # The training data directory
config.img_setting.save_bit = 16
# ------------------------------Net Setting----------------------------------
config.net_setting.gpu_idx=0
config.net_setting.denoise_model = 'LF_attention_denoise'
config.net_setting.SR_model = 'LF_SA_small'
config.net_setting.Recon_model = 'MultiRes_UNet'
config.net_setting.ngf=[32,64,128]                 # Unet channel
config.net_setting.is_bias = False
config.net_setting.Unetpyrimid_list=[128,256,512,512,512]

# ------------------------------Pretrain Setting----------------------------------
config.Pretrain.loading_pretrain_model=True
config.Pretrain.Training_epoch=[51,101,151]
config.local_pre_SRVCD_dict={
        'lr_init':5*1e-4,
        'decay_every':50,
        'lr_decay':0.5,
        'ckpt_save':'./SR_VCD_pre',
        'sample_save': './SR_VCD_pre',
        'n_epoch':101,
    }

# ------------------------------Preprocess Setting----------------------------------
config.preprocess.normalize_mode = 'percentile' if config.img_setting.save_bit!=32 else None
config.preprocess.discard_view = []
# ------------------------------Training Setting----------------------------------
config.TRAIN.to_Disk=False
config.TRAIN.test_saving_path = "sample/test/{}/".format(label)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir = "checkpoint/{}/".format(label)
config.TRAIN.log_dir = "log/{}/".format(label)
config.TRAIN.valid_on_the_fly = False

config.TRAIN.sample_ratio = 0.2
config.TRAIN.shuffle_all_data = False
config.TRAIN.shuffle_for_epoch = True
config.TRAIN.device = 0

# mino
config.TRAIN.batch_size = 1
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9
config.TRAIN.n_epoch = 101
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 25
# ---------------Loss Settings-----------------

config.Loss.Ratio = [0.1, 0.2, 0.8]

config.Loss.denoise_loss = {'mse_loss': 1.0}

config.Loss.SR_loss = {'mse_loss': 1.0,
                       'EPI_mse_loss': 0.1
                       }
config.Loss.Recon_loss = {'mse_loss': 1.0,
                          'edge_loss': 0.1
                          }
