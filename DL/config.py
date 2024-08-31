from easydict import EasyDict as edict
import mat73
import os
from misc.utils import add_delimiter
import json

def configs_settings(training_settings_path):
    config = edict()
    config.img_setting = edict()
    config.preprocess = edict()
    config.net_setting = edict()
    config.Pretrain = edict()
    config.TRAIN = edict()
    config.Loss = edict()
    aa= os.getcwd()
    if 'DL' in aa:
        path_str=aa.split('\\')
        path_str=path_str[:-1]
        path_str = ['%s\\'%ii for ii in path_str]
        root_path= ''.join(path_str)
    else:
        root_path= aa
    print('--------------root_path:%s---------'%root_path)
    train_mat_path=os.path.join(root_path,training_settings_path)
    if not os.path.exists(train_mat_path):
        raise ValueError("No TrainingSetings found")

    training_pairs_setting= mat73.loadmat(train_mat_path)
    label = training_pairs_setting['traingpairs_settings']['label']
    gpu_idx = int(training_pairs_setting['traingpairs_settings']['gpu_idx'])

    json_path =training_pairs_setting['traingpairs_settings']['json_file_path']
    real_input = os.path.join(root_path,'logging',"data_temp.json")
    add_delimiter(json_path, real_input)
    with open(real_input, "r", encoding="utf-8") as fr:
        presettings= json.load(fr)
    vol_sr_factor=presettings[0]['vol_sr_factor']
    shift_times=presettings[0]['shift_times']

    config.img_setting.img_size = int(training_pairs_setting['traingpairs_settings']['patch_size'])
    config.label = label
    config.root_path=root_path
    config.img_setting.sr_factor = shift_times
    config.img_setting.ReScale_factor = [vol_sr_factor / shift_times, vol_sr_factor / shift_times]
    config.img_setting.Nnum = presettings[1]['Nnum']                                # N number of the light field psf
    config.img_setting.n_slices = presettings[1]['enhanced_n_slices']

    config.img_setting.data_root_path = os.path.join(root_path,presettings[0]['Training_data_dir'])
    config.img_setting.save_hdf5 = presettings[0]['save_hdf5']
    config.img_setting.save_bit = presettings[0]['save_bit']
    # ------------------------------net setting Setting----------------------------------
    config.net_setting.gpu_idx=gpu_idx
    config.net_setting.denoise_model = 'LF_attention_denoise' 
    config.net_setting.SR_model = 'LF_SA_small'  
    config.net_setting.Recon_model = 'MultiRes_UNet'
    config.net_setting.ngf=[32,64,128]                 # Unet channel
    config.net_setting.is_bias = False
    config.net_setting.Unetpyrimid_list=[128,256,512,512,512]  

    # ------------------------------Pretrain Setting----------------------------------
    config.Pretrain.loading_pretrain_model=training_pairs_setting['traingpairs_settings']['pretrain_flag']
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

    config.TRAIN.sample_ratio = 1.0
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
    return config