from easydict import EasyDict as edict
config = edict()
config.img_setting = edict()
config.preprocess = edict()
config.net_setting = edict()
config.Pretrain = edict()
config.TRAIN = edict()
config.Loss = edict()

# ------------------------------image Setting----------------------------------
config.img_setting.img_size = 32
config.img_setting.sr_factor = [1,2,2]
config.img_setting.n_channels = 1

root_path = r''
config.img_setting.HR = root_path + 'HR'
config.img_setting.MR = root_path + 'MR'
config.img_setting.LR = root_path + 'LR'
config.img_setting.LLR = root_path +'LLR'

# ------------------------------net setting Setting----------------------------------
config.net_setting.is_bias = False
# ------------------------------Label generate----------------------------------
label = r'mymodel'
config.label = label
# ------------------------------Pretrain Setting----------------------------------
config.Pretrain.loading_pretrain_model = False
config.Pretrain.ckpt_dir = ''

# ------------------------------Training Setting----------------------------------
config.TRAIN.test_saving_path = "sample/test/{}/".format(label)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir = "checkpoint/{}/".format(label)
config.TRAIN.log_dir = "log/{}/".format(label)

config.TRAIN.sample_ratio = 1.0
config.TRAIN.shuffle_all_data = False
config.TRAIN.shuffle_for_epoch = True
config.TRAIN.device = 0

# mino
config.TRAIN.batch_size = 1
config.TRAIN.lr_init =1e-4
config.TRAIN.beta1 = 0.9
config.TRAIN.n_epoch = 101
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 50

# ---------------Loss Settings-----------------
config.Loss.denoise_loss = {'mse_loss': 1.0,
                            'mae_loss': 0.1}

config.Loss.SR_loss = {'mse_loss': 1.0,
                            'mae_loss': 0.1}

config.Loss.Recon_loss = {'mse_loss': 1.0,
                          'edge_loss': 0.1
                          }
