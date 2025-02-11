from easydict import EasyDict as edict
config = edict()
config.img_setting = edict()
config.preprocess = edict()
config.net_setting = edict()
config.TRAIN = edict()
config.VALID = edict()

#------------------------------image Setting----------------------------------
config.img_setting.sr_factor = [1,2,2]
config.img_setting.n_channels = 1

label = r'Rab7_x4_[RCAN_multistage]'
config.label = label

#ckpt
config.TRAIN.ckpt_dir             = "checkpoint/{}/".format(label)

## Inference
config.VALID.save_type='LFP'
config.VALID.LLR_path            =r'example_data/'
config.VALID.saving_path          = '{}SR_{}/'.format(config.VALID.LLR_path,label)

