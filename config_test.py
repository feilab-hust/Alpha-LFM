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



## passed GUI parameters
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
pass

