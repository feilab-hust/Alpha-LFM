import numpy as np
import mat73
import copy
import os
from config import config as train_configs

label = 'lysosome_enhanced'
validation_data_path= r'./example/validation_data/lyso'

config = copy.deepcopy(train_configs)
config['root_path']=os.getcwd()
config['validation_data_path']=validation_data_path
config['label']=label
config['eval_ckpt']= 'best'
config['img_gamma']= 0.8


