import tensorflow as tf
import tifffile

import tensorlayer as tl
import numpy as np
import os
import time
from model import *
from utils import *
from config_test import config


#---------------parameters-------------
label = config.label
sr_factor_list = config.img_setting.sr_factor
normalize_fn = normalize_percentile


def read_valid_images(path):
    def __cast(im, dtype=np.float32):
        return im if im.dtype is np.float32 else im.astype(np.float32, casting='unsafe')

    img_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
    img_set = [__cast(get_2d_imgs(img_file, path,normalize_fn=normalize_fn)) for img_file in img_list]
    print('read %d from %s' % (len(img_set), path))
    img_set = np.asarray(img_set)
    _, height, width, _ = img_set.shape
    return img_set, img_list, height, width


def infer(epoch, batch_size=1):

    epoch = 'best' if epoch == 0 else epoch
    checkpoint_dir = config.TRAIN.ckpt_dir
    valid_lr_img_path = config.VALID.LLR_path
    save_dir = config.VALID.saving_path
    tl.files.exists_or_mkdir(save_dir)
    valid_lf_extras, names, height, width = read_valid_images(valid_lr_img_path)
    t_image = tf.placeholder('float32', [batch_size, height, width, 1])

    tag_list = ['DenoiseNet', 'SRNet', 'ReconNet']
    with tf.device('/gpu:0'):
        denoise_net = RCAN(lr=t_image, sr_factor=sr_factor_list[0], format_out=True, reuse=False, name=tag_list[0])
        SR_net = RCAN(lr=denoise_net.outputs, sr_factor=sr_factor_list[1], format_out=True, reuse=False, name=tag_list[1])
        Recon_net = RCAN(lr=SR_net.outputs, sr_factor=sr_factor_list[2], format_out=True, reuse=False, name=tag_list[2])


    denoise_ckpt= [filename for filename in os.listdir(checkpoint_dir) if
                    ('.npz' in filename and epoch in filename and 'denoise' in filename)]
    SR_ckpt_file = [filename for filename in os.listdir(checkpoint_dir) if
                    ('.npz' in filename and epoch in filename and 'sr' in filename)]
    Recon_ckpt_file = [filename for filename in os.listdir(checkpoint_dir) if
                    ('.npz' in filename and epoch in filename and 'recon' in filename)]

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, denoise_ckpt[0]),
                                     network=denoise_net)
        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, SR_ckpt_file[0]),
                                     network=SR_net)
        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, Recon_ckpt_file[0]),
                                     network=Recon_net)

        for idx in range(0, len(valid_lf_extras), batch_size):
            recon_out = sess.run(Recon_net.outputs, {t_image: valid_lf_extras[idx:idx + batch_size]})
            print("\rvalidation on %s " % (names[idx]), end='')
            recon_out = np.clip(recon_out,a_min=0,a_max=1)
            recon_out = np.squeeze((recon_out-np.amin(recon_out))/(np.amax(recon_out)-np.amin(recon_out)))*255
            tifffile.imwrite(save_dir + '%s-%s' % ('Pred_', names[idx]),recon_out.astype(np.uint8))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=int, default=0)
    args = parser.parse_args()
    ckpt = args.ckpt
    infer(ckpt)

