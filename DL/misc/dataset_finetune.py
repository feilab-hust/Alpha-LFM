import numpy as np
import os
from misc.utils import *
from config_finetune import  config
import h5py
class Dataset:
    def __init__(self,
                 base_path,
                 is_hdf5,
                 LFexp_path,
                 wf_path,
                 n_num, normalize_mode='max', shuffle_for_epoch=True, multi_scale=False, **kwargs):

        self.base_path=base_path
        self.is_hdf5=is_hdf5
        self.LFexp_path = LFexp_path
        self.wf_path = wf_path
        self.n_num = n_num
        self.shuffle_all_data = False
        self.shuffle_for_epoch = shuffle_for_epoch
        self.sample_ratio = 1.0

        if normalize_mode == 'normalize_percentile_z_score':
            self.normalize_fn = normalize_percentile_z_score
        elif normalize_mode == 'percentile':
            self.normalize_fn = normalize_percentile
        elif normalize_mode == 'constant':
            self.normalize_fn = normalize_constant
        else:
            self.normalize_fn = normalize

        self.update_parameters(allow_new=True, **kwargs)

    def update_parameters(self, allow_new=False, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def _load_dataset(self, shuffle=True):
        def _load_imgs(path, fn, regx='.*.tif', printable=False, type_name=None, **kwargs, ):
            img_list = sorted(get_file_list(path=path, regx=regx, printable=printable))
            imgs = []

            list_len = int(len(img_list) * self.sample_ratio)
            # print(img_list[0:list_len])
            for img_file in img_list[0:list_len]:
                img = fn(img_file, path, **kwargs)
                if (img.dtype != np.float32):
                    img = img.astype(np.float32, casting='unsafe')
                print('\r%s training data loading: %s -- %s  ---min: %f max:%f' % (
                    type_name, img_file, str(img.shape), np.min(img), np.max(img)), end='')
                imgs.append(img)
            return np.asarray(imgs), img_list[0:list_len]

        ###loading
        print('sample ratio: %0.2f' % self.sample_ratio)

        if self.is_hdf5:
            data_block = h5py.File(os.path.join(self.base_path, 'training_data.h5'))

            sample_len =  int(data_block['nLF'].shape[-1]* self.sample_ratio)
            self.training_lf2d = np.transpose(np.expand_dims(data_block['nLF'][:,:,:sample_len], 0), (3, 2, 1, 0))  # h w b
            self.training_SynView = np.transpose(np.expand_dims(data_block['cLF'][:,:,:sample_len], 0), (3, 2, 1, 0))  # h w b
            self.training_ScanView = np.transpose(np.expand_dims(data_block['sLF'][:,:,:sample_len], 0), (3, 2, 1, 0))  # h w b
            self.training_Target3D = np.transpose(data_block['sVol'][...,:sample_len], (3, 2, 1, 0))  # d w h b

        else:

            self.Target3D_path = os.path.join(self.base_path, 'sVol')
            self.Scan_iew_path = os.path.join(self.base_path, 'sLF')
            self.Synth_view_path = os.path.join(self.base_path, 'cLF')
            self.LFP_path = os.path.join(self.base_path, 'nLF')

            self.training_Target3D, self.training_Target3D_list = _load_imgs(self.Target3D_path, fn=get_3d_stack,
                                                                               normalize_fn=None,
                                                                               type_name='Target3D')
            self.training_ScanView, self.training_ScanView_list = _load_imgs(self.Scan_iew_path, fn=get_2d_lf,
                                                                             normalize_fn=None,
                                                                             type_name='ScanVIew',
                                                                             read_type=config.preprocess.SynView_type,
                                                                             angRes=self.n_num)
            self.training_SynView, self.training_SynView_list = _load_imgs(self.Synth_view_path, fn=get_2d_lf,
                                                                           normalize_fn=None,
                                                                           type_name='SynView',
                                                                           read_type=config.preprocess.SynView_type,
                                                                           angRes=self.n_num)
            self.training_lf2d, self.training_lf2d_list = _load_imgs(self.LFP_path, fn=get_2d_lf, n_num=self.n_num,
                                                                     normalize_fn=None, type_name='LFP',
                                                                     read_type=config.preprocess.LFP_type,
                                                                     angRes=self.n_num)


        self.training_lfexp, self.training_lfexp_list = _load_imgs(self.LFexp_path, fn=get_2d_lf, n_num=self.n_num,
                                                                   normalize_fn=self.normalize_fn, type_name='LFP',
                                                                   read_type=config.preprocess.LFP_type,
                                                                   angRes=self.n_num)
        self.training_wf, self.training_wf_list = _load_imgs(self.wf_path, fn=get_img2d_fn,
                                                             normalize_fn=self.normalize_fn)

        ##check
        if (len(self.training_Target3D) == 0) or (len(self.training_lf2d) == 0):
            raise Exception("none of the images have been loaded, please check the file directory in config")
        assert len(self.training_Target3D) == len(self.training_lf2d)
        return self.training_Target3D,self.training_ScanView,self.training_SynView,self.training_lf2d,self.training_lfexp,self.training_wf

    def for_test(self):
        self.test_img_num = 4
        n = self.test_img_num
        return self.training_Target3D[0: n], \
            self.training_ScanView[0: n], \
            self.training_SynView[0: n], \
            self.training_lf2d[0: n], \
            self.training_lfexp[0: n], \
            self.training_wf[0: n]



