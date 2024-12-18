import numpy as np
import os
import h5py
from misc.utils import *
class Dataset:
    def __init__(self, base_path, data_str, n_slices, n_num, lf2d_base_size, normalize_mode='max',
                 shuffle_for_epoch=True, **kwargs):

        self.base_path=base_path
        self.data_str=data_str
        self.lf2d_base_size = lf2d_base_size
        self.n_slices = n_slices
        self.n_num = n_num
        self.shuffle_all_data = False
        self.save_hdf5 =True
        if normalize_mode == 'normalize_percentile_z_score':
            self.normalize_fn = normalize_percentile_z_score
        elif normalize_mode == 'percentile':
            self.normalize_fn = normalize_percentile
        elif normalize_mode == None:
            self.normalize_fn = None
        elif normalize_mode == 'constant':
            self.normalize_fn = normalize_constant
        else:
            self.normalize_fn = normalize
        self.shuffle_for_epoch = shuffle_for_epoch
        self.sample_ratio = 1.0
        # define mask path
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
            for img_file in img_list[0:list_len]:
                img = fn(img_file, path, **kwargs)
                if (img.dtype != np.float32):
                    img = img.astype(np.float32, casting='unsafe')
                print('\r%s training data loading: %s -- %s  ---min: %f max:%f' % (
                type_name, img_file, str(img.shape), np.min(img), np.max(img)), end='')
                imgs.append(img)
            return imgs

        ###loading

        print('sample ratio: %0.2f' % self.sample_ratio)
        if self.save_hdf5:
            data_block = h5py.File(self.base_path, 'r')
            self.training_Target3D = np.transpose(data_block[self.data_str[0]], (3, 2, 1, 0))  # d w h b
            self.training_ScanView = np.transpose(np.expand_dims(data_block[self.data_str[1]], 0), (3, 2, 1, 0))  # h w b
        else:
            self.training_Target3D = _load_imgs(os.path.join(self.base_path,self.data_str[0]),
                                                fn=get_3d_stack,normalize_fn=self.normalize_fn,type_name=self.data_str[0])
            self.training_ScanView = _load_imgs(os.path.join(self.base_path,self.data_str[1]),
                                            fn=get_2d_lf, n_num=self.n_num,
                                       normalize_fn=self.normalize_fn, type_name=self.data_str[1],
                                       read_type='1_%s' % ('LFP'), angRes=self.n_num)


        ##check
        if (len(self.training_Target3D) == 0) or (len(self.training_ScanView) == 0):
            raise Exception("none of the images have been loaded, please check the file directory in config")
        assert len(self.training_ScanView) == len(self.training_Target3D)

        self.training_pair_num = len(self.training_Target3D)



    def prepare(self, batch_size, n_epochs):
        '''
        this function must be called after the Dataset instance is created
        '''
        # loading imgs
        self._load_dataset()

        self.test_img_num = int(self.training_pair_num * 0.1)
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.cursor = self.test_img_num
        self.epoch = 0
        print('\nHR dataset : %d\nLF dataset: %d\n' % (
        len(self.training_Target3D), len(self.training_ScanView),))

        data_shuffle_matrix = []

        for idx in range(self.n_epochs + 1):
            temp = np.arange(0, self.test_img_num, dtype=np.int32)
            temp = np.append(temp,
                             np.random.permutation(self.training_pair_num - self.test_img_num) + self.test_img_num)

            if self.shuffle_for_epoch == True:
                data_shuffle_matrix.append(temp)
            else:
                temp.sort()
                data_shuffle_matrix.append(temp)

        self.data_shuffle_matrix = np.stack(data_shuffle_matrix, axis=0)
        return self.training_pair_num - self.test_img_num

    def for_test(self):
        n = self.test_img_num
        return np.asarray(self.training_Target3D[0: n]), np.asarray(self.training_ScanView[0: n])
    def hasNext(self):
        return True if self.epoch < self.n_epochs else False

    def iter(self):
        '''
        return the next batch of the training data
        '''
        nt = self.test_img_num
        if self.epoch < self.n_epochs:
            if self.cursor + self.batch_size > self.training_pair_num:
                self.epoch += 1
                self.cursor = nt

            idx = self.cursor
            end = idx + self.batch_size
            self.cursor += self.batch_size
            shuffle_idx = self.data_shuffle_matrix[self.epoch][idx:end]
            hr_pyramid = [self.training_Target3D[i] for i in shuffle_idx]
            lr_pyramid = [self.training_ScanView[i] for i in shuffle_idx]
            return hr_pyramid[0][None,...],lr_pyramid[0][None,...], idx - nt, self.epoch
        raise Exception('epoch index out of bounds:%d/%d' % (self.epoch, self.n_epochs))
