import tensorlayer as tl
import numpy as np
import os

from utils import *
import PIL.Image as pilImg
from config import config


class Dataset:
    def __init__(self,
                 HR_path,
                 MR_path,
                 LR_path,
                 LLR_path,
                 normalize_mode='max',
                 shuffle_for_epoch=True,
                 **kwargs):

        self.HR_path = HR_path
        self.MR_path = MR_path
        self.LR_path = LR_path
        self.LLR_path = LLR_path
        self.shuffle_all_data = False
        self.shuffle_for_epoch = shuffle_for_epoch
        self.sample_ratio = 1.0
        self.normalize_fn = normalize_percentile
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
            from skimage import io
            img_list = sorted(tl.files.load_file_list(path=path, regx=regx, printable=printable))
            imgs = []
            list_len = int(len(img_list) * self.sample_ratio)
            for img_file in img_list[0:list_len]:
                img = np.expand_dims(io.imread(os.path.join(path, img_file)), axis=-1)
                img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

                if (img.dtype != np.float32):
                    img = img.astype(np.float32, casting='unsafe')
                print('\r%s training data loading: %s -- %s  ---min: %f max:%f' % (
                type_name, img_file, str(img.shape), np.min(img), np.max(img)), end='')
                imgs.append(img)
            return imgs, img_list[0:list_len]

        ###loading
        print('sample ratio: %0.2f' % self.sample_ratio)
        self.training_HR, self.training_HR_list = _load_imgs(self.HR_path, fn=get_2d_imgs,
                                                               normalize_fn=self.normalize_fn,type_name='HR')
        self.training_MR, self.training_MR_list = _load_imgs(self.MR_path, fn=get_2d_imgs,
                                                               normalize_fn=self.normalize_fn,type_name='MR')
        self.training_LR, self.training_LR_list = _load_imgs(self.LR_path, fn=get_2d_imgs,
                                                               normalize_fn=self.normalize_fn,type_name='LR')
        self.training_LLR, self.training_LLR_list = _load_imgs(self.LLR_path, fn=get_2d_imgs,
                                                                 normalize_fn=self.normalize_fn,type_name='LLR')

        if (len(self.training_HR_list) == 0) or (len(self.training_LLR) == 0):
            raise Exception("none of the images have been loaded, please check the file directory in config")
        assert len(self.training_HR_list) == len(self.training_LLR)

        self.training_pair_num = len(self.training_MR)

    def prepare(self, batch_size, n_epochs):
        '''
        this function must be called after the Dataset instance is created
        '''
        if os.path.exists(self.LLR_path) and os.path.exists(self.LR_path):
            self._load_dataset()
        else:
            raise Exception('image data path doesn\'t exist')

        self.test_img_num = int(self.training_pair_num * 0.1)

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.cursor = self.test_img_num
        self.epoch = 0
        print('\nTarget3D dataset : %d\nSynView dataset : %d\nLF dataset: %d\n' % (
        len(self.training_HR_list), len(self.training_LR), len(self.training_LLR)))

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
        return np.asarray(self.training_HR[0: n]), \
            np.asarray(self.training_MR[0: n]), \
            np.asarray(self.training_LR[0: n]), \
            np.asarray(self.training_LLR[0: n])

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

            return np.asarray([self.training_HR[i] for i in shuffle_idx]), \
                np.asarray([self.training_MR[i] for i in shuffle_idx]), \
                np.asarray([self.training_LR[i] for i in shuffle_idx]), \
                np.asarray([self.training_LLR[i] for i in shuffle_idx]), \
                idx - nt, \
                self.epoch, \
                # [self.training_HR[i] for i in shuffle_idx],\
            # [self.training_LR_list[i] for i in shuffle_idx]

        raise Exception('epoch index out of bounds:%d/%d' % (self.epoch, self.n_epochs))
