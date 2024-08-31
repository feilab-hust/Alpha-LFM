import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.disable(logging.INFO)
os.environ["CUDNN_LOGINFO_DBG"] = '0'
import numpy as np
import tifffile
import PIL.Image as pilimg
import tensorlayer as tl
import  os
from skimage import io
import mat73
import re
__all__ = [
    'get_img3d_fn',
    'rearrange3d_fn',
    'get_and_rearrange3d',
    'get_img2d_fn',
    'get_lf_extra',
    'get_2d_lf',
    'lf_extract_fn',
    'write3d',
    'normalize_percentile',
    'normalize',
    'normalize_percentile_z_score',
    'min_max',
    'normalize_constant',
    'save_configs',
    'binary_normal',
    '_raise',
    'add_delimiter',
    'get_3d_stack',
    'get_file_list'

]

def get_file_list(path, regx='.*.tif', printable=False):
    if path is None:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    return return_list


def _raise(e):
    raise (e)

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False
def save_configs(save_folder,cg):
    configs = {key: value for key, value in cg.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
    np.save(os.path.join(save_folder, 'training_configs'),configs)

def get_2d_lf(filename, path, normalize_fn, **kwargs):
    def _LFP2ViewMap(img, angRes):
        img = np.squeeze(img)
        h, w = img.shape
        base_h = h // angRes
        base_w = w // angRes
        VP_ = np.zeros(img.shape, np.float32)
        for v in range(angRes):
            for u in range(angRes):
                VP_[v * base_h:(v + 1) * base_h, u * base_w:(u + 1) * base_w] = img[v::angRes, u::angRes]
        return VP_

    def _ViewMap2LFP(img, angRes):
        img = np.squeeze(img)
        h, w = img.shape
        base_h = h // angRes
        base_w = w // angRes
        LFP_ = np.zeros(img.shape, np.float32)
        for v in range(angRes):
            for u in range(angRes):
                LFP_[v::angRes, u::angRes] = img[v * base_h:(v + 1) * base_h, u * base_w:(u + 1) * base_w]
        return LFP_

    def _identity(img, angRes):
        return img

    image = tifffile.imread(os.path.join(path,filename))
    if 'read_type' in kwargs:
        read_type = kwargs['read_type']
    else:
        read_type = None

    if read_type is not None:
        assert 'ViewMap' in read_type or 'LFP' in read_type, 'wrong img type'
        if '1' in read_type:
            trans_func = _identity if 'LFP' in read_type else _ViewMap2LFP
        elif '2' in read_type:
            trans_func = _identity if 'ViewMap' in read_type else _LFP2ViewMap
        else:
            raise Exception('wrong img type')
        image = trans_func(image, angRes=kwargs['angRes'])

    image = image[:, :, np.newaxis] if image.ndim == 2 else image


    if normalize_fn is not None:
        return normalize_fn(image)
    else:
        return image


def get_img3d_fn(filename, path, normalize_fn):
    """
    Parames:
        mode - Depth : read 3-D image in format [depth=slices, height, width, channels=1]
               Channels : [height, width, channels=slices]
    """
    image = tifffile.imread(path + filename)  # [depth, height, width]
    # image = image[..., np.newaxis] # [depth, height, width, channels]
    if normalize_fn is not None:
        return normalize_fn(image)
    else:
        return image

def get_3d_stack(filename,path,normalize_fn):

    img = io.imread(os.path.join(path, filename))

    if (img.dtype != np.float32):
        img = img.astype(np.float32, casting='unsafe')
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if normalize_fn is not None:
        return normalize_fn(img)
    else:
        return img


def rearrange3d_fn(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """

    image = np.squeeze(image)  # remove channels dimension
    # print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth])
    for d in range(depth):
        image_re[:, :, d] = image[d, :, :]
    return image_re

def get_and_rearrange3d(filename, path, normalize_fn):
    image = get_img3d_fn(filename, path, normalize_fn=normalize_fn)
    return rearrange3d_fn(image)


def get_img2d_fn(filename, path, normalize_fn, **kwargs):
    image = tifffile.imread(os.path.join(path , filename)).astype(np.uint16)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    # print(image.shape)
    return normalize_fn(image, **kwargs)


def get_lf_extra(filename, path, n_num, normalize_fn, padding=False, **kwargs):
    image = get_img2d_fn(filename, path, normalize_fn, **kwargs)
    extra = lf_extract_fn(image, n_num=n_num, padding=padding)
    return extra


def normalize(x):
    max_ = np.max(x) * 1.1
    # max_ = 255.
    # max_ = np.max(x)
    x = x / (max_ / 2.)
    x = x - 1
    return x


def normalize_constant(im):
    assert im.dtype in [np.uint8, np.uint16]
    x = im.astype(np.float)
    max_ = 255. if im.dtype == np.uint8 else 65536.
    # x = x / (max_ / 2.) - 1.
    x = x / (max_)
    return x


def min_max(x, eps=1e-7):
    max_ = np.max(x)
    min_ = np.min(x)
    return (x - min_) / (max_ - min_ + eps)


def normalize_percentile(im, low=0, high=100, clip=False, is_random=False,**kwargs):
    if is_random:
        _p_low = np.random.uniform(0.1, 0.5)
        p_low = np.percentile(im, _p_low)

        _p_high = np.random.uniform(99.5, 99.9)
        p_high = np.percentile(im, _p_high)
    else:
        p_low = np.percentile(im, low)
        p_high = np.percentile(im, high)
    eps = 1e-7
    x = ((im - p_low) / (p_high - p_low + eps)).astype(np.float32)
    if clip:
        # x[x>1.0]=1.0
        x[x < .0] = .0
    if 'gamma' in kwargs:
        gamma = kwargs['gamma']
        x=np.power(x, gamma)
    # return x
    return x.astype(np.float32)



def normalize_percentile_z_score(im, low=0.2, high=99.8):
    p_low = np.percentile(im, low)
    p_high = np.percentile(im, high)
    eps = 1e-7
    x = np.clip(im, p_low, p_high)
    mean_ = np.mean(x)
    std = np.std(x)
    return (x - mean_) / std

def binary_normal(x):
    # max_ = np.max(x)
    max_ = 255.
    # max_ = np.max(x)
    x = x / max_

    return x


def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
    """
    Extract different views from a single LF projection

    Params:
        -lf2d - 2-D light field projection
        -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
        -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                     False : shrink size of extracted views to (lf2d.shape / Nnum);
    Returns:
        ndarray [height, width, channels=n_num^2] if mode is 'toChannel'
                or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
    """
    n = n_num
    h, w, c = lf2d.shape
    if padding:
        if mode == 'toDepth':
            lf_extra = np.zeros([n * n, h, w, c])  # [depth, h, w, c]

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, i: h: n, j: w: n, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        elif mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([h, w, n * n])

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[i: h: n, j: w: n, d] = lf2d[i: h: n, j: w: n]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
    else:
        new_h = int(np.ceil(h / n))
        new_w = int(np.ceil(w / n))

        if mode == 'toChannel':

            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([new_h, new_w, n * n])

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[:, :, d] = lf2d[i: h: n, j: w: n]
                    d += 1

        elif mode == 'toDepth':
            lf_extra = np.zeros([n * n, new_h, new_w, c])  # [depth, h, w, c]

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, :, :, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)

    return lf_extra


def do_nothing(x):
    return x

def _write3d(x, path, bitdepth=8, clip=True):
    """
    x : [depth, height, width, channels=1]
    """
    assert (bitdepth in [8, 16, 32])
    max_ =  np.max(x)
    if clip:
        x = np.clip(x, 0, max_)
    if bitdepth == 32:
        x = x.astype(np.float32)
    else:
        min_ = np.min(x)
        x = (x - min_) / (max_ - min_)
        if bitdepth == 8:
            x = x * 255
            x = x.astype(np.uint8)
        else:
            x = x * 65535
            x = x.astype(np.uint16)

    tifffile.imwrite(path, x)


def write3d(x, path, bitdepth=32):
    """
    save the volumetric image from x
    input: x : [batch, height, width, channels]
    """
    # dims = len(x.shape)
    if isinstance(path,list):
        index = 0
        # new_path + '_' + str(index) + '.' + fragments[-1], bitdepth
        # x = np.clip(x, a_min=0, a_max=1)
        for _f_name,_img in zip(path,x):
            recon_out = _img
            recon_out = np.transpose(np.transpose(recon_out, [1, 2, 0]), [1, 2, 0])
            save_name = _f_name
            _write3d(
                    recon_out,
                    save_name,
                    bitdepth=bitdepth
                    )
    else:
        recon_out = x[0]
        recon_out = np.transpose(np.transpose(recon_out, [1, 2, 0]), [1, 2, 0])
        _write3d(
            recon_out,
            path,
            bitdepth=bitdepth
        )

def add_delimiter(input_data, real_input):
    with open(input_data, 'r', encoding="utf-8") as fr:
        last_cursor= len(fr.readlines()) - 1
    with open(input_data, 'r', encoding="utf-8") as fr:
        with open(real_input, 'w', encoding="utf-8") as fw:
            for idx,line in enumerate(fr):
                if line == "}\n" and idx != last_cursor:
                    fw.writelines(line.strip("\n") + ',' + "\n")
                else:
                    fw.writelines(line)
    with open(real_input, 'r+', encoding="utf-8") as fs:
        content = fs.read()
        fs.seek(0)
        fs.write("[")
        fs.write(content)
        fs.seek(0, 2)
        fs.write("]")
