import numpy as np
import imageio
import PIL.Image as pilimg
import tensorlayer as tl
import os

__all__ = [
    'get_img2d_fn',
    'get_2d_imgs',
    'normalize_percentile',
    '_raise',

]

def _raise(e):
    raise e

def get_2d_imgs(filename, path, normalize_fn, **kwargs):
    image = imageio.imread(os.path.join(path, filename)).astype(np.uint16)
    image = image[:, :, np.newaxis] if image.ndim == 2 else image
    return normalize_fn(image)


def get_img2d_fn(filename, path, normalize_fn, **kwargs):
    image = np.asarray(imageio.imread(path + filename).astype(np.uint16))
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    # print(image.shape)
    return normalize_fn(image, **kwargs)


def normalize_percentile(im, low=0, high=100, clip=True, is_random=False):
    if is_random:
        _p_low = np.random.uniform(0.1, 0.5)
        p_low = np.percentile(im, _p_low)

        _p_high = np.random.uniform(99.5, 99.9)
        p_high = np.percentile(im, _p_high)
    else:
        p_low = np.percentile(im, low)
        p_high = np.percentile(im, high)
    eps = 1e-7
    x = (im - p_low) / (p_high - p_low + eps)
    if clip:
        # x[x>1.0]=1.0
        x[x < .0] = .0
    # print('%.2f-%.2f' %  (np.min(x), np.max(x)))
    return x
