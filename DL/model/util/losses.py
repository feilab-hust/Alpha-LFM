import tensorflow as tf
import numpy as np
import tensorlayer as tl
from tensorflow.keras import backend as K


__all__ = ['mse_loss',
           'mae_loss',
           'HuberLoss',
           'edge_loss',
           'EPI_mse_loss',
           'SSIM_loss',
           'get_lpips_loss',
           'get_weighted_loss',
           'g_mse_loss',
           'Reprojection_loss',
           ]

AngRes=15

def Reprojection_loss(image, reference, **kwargs):

    if "projection_range" in kwargs:
        local_DOF= int(kwargs['projection_range'])
    else:
        local_DOF=30
    def generate_custom_square_wave(z_max, total_range=161, range_width=30):
        z = tf.range(total_range, dtype=tf.int32)
        z_max_int = tf.cast(z_max, tf.int32)
        condition = tf.logical_and(z >= (z_max_int - range_width), z <= (z_max_int + range_width))
        condition = tf.cast(condition,tf.float32)
        return condition
    with tf.variable_scope('Reprojection_loss'):
        wf_size = reference.get_shape().as_list()
        inten_list = tf.reduce_mean(image, axis=[1,2])
        Depth = image.get_shape().as_list()[-1]
        z_max = tf.to_float(tf.arg_max(inten_list,dimension=1))
        z_inten_list = generate_custom_square_wave(z_max=z_max,total_range=Depth,range_width=local_DOF)

        reSample3D = tf.image.resize_images(image, [wf_size[1], wf_size[2]])
        z_inten_list = z_inten_list[None,None,None,...]# re-sample
        reSample3D  = reSample3D*z_inten_list
        proj_img_max = tf.reduce_max(reSample3D,axis=3)
        proj_img_max = proj_img_max / tf.reduce_max(proj_img_max)
        proj_img_max = tf.expand_dims(proj_img_max, axis=-1)
        wf_loss_1 = tl.cost.mean_squared_error(proj_img_max, reference, is_mean=True)

        return 5 * wf_loss_1


def mse_loss(image, reference):
    with tf.variable_scope('l2_loss'):
        return tl.cost.mean_squared_error(image, reference,is_mean=True)
def mae_loss(image, reference):
    with tf.variable_scope('l1_loss'):
        return tl.cost.absolute_difference_error(image,reference,is_mean=True)

def HuberLoss(image,reference):
    loss = tf.reduce_mean(tf.losses.huber_loss(image, reference))
    return loss

def edge_loss(image, reference):
    with tf.variable_scope('edges_loss'):
        edges_sr = tf.image.sobel_edges(image)
        edges_hr = tf.image.sobel_edges(reference)
        return mse_loss(edges_sr, edges_hr)


def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                    ):
    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                             vals,
                             vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def SSIM_loss(image, reference,max_v=1.0,filter_size=5,filter_sigma=0.8):
    #return tf.reduce_mean(1-tf.image.ssim(image,reference,max_val=max_v))
    batch_size, H, W, z_depth = image.get_shape().as_list()
    loss_ssim=0
    for i in range(z_depth):
        # y1 =reference[..., i:i + 1]
        # y2 =image[..., i:i + 1]
        y1 = K.tile((reference[..., i:i + 1]), [1, 1, 1, 3])
        y2 = K.tile((image[..., i:i + 1]), [1, 1, 1, 3])
        temp = tf.reduce_mean(1-tf.image.ssim(y1,y2,max_val=max_v))
        loss_ssim = temp + loss_ssim
    loss_ssim = loss_ssim / z_depth
    return loss_ssim

def get_weighted_loss(image,reference,ang_res=15):
    _,h,w,_=image.get_shape().as_list()
    n_num=ang_res
    new_h = int(np.ceil(h / n_num))
    new_w = int(np.ceil(w / n_num))
    angRes = n_num
    out_loss= []
    for i in range(angRes):
        for j in range(angRes):
            temp_view = image[:, i * new_h:(i + 1) * new_h, j * new_w:(j + 1) * new_w, :]
            temp_ref  = reference[:, i * new_h:(i + 1) * new_h, j * new_w:(j + 1) * new_w, :]
            mse=tf.square(temp_view - temp_ref)
            mse=mse/tf.reduce_max(mse)
            out_loss.append((tf.reduce_mean(mse)))
    return tf.reduce_sum(out_loss)/255

def EPI_mse_loss(image,reference,**kwargs):

    def _SAI2ViewMap(input_tensor,angRes):
        _,h,w,_=input_tensor.get_shape().as_list()
        n_num=angRes
        new_h = int(np.ceil(h / n_num))
        new_w = int(np.ceil(w / n_num))
        angRes = n_num
        out = []
        for i in range(angRes):
            out_u = []
            for j in range(angRes):
                temp_view = input_tensor[:, i * new_h:(i + 1) * new_h, j * new_w:(j + 1) * new_w, :]
                out_u.append(temp_view)
            u_list = tf.concat(out_u, 3)
            u_stack = u_list[..., tf.newaxis]
            out.append(u_stack)
        out = tf.concat(out, 4)   #b,h,w,u,v
        return out
    def _gradient(pred):
        D_dx = pred[:, 1:, :, :, :] - pred[:, :-1, :, :, :]
        D_dy = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        D_dax = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        D_day = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        return D_dx, D_dy, D_dax, D_day

    ang_res=AngRes
    pred=_SAI2ViewMap(image,angRes=ang_res)
    label=_SAI2ViewMap(reference,angRes=ang_res)
    pred_dx, pred_dy, pred_dax, pred_day = _gradient(pred)
    label_dx, label_dy, label_dax, label_day = _gradient(label)
    return mse_loss(pred_dx, label_dx) + mse_loss(pred_dy, label_dy) + mse_loss(pred_dax,label_dax) + mse_loss(pred_day, label_day)

def get_lpips_loss(image, reference,is_scale=False):
    from .lpips_tensorflow import lp

    batch_size,H,W,z_depth = image.get_shape().as_list()
    if is_scale:
        pred = tf.image.resize_images(image, size=[128, 128],method=tf.image.ResizeMethod.BICUBIC, align_corners=False)
        reference = tf.image.resize_images(reference, size=[128, 128],method=tf.image.ResizeMethod.BICUBIC, align_corners=False)
    loss_lp = 0
    for i in range(z_depth):
        y1 = K.tile((reference[..., i:i + 1]), [1, 1, 1, 3])
        y2 = K.tile((image[..., i:i + 1]), [1, 1, 1, 3])
        temp = lp.lpips(y1, y2)
        loss_lp = temp + loss_lp
    loss_lp = loss_lp / z_depth
    return tf.reduce_mean(loss_lp,axis=0)

def g_mse_loss(image, reference):
    import math
    def normal_distribution(mean, sigma, z_depth, cut=True):
        x = np.linspace(0, z_depth, z_depth)
        eps = 1e-7
        y = (tf.exp(-1 * ((x - mean) ** 2) / ((2 * (sigma ** 2 + eps))) / (math.sqrt(2 * np.pi) * sigma + eps)) + eps)
        return tf.convert_to_tensor(y)
    ratio = []
    batch_size, H, W, z_depth = image.get_shape().as_list()
    for i in range(z_depth):
        ratio.append(tf.reduce_mean(reference[:, :, :, i]))
    ratio = tf.convert_to_tensor(ratio)
    ratio = tf.nn.sigmoid(ratio/tf.reduce_max(ratio))
    return tf.reduce_mean(ratio * (tf.squared_difference(image, reference)))


