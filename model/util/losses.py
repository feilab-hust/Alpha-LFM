import tensorflow as tf
import tensorlayer as tl

__all__ = ['mse_loss',
           'mae_loss',
           'edge_loss',
           ]
def mse_loss(image, reference):
    with tf.variable_scope('l2_loss'):
        return tl.cost.mean_squared_error(image, reference,is_mean=True)
def mae_loss(image, reference):
    with tf.variable_scope('l1_loss'):
        return tl.cost.absolute_difference_error(image,reference,is_mean=True)

def edge_loss(image, reference):

    '''
    params:
        -image : tensor of shape [batch, depth, height, width, channels], the output of DVSR
        -reference : same shape as the image
    '''

    with tf.variable_scope('edges_loss'):
        edges_sr = tf.image.sobel_edges(image)
        edges_hr = tf.image.sobel_edges(reference)
        return mse_loss(edges_sr, edges_hr)

