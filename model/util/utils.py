import tensorflow as tf
import numpy as np
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.layers import Layer
from config import config

w_init = tf.glorot_uniform_initializer  #Xavier uniform initializer.

if config.net_setting.is_bias:
    b_init = tf.constant_initializer(value=0.0)
else:
    b_init = None  #for previous unetzh
g_init = tf.random_normal_initializer(1., 0.02)


def conv2d(layer, n_filter, filter_size=3, stride=1, act=tf.identity, W_init=w_init,padding='SAME', b_init=b_init, name = 'conv2d'):
    return tl.layers.Conv2d(layer, n_filter=int(n_filter), filter_size=(filter_size, filter_size), strides=(stride, stride), act=act, padding=padding, W_init=W_init, b_init=b_init, name=name)


def concat(layer, concat_dim=-1, name='concat'):
    return ConcatLayer(layer, concat_dim=concat_dim, name=name)

def merge(layers, name='merge'):
    '''
    merge two Layers by element-wise addition
    Params : 
        -layers : list of Layer instances to be merged : [layer1, layer2, ...]
    '''
    return tl.layers.ElementwiseLayer(layers, combine_fn=tf.add, name=name)

