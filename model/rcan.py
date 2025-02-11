import tensorflow as tf
from tensorlayer.layers import Layer, InputLayer, ElementwiseLayer,SubpixelConv2d
from .util.utils import conv2d, concat
__all__ = ['RCAN']

n_resGroup=4
n_resblock=4

class Mean_CA(Layer):
    def __init__(self, layer=None, name='Mean_CA_3D'):
        Layer.__init__(self, name=name)
        print("  [Self] Mean_CA %s: size:%s fn:%s" % (self.name, layer.outputs.get_shape(), tf.reduce_mean.__name__))
        self.inputs = layer.outputs
        with tf.variable_scope(name):
            x = tf.reduce_mean(self.inputs, axis=(1, 2), keepdims=True)
            h = self.inputs.shape[1]
            w = self.inputs.shape[2]
            self.outputs = tf.tile(x, [1,h, w, 1])
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])

class Max_CA(Layer):
    def __init__(self, layer=None, name='Max_CA_3D'):
        Layer.__init__(self, name=name)
        print("  [Self] Max_CA %s: size:%s fn:%s" % (self.name, layer.outputs.get_shape(), tf.reduce_max.__name__))
        self.inputs = layer.outputs
        with tf.variable_scope(name):
            x = tf.reduce_max(self.inputs, axis=(1, 2), keepdims=True)
            h = self.inputs.shape[1]
            w = self.inputs.shape[2]
            self.outputs = tf.tile(x, [1, h, w, 1])
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
class Mean_SA(Layer):
    def __init__(self, layer=None, name='Mean_SA_3D'):
        Layer.__init__(self, name=name)
        print("  [Self] Mean_SA %s: size:%s fn:%s" % (
            self.name, layer.outputs.get_shape(), tf.reduce_mean.__name__))
        self.inputs = layer.outputs
        with tf.variable_scope(name):
            x = tf.reduce_mean(self.inputs, axis=3, keepdims=True)
            self.outputs = x
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])

class Max_SA(Layer):
    def __init__(self, layer=None, name='Max_SA_pool_3D'):
        Layer.__init__(self, name=name)
        print("  [Self] Max_SA %s: size:%s fn:%s" % (
            self.name, layer.outputs.get_shape(), tf.reduce_max.__name__))
        self.inputs = layer.outputs
        with tf.variable_scope(name):
            x = tf.reduce_max(self.inputs, axis=3, keepdims=True)
            self.outputs = x
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
##########################################################################
## Channel Attention Layer
def CALayer(input, n_filter=64,name='fca'):
    with tf.variable_scope(name):
        W2 = Mean_SA(input, name='CALayer_Mean_SA')
        W3 = Max_SA(input, name='CALayer_Max_SA')
        W4 = concat([W2, W3],concat_dim=-1,name='concat_sa')
        W5 = conv2d(W4, n_filter, filter_size=1, stride=1, act=tf.nn.sigmoid, padding='SAME', name='CALayer_conv2d_2')

        W6 = Mean_CA(input, name='CALayer_mean_ca')
        W7 = conv2d(W6, n_filter // 16, filter_size=1, stride=1, act=tf.nn.relu, padding='SAME', name='CALayer_conv2d_4')
        W8 = conv2d(W7, n_filter, filter_size=1, stride=1, act=tf.nn.sigmoid, padding='SAME', name='CALayer_conv2d_5')


        W9 = concat([W5, W8], concat_dim=-1, name='concat_sa_and_ca')
        W10 = conv2d(W9, n_filter, filter_size=1, stride=1, act=tf.nn.sigmoid, padding='SAME', name='CALayer_conv2d_6')
        mul = ElementwiseLayer([W10, input], combine_fn=tf.multiply, name='CALayer_multiply_out')
        return mul


##########################################################################
## Channel  Attention Block (CAB)
def CAB(input, n_filter=64, name='fca'):
    with tf.variable_scope(name):
        conv1 = conv2d(input, n_filter, filter_size=3, stride=1, act=tf.nn.relu, padding='SAME', name='CAB_conv2d_1')
        conv2 = conv2d(conv1, n_filter, filter_size=3, stride=1, padding='SAME', name='CAB_conv2d_2')
        att = CALayer(conv2, n_filter, name='CAB_att_1')
        output = ElementwiseLayer([att, input], combine_fn=tf.add, name='CAB_add_out')
        # output1 = tf.add(att.outputs, input.outputs)
        # output2 = InputLayer(output1, name='CAB_add')
        return output


def ResidualGroup(input, G=64, name='ResidualGroup'):
    # G0 = input.outputs.shape[-1]
    # if G0 != G:
    #     raise Exception('G0(%d) and G(%d) must be equal in RDB' % (G0, G))
    with tf.variable_scope(name):
        conv = input
        for i in range(n_resblock):
            conv = CAB(conv, n_filter=G, name='CAB_%d' % i)
        conv = conv2d(conv, n_filter=G, filter_size=3, stride=1, padding='SAME', name='ResidualGroup_conv2d_%d' % i)
        output = ElementwiseLayer([conv, input], combine_fn=tf.add, name='ResidualGroup_add_out')
        # output1 = tf.add(conv.outputs, input.outputs)
        # output2 = InputLayer(output1, name='ResidualGroup_add')
        return output


def RCAN(lr, sr_factor=1, format_out=True, reuse=False, name='rcan'):
    assert sr_factor in [1, 2, 3, 4]
    with tf.variable_scope(name, reuse=reuse):
        inputs = InputLayer(lr, name='RCAN_input') if not isinstance(lr, Layer) else lr
        conv = conv2d(inputs, n_filter=64, filter_size=3,  stride=1, padding='SAME', name='RCAN_conv2d_1')
        for j in range(n_resGroup):
            conv = ResidualGroup(conv, 64, name='ResidualGroup_%d'%j)

        if format_out:
            if sr_factor == 4:
                conv = conv2d(conv, n_filter=64, filter_size=3, stride=1, padding='SAME', name='RCAN_conv2d_2')
                n8 = SubpixelConv2d(conv, scale=2, name='SubpixelConv2d1')
                n8 = conv2d(n8, n_filter=64, filter_size=3, stride=1, padding='SAME', name='RCAN_conv2d_3')
                n8 = conv2d(n8, n_filter=64, filter_size=3, stride=1, padding='SAME', name='RCAN_conv2d_4')
                n8 = SubpixelConv2d(n8, scale=2, name='SubpixelConv2d2')
                n8 = conv2d(n8, n_filter=64, filter_size=3, stride=1, padding='SAME', name='RCAN_conv2d_5')
                n8 = conv2d(n8, n_filter=64, filter_size=3, stride=1, padding='SAME', name='RCAN_conv2d_6')
            elif sr_factor == 3:
                n8 = conv2d(conv, n_filter=27, filter_size=3, name='conv3')
                n8 = SubpixelConv2d(n8, scale=3, name='SubpixelConv2d1')
            elif sr_factor == 2:
                # n8 = conv2d(n7, n_filter=8, filter_size=3, name='conv3')
                n8 = SubpixelConv2d(conv, scale=2, name='SubpixelConv2d1')
            else:
                n8 = conv
            out = conv2d(n8, n_filter=1, filter_size=3, act=tf.identity, name='out')
        else:
            out = conv
        return out

