import numpy as np
import tensorflow as tf
from tf_octConv import *
from tf_cnn_basic import *
from oct_unet_unit import *

G = 1
alpha = 0.25
use_fp16 = True
k_sec = {2: 3, 3: 4, 4: 6, 5: 3}
'''
# Unet
def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out
'''


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))

    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1]
                                    )  # Ref to https://github.com/tensorflow/tensorflow/issues/20334

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def oct_unet(input, alpha=0.25):
    # data = tf.Variable(name="data")
    # data = tf.cast(x=data, dtype=np.float16) if use_fp16 else data

    # conv1
    conv1_hf, conv1_lf = firstOctConv_BN_AC(data=input, alpha=alpha, num_filter_in=32, num_filter_out=64,
                                            kernel=(3, 3), name='g_conv1_1', pad='same')
    conv1_hf, conv1_lf = octConv_BN_AC(hf_data=conv1_hf, lf_data=conv1_lf, alpha=alpha, num_filter_in=64,
                                       num_filter_out=64,
                                       kernel=(3, 3), name='g_conv1_2', pad='same')
    pool1_hf = Pooling(data=conv1_hf, pool_type="max", kernel=(2, 2), pad="same", name="pool1_hf")
    pool1_lf = Pooling(data=conv1_lf, pool_type="max", kernel=(2, 2), pad="same", name="pool1_lf")

    # conv2
    conv2_hf, conv2_lf = octConv_BN_AC(hf_data=pool1_hf, lf_data=pool1_lf, alpha=alpha, num_filter_in=64,
                                       num_filter_out=128,
                                       kernel=(3, 3),
                                       name='g_conv2_1', pad='same')
    conv2_hf, conv2_lf = octConv_BN_AC(hf_data=conv2_hf, lf_data=conv2_lf, alpha=alpha, num_filter_in=128,
                                       num_filter_out=128,
                                       kernel=(3, 3), name='g_conv2_2', pad='same')
    pool2_hf = Pooling(data=conv2_hf, pool_type="max", kernel=(2, 2), pad="same", name="pool2_hf")
    pool2_lf = Pooling(data=conv2_lf, pool_type="max", kernel=(2, 2), pad="same", name="pool2_lf")

    # conv3
    conv3_hf, conv3_lf = octConv_BN_AC(hf_data=pool2_hf, lf_data=pool2_lf, alpha=alpha, num_filter_in=128,
                                       num_filter_out=256,
                                       kernel=(3, 3),
                                       name='g_conv3_1', pad='same')
    conv3_hf, conv3_lf = octConv_BN_AC(hf_data=conv3_hf, lf_data=conv3_lf, alpha=alpha, num_filter_in=256,
                                       num_filter_out=256,
                                       kernel=(3, 3), name='g_conv3_2', pad='same')
    pool3_hf = Pooling(data=conv3_hf, pool_type="max", kernel=(2, 2), pad="same", name="pool3_hf")
    pool3_lf = Pooling(data=conv3_lf, pool_type="max", kernel=(2, 2), pad="same", name="pool3_lf")

    # conv4
    conv4_hf, conv4_lf = octConv_BN_AC(hf_data=pool3_hf, lf_data=pool3_lf, alpha=alpha, num_filter_in=256,
                                       num_filter_out=512,
                                       kernel=(3, 3),
                                       name='g_conv4_1', pad='same')
    conv4_hf, conv4_lf = octConv_BN_AC(hf_data=conv4_hf, lf_data=conv4_lf, alpha=alpha, num_filter_in=512,
                                       num_filter_out=512,
                                       kernel=(3, 3), name='g_conv4_2', pad='same')
    pool4_hf = Pooling(data=conv4_hf, pool_type="max", kernel=(2, 2), pad="same", name="pool4_hf")
    pool4_lf = Pooling(data=conv4_lf, pool_type="max", kernel=(2, 2), pad="same", name="pool4_lf")

    # conv5
    conv5_hf, conv5_lf = octConv_BN_AC(hf_data=pool4_hf, lf_data=pool4_lf, alpha=alpha, num_filter_in=512,
                                       num_filter_out=1024,
                                       kernel=(3, 3),
                                       name='g_conv5_1', pad='same')
    conv5_hf, conv5_lf = octConv_BN_AC(hf_data=conv5_hf, lf_data=conv5_lf, alpha=alpha, num_filter_in=1024,
                                       num_filter_out=1024,
                                       kernel=(3, 3), name='g_conv5_2', pad='same')

    up6_hf = upsample_and_concat(conv5_hf, conv4_hf, 384, 768)
    up6_lf = upsample_and_concat(conv5_lf, conv4_lf, 128, 256)
    # conv6
    conv6_hf, conv6_lf = octConv_BN_AC(hf_data=up6_hf, lf_data=up6_lf, alpha=alpha, num_filter_in=1024,
                                       num_filter_out=512,
                                       kernel=(3, 3),
                                       name='g_conv6_1', pad='same')
    conv6_hf, conv6_lf = octConv_BN_AC(hf_data=conv6_hf, lf_data=conv6_lf, alpha=alpha, num_filter_in=512,
                                       num_filter_out=512,
                                       kernel=(3, 3), name='g_conv6_2', pad='same')

    up7_hf = upsample_and_concat(conv6_hf, conv3_hf, 192, 384)
    up7_lf = upsample_and_concat(conv6_lf, conv3_lf, 64, 128)
    # conv7
    conv7_hf, conv7_lf = octConv_BN_AC(hf_data=up7_hf, lf_data=up7_lf, alpha=alpha, num_filter_in=512,
                                       num_filter_out=256,
                                       kernel=(3, 3),
                                       name='g_conv7_1', pad='same')
    conv7_hf, conv7_lf = octConv_BN_AC(hf_data=conv7_hf, lf_data=conv7_lf, alpha=alpha, num_filter_in=256,
                                       num_filter_out=256,
                                       kernel=(3, 3), name='g_conv7_2', pad='same')

    up8_hf = upsample_and_concat(conv7_hf, conv2_hf, 96, 192)
    up8_lf = upsample_and_concat(conv7_lf, conv2_lf, 32, 64)
    # conv8
    conv8_hf, conv8_lf = octConv_BN_AC(hf_data=up8_hf, lf_data=up8_lf, alpha=alpha, num_filter_in=256,
                                       num_filter_out=128,
                                       kernel=(3, 3),
                                       name='g_conv8_1', pad='same')
    conv8_hf, conv8_lf = octConv_BN_AC(hf_data=conv8_hf, lf_data=conv8_lf, alpha=alpha, num_filter_in=128,
                                       num_filter_out=128,
                                       kernel=(3, 3), name='g_conv8_2', pad='same')

    up9_hf = upsample_and_concat(conv8_hf, conv1_hf, 48, 96)
    up9_lf = upsample_and_concat(conv8_lf, conv1_lf, 16, 32)
    # conv9
    conv9_hf, conv9_lf = octConv_BN_AC(hf_data=up9_hf, lf_data=up9_lf, alpha=alpha, num_filter_in=128,
                                       num_filter_out=64,
                                       kernel=(3, 3),
                                       name='g_conv9_1', pad='same')
    conv9_hf, conv9_lf = octConv_BN_AC(hf_data=conv9_hf, lf_data=conv9_lf, alpha=alpha, num_filter_in=64,
                                       num_filter_out=64,
                                       kernel=(3, 3), name='g_conv9_2', pad='same')

    # conv10
    conv10 = lastOctConv_BN(hf_data=conv9_hf, lf_data=conv9_lf, alpha=alpha, num_filter_in=64,
                            num_filter_out=12,
                            kernel=(1, 1), stride=(2, 2),
                            name='g_conv10', pad='same')
    # filter = tf.Variable(tf.truncated_normal([2, 2, 12, 12], stddev=0.02))
    # conv10_up = tf.nn.conv2d_transpose(conv10, filter, [1, 12, 512, 512], strides=[1, 2, 2, 1])
    out = tf.depth_to_space(conv10, 2)
    return out


'''
    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
'''
