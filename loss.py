import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import tensorflow as tf
import scipy.io

vgg_rawnet = scipy.io.loadmat('../../datasets/pretrain-features/imagenet-vgg-verydeep-19.mat')
print("Loaded vgg19 pretrained imagenet")


# 1xWxHx3
def learn_align(prediction, target, tar_w, tar_h):
    shift = tf.Variable(tf.random_normal([1, 2]), name="shift")
    translated_image = tf.contrib.image.translate(target,
                                                  shift,
                                                  interpolation='BILINEAR')
    cropped_image = tf.slice(translated_image, [0, 0, 0, 0], [1, tar_h, tar_w, 3])
    loss = tf.reduce_mean(tf.abs(cropped_image - prediction))
    return loss, cropped_image


def build_net(ntype, nin, nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def build_vgg19(input, features='conv1_2', reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net = {}
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = input - np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
    net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
    net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
    net['pool1'] = build_net('pool', net['conv1_2'])
    net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
    net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
    net['pool2'] = build_net('pool', net['conv2_2'])
    net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
    net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
    net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
    net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
    net['pool3'] = build_net('pool', net['conv3_4'])
    net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
    net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
    net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
    net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
    net['pool4'] = build_net('pool', net['conv4_4'])
    net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
    net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
    return net


def compute_percep_loss(input, output, features, withl1=False, reuse=False):
    loss_sum = 0
    vgg_real = build_vgg19(output * 255.0, features, reuse=reuse)
    vgg_fake = build_vgg19(input * 255.0, features, reuse=True)
    if withl1:
        loss_sum += compute_l1_loss(vgg_real['input'], vgg_fake['input'])
    if "conv1_2" in features:
        loss_sum += compute_l1_loss(vgg_real['conv1_2'], vgg_fake['conv1_2'])
    if "conv2_2" in features:
        loss_sum += compute_l1_loss(vgg_real['conv2_2'], vgg_fake['conv2_2'])
    if "conv3_2" in features:
        loss_sum += compute_l1_loss(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 3.7
    if "conv4_2" in features:
        loss_sum += compute_l1_loss(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 5.6
    if "conv5_2" in features:
        loss_sum += compute_l1_loss(vgg_real['conv5_2'], vgg_fake['conv5_2']) * 10 / 1.5
    return loss_sum / 255.


def compute_l1_loss(input, output):
    return tf.reduce_mean(tf.abs(input - output), [1, 2, 3], keepdims=True)


def compute_l2_loss(input, output):
    return tf.reduce_mean(tf.square(input - output), [1, 2, 3], keepdims=True)
