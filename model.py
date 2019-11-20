from __future__ import division
import os
import scipy.misc as sm
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from ops import *


class UNet3D():

    def __init__(self, height=256, weight=512, batch_size=16, max_disp=192):
        self.reg = 1e-4  # TODO
        self.max_disp = max_disp  # TODO
        # self.image_size_tf = None
        self.height = height
        self.weight = weight
        self.batch_size = batch_size
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # self.lr = 0.001

    def construct_model(self, input):
        # self.left = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        # self.right = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        # self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight])
        # self.image_size_tf = tf.shape(self.x)[1:3]

        x = tf.split(input, 4, axis=3)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        conv4_1 = self.CNN(x1)
        conv4_2 = self.CNN(x2, True)
        conv4_3 = self.CNN(x3, True)
        conv4_4 = self.CNN(x4, True)
        f1 = self.SPP(conv4_1)
        f2 = self.SPP(conv4_2, True)
        f3 = self.SPP(conv4_3, True)
        f4 = self.SPP(conv4_4, True)

        cost_vol = self.cost_vol(f1, f2, f3, f4, self.max_disp)
        output = self.CNN3D(cost_vol, type="sm_hourglass")

        # output = self.output(outputs)  # size of (B, H, W),3out
        return output
        # self.y = disps[2]
        # print(self.disps.shape)

    def CNN(self, bottom, reuse=False):
        with tf.variable_scope('CNN'):
            with tf.variable_scope('conv0'):
                bottom = conv_block(tf.layers.conv2d, bottom, 32, 3, strides=1, name='conv0_1', reuse=reuse,
                                    reg=self.reg)
                for i in range(1, 3):
                    bottom = conv_block(tf.layers.conv2d, bottom, 32, 3, name='conv0_%d' % (i + 1), reuse=reuse,
                                        reg=self.reg)
            with tf.variable_scope('conv1'):
                for i in range(3):
                    bottom = res_block(tf.layers.conv2d, bottom, 32, 3, name='conv1_%d' % (i + 1), reuse=reuse,
                                       reg=self.reg)
            with tf.variable_scope('conv2'):
                bottom = res_block(tf.layers.conv2d, bottom, 64, 3, strides=1, name='conv2_1', reuse=reuse,
                                   reg=self.reg,
                                   projection=True)
                for i in range(1, 16):
                    bottom = res_block(tf.layers.conv2d, bottom, 64, 3, name='conv2_%d' % (i + 1), reuse=reuse,
                                       reg=self.reg)
            with tf.variable_scope('conv3'):
                bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=2, name='conv3_1', reuse=reuse,
                                   reg=self.reg, projection=True)
                for i in range(1, 3):
                    bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=2, name='conv3_%d' % (i + 1),
                                       reuse=reuse,
                                       reg=self.reg)
            with tf.variable_scope('conv4'):
                for i in range(3):
                    bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=4, name='conv4_%d' % (i + 1),
                                       reuse=reuse,
                                       reg=self.reg)
        return bottom

    def SPP(self, bottom, reuse=False):
        with tf.variable_scope('SPP'):
            branches = []
            for i, p in enumerate([64, 32, 16, 8]):
                branches.append(SPP_branch(tf.layers.conv2d, bottom, p, 32, 3, name='branch_%d' % (i + 1), reuse=reuse,
                                           reg=self.reg))
            # if not reuse:
            conv2_16 = tf.get_default_graph().get_tensor_by_name('CNN/conv2/conv2_16/add:0')
            conv4_3 = tf.get_default_graph().get_tensor_by_name('CNN/conv4/conv4_3/add:0')
            # else:
            #    conv2_16 = tf.get_default_graph().get_tensor_by_name('CNN_1/conv2/conv2_16/add:0')
            #    conv4_3 = tf.get_default_graph().get_tensor_by_name('CNN_1/conv4/conv4_3/add:0')
            concat = tf.concat([conv2_16, conv4_3] + branches, axis=-1, name='concat')
            with tf.variable_scope('fusion'):
                bottom = conv_block(tf.layers.conv2d, concat, 128, 3, name='conv1', reuse=reuse, reg=self.reg)
                fusion = conv_block(tf.layers.conv2d, bottom, 32, 1, name='conv2', reuse=reuse, reg=self.reg)
        return fusion

    def cost_vol(self, f1, f2, f3, f4, max_disp=192):
        with tf.variable_scope('cost_vol'):
            disparity_costs = []
            # shape = tf.shape(right) #(N,H,W,F)
            # cost = tf.concat([f1, f2, f3, f4], axis=3)
            for i in range(3):
                disparity_costs.append(f1)
                disparity_costs.append(f2)
                disparity_costs.append(f3)
                disparity_costs.append(f4)

            cost_vol = tf.stack(disparity_costs, axis=1)
        return cost_vol

    def CNN3D(self, bottom, type="basic"):
        with tf.variable_scope('CNN3D'):
            # for i in range(2):
            if type == "basic":
                bottom = conv_block(tf.layers.conv3d, bottom, 64, 3, name='3Dconv0_1', reg=self.reg)
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv0_2', reg=self.reg)

                _3Dconv1 = res_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv1', reg=self.reg)
                # _3Dconv1 = _3Dconv1 + bottom

                _3Dconv2 = res_block(tf.layers.conv3d, _3Dconv1, 32, 3, name='3Dconv2', reg=self.reg)
                # _3Dconv2 = _3Dconv2 + _3Dconv1

                _3Dconv3 = res_block(tf.layers.conv3d, _3Dconv2, 32, 3, name='3Dconv3', reg=self.reg)
                # _3Dconv3 = _3Dconv3 + _3Dconv2

                _3Dconv4 = res_block(tf.layers.conv3d, _3Dconv3, 32, 3, name='3Dconv4', reg=self.reg)
                # _3Dconv4 = _3Dconv4 + _3Dconv3

                output_1 = conv_block(tf.layers.conv3d, _3Dconv4, 32, 3, name='output_1_1', reg=self.reg)
                output_1 = conv_block(tf.layers.conv3d, output_1, 1, 3, name='output_1', reg=self.reg, apply_bn=False,
                                      apply_relu=False, use_bias=False)
                output_1 = tf.squeeze(output_1, axis=4)
                output_1 = tf.transpose(output_1, [0, 3, 2, 1])
                output = tf.depth_to_space(output_1, 2)
            elif type == "hourglass":
                bottom = conv_block(tf.layers.conv3d, bottom, 64, 3, name='3Dconv0_1', reg=self.reg)
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv0_2', reg=self.reg)

                _3Dconv1 = res_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv1', reg=self.reg)

                _3Dstack = [hourglass('3d', _3Dconv1, [64, 64, 64, 32], [3, 3, 3, 3], [None, None, -2, _3Dconv1],
                                      name='3Dstack1', reg=self.reg)]
                for i in range(1, 3):
                    _3Dstack.append(hourglass('3d', _3Dstack[-1][-1], [64, 64, 64, 32], [3, 3, 3, 3],
                                              [_3Dstack[-1][-2], None, _3Dstack[0][0], _3Dconv1],
                                              name='3Dstack%d' % (i + 1),
                                              reg=self.reg))
                output_1 = conv_block(tf.layers.conv3d, _3Dstack[0][3], 32, 3, name='output_1_1', reg=self.reg)
                output_1 = conv_block(tf.layers.conv3d, output_1, 1, 3, name='output_1', reg=self.reg, apply_bn=False,
                                      apply_relu=False, use_bias=False)
                output_1 = tf.squeeze(output_1, axis=4)
                output_1 = tf.transpose(output_1, [0, 3, 2, 1])
                output = tf.depth_to_space(output_1, 2)
            elif type=="sm_hourglass":
                bottom = conv_block(tf.layers.conv3d, bottom, 64, 3, name='3Dconv0_1', reg=self.reg)
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv0_2', reg=self.reg)

                _3Dconv1 = res_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv1', reg=self.reg)

                _3Dstack = [hourglass('3d', _3Dconv1, [64, 64, 64, 32], [3, 3, 3, 3], [None, None, -2, _3Dconv1],
                                      name='3Dstack1', reg=self.reg)]
                for i in range(1, 3):
                    _3Dstack.append(hourglass('3d', _3Dstack[-1][-1], [64, 64, 64, 32], [3, 3, 3, 3],
                                              [_3Dstack[-1][-2], None, _3Dstack[0][0], _3Dconv1],
                                              name='3Dstack%d' % (i + 1),
                                              reg=self.reg))
                output_1 = conv_block(tf.layers.conv3d, _3Dstack[0][3], 32, 3, name='output_1_1', reg=self.reg)
                output_1 = conv_block(tf.layers.conv3d, output_1, 4, 3, name='output_1', reg=self.reg, apply_bn=False,
                                      apply_relu=False, use_bias=False)
                # output_1 = tf.squeeze(output_1, axis=4)
                # output_1 = tf.transpose(output_1, [0, 3, 2, 1])
                # output = tf.depth_to_space(output_1, 2)
                # TODO: softmax hourglass
                weight_volume = tf.nn.softmax(output_1, axis=4)
                output = output_1 * weight_volume

        return output

import rawpy
import os
import numpy as np
for fileraw in os.listdir('./'):
    if fileraw.endswith('.CR3'):
            raw = rawpy.imread(fileraw)
            print(raw.color_desc, raw.raw_pattern, raw.black_level_per_channel)