# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io, imageio
import rawpy
import PIL
import cv2
import tensorflow as tf

tf.set_random_seed(819)
import tensorflow.contrib.slim as slim
import numpy as np
np.random.seed(819)
# import rawpy
import glob

from loss import *

input_dir = '../../datasets/SID/Sony/short/'
gt_dir = '../../datasets/SID/Sony/gt/'  # use rgb gt instead of raw
checkpoint_dir = './checkpoint/Sony/'
result_dir = './result_Sony/'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# get train IDs
train_fns = glob.glob(gt_dir + '0*.png')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = 512  # patch size for training
lmd = 0.5  # l1 and perceptual loss weight
save_freq = 500

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


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


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config,
                  graph=tf.get_default_graph())

in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)
# 测试的时候才看metric，训练的时候没有意义
# psnr = tf.reduce_mean(tf.image.psnr(out_image, gt_image, max_val=1.0), axis=0)
# ssim = tf.reduce_mean(tf.image.ssim(out_image, gt_image, max_val=1.0), axis=0)

# tf.summary.image('psnr', psnr)
# tf.summary.image('ssim', ssim)

# G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
G_l1loss = tf.reduce_mean(compute_l1_loss(out_image, gt_image))
features = ["conv1_2", "conv2_2", "conv3_2"]
G_perceploss = tf.reduce_mean(compute_percep_loss(gt_image, out_image, features, withl1=False))
G_loss = lmd * G_l1loss + (1 - lmd) * G_perceploss
tf.summary.scalar('l1loss', G_l1loss)
tf.summary.scalar('perceploss', G_perceploss)
tf.summary.scalar('sum_loss', G_loss)

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)
tf.summary.scalar('lr', lr)

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

merged = tf.summary.merge_all()
log_dir = result_dir + 'logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
train_writer = tf.summary.FileWriter(log_dir, sess.graph)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Raw data takes long time to load. Keep them in memory after loaded.
# gt_images = [None] * 6000
input_images = {}
input_images['300'] = [None] * len(train_ids)
input_images['250'] = [None] * len(train_ids)
input_images['100'] = [None] * len(train_ids)
# input_images['300'] = None
# input_images['250'] = None
# input_images['100'] = None

g_loss = np.zeros((5000, 1))

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
cnt = 0
for epoch in range(lastepoch, 4001):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    # cnt = 0
    if epoch > 2000:
        learning_rate = 1e-5
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.png' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            # gt_raw = rawpy.imread(gt_path)
            # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)
            # raw = rawpy.imread(in_path)
            # input_images[str(ratio)[0:3]] = np.expand_dims(pack_raw(raw), axis=0) * ratio
            #
            # gt_image_rgb = np.expand_dims(
            #     np.float32(np.array(PIL.Image.open(gt_path)) / 65535.), axis=0)
            # TODO: 目前gt rgb图有问题，导致loss收敛但结果有问题，
            #  修改需要读取16bit png 用cv2尝试
        gt_image_rgb = np.expand_dims(
            np.float32(cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[..., ::-1] / 65535.), axis=0)


        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        # gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]
        gt_patch = gt_image_rgb[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))
        #     gt_patch = np.transpose(gt_patch, (0, 2, 1))

        input_patch = np.minimum(input_patch, 1.0)
        # print('**debug shape: ', sess.run(tf.shape(input_patch)),
        #       sess.run(tf.shape(gt_patch)),
        #       sess.run(tf.shape(out_image)))

        summary, _, G_current, output = sess.run([merged, G_opt, G_loss, out_image],
                                                 feed_dict={in_image: input_patch, gt_image: gt_patch,
                                                            lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[ind] = G_current
        # if cnt % 20 == 0:
        train_writer.add_summary(summary, cnt)
        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            # PIL.Image.fromarray((temp * 255).astype('uint8')).convert('RGB').save(
            #     result_dir + '%04d/%05d_00_train_%d_pil.jpg' % (epoch, train_id, ratio))
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

    saver.save(sess, checkpoint_dir + 'model.ckpt')
