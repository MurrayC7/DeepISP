# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import rawpy
import cv2
import tensorflow as tf

tf.set_random_seed(819)
import tensorflow.contrib.slim as slim
import numpy as np
# import rawpy
import glob

input_dir = '../../datasets/raw/fivek_dataset/raw_photos/HQa1to5000/'
gt_dir = '../../datasets/raw/fivek_dataset/fivek_c/'
checkpoint_dir = './checkpoint/fivec/'
result_dir = './result_fivec/'

# get test IDs
test_fns = glob.glob(gt_dir + 'a48*.tif')
test_ids = [os.path.basename(test_fn)[0:-4] for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:-4]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


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
    # im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    black_level = raw.black_level_per_channel[0]
    im = np.maximum(im - black_level, 0) / (np.max(raw.raw_image) - black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

psnr = {}
ssim = {}
psnr_dcraw = {}
ssim_dcraw = {}
for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%s.dng' % test_id)
    # for k in range(len(in_files)):
    in_path = in_files[0]
    in_fn = os.path.basename(in_path)
    print(in_fn)
    gt_files = glob.glob(gt_dir + '%s.tif' % test_id)
    gt_path = gt_files[0]
    # gt_fn = os.path.basename(gt_path)
    # in_exposure = float(in_fn[9:-5])
    # gt_exposure = float(gt_fn[9:-5])
    # ratio = min(gt_exposure / in_exposure, 300)

    raw = rawpy.imread(in_path)
    input_full = np.expand_dims(pack_raw(raw), axis=0)

    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
    scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

    gt_tif = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[..., ::-1]
    gt_full = np.expand_dims(np.float32(gt_tif / 65535.0), axis=0)

    input_full = np.minimum(input_full, 1.0)

    output = sess.run(out_image, feed_dict={in_image: input_full})
    output = np.minimum(np.maximum(output, 0), 1) * 10

    output = output[0, :, :, :]
    gt_full = gt_full[0, :, :, :]
    scale_full = scale_full[0, :, :, :]
    scale_full = scale_full * np.mean(gt_full) / np.mean(
        scale_full)  # scale the low-light image to the same mean of the groundtruth

    gt_full = cv2.resize(gt_full, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_CUBIC)
    print(output.shape, scale_full.shape, gt_full.shape)
    output_im = tf.image.convert_image_dtype(output, tf.float32)
    scale_im = tf.image.convert_image_dtype(scale_full, tf.float32)
    gt_im = tf.image.convert_image_dtype(gt_full, tf.float32)
    psnr[test_id] = sess.run(
        tf.image.psnr(output_im, gt_im, max_val=1.0))
    ssim[test_id] = sess.run(
        tf.image.ssim(output_im, gt_im, max_val=1.0))
    psnr_dcraw[test_id] = sess.run(
        tf.image.psnr(scale_im, gt_im, max_val=1.0))
    ssim_dcraw[test_id] = sess.run(
        tf.image.ssim(scale_im, gt_im, max_val=1.0))
    print("psnr: ", psnr[test_id])
    print("ssim: ", ssim[test_id])
    print("psnr_dcraw: ", psnr_dcraw[test_id])
    print("ssim_dcraw: ", ssim_dcraw[test_id])

    scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
        result_dir + 'final/%5s_out.png' % test_id)
    scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        result_dir + 'final/%5s_scale.png' % test_id)
    scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        result_dir + 'final/%s_gt.png' % test_id)

if len(psnr) > 0 and len(ssim) > 0:
    print("average psnr:", sum(psnr.values()) / len(psnr))
    print("average ssim:", sum(ssim.values()) / len(ssim))
if len(psnr_dcraw) > 0 and len(ssim_dcraw) > 0:
    print("average psnr_dcraw:", sum(psnr_dcraw.values()) / len(psnr_dcraw))
    print("average ssim_dcraw:", sum(ssim_dcraw.values()) / len(ssim_dcraw))
