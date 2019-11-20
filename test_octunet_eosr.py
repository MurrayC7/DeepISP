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

np.random.seed(819)
# import rawpy
import glob

from octconv_unet import oct_unet

input_dir = '../../datasets/raw/eosr/val/'
gt_dir = '../../datasets/raw/eosr/val/'
checkpoint_dir = './checkpoint/eosr_oct/'
result_dir = './result_eosr_oct/'

# get test IDs
test_fns = glob.glob(gt_dir + '*.JPG')
test_ids = [os.path.basename(test_fn)[0:-4] for test_fn in test_fns]

ps = 1024
alpha = 0.25
DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:-4]


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

    cfa = raw.raw_pattern
    cfa_dict = {'RGGB': [[0, 1], [3, 2]], 'BGGR': [[2, 3], [1, 0]], 'GBRG': [[3, 2], [0, 1]]}
    if (cfa == cfa_dict['RGGB']).all():
        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :],
                              im[1:H:2, 1:W:2, :]), axis=2)
    elif (cfa == cfa_dict['BGGR']).all():
        out = np.concatenate((im[1:H:2, 1:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :],
                              im[0:H:2, 0:W:2, :]), axis=2)
    elif (cfa == cfa_dict['GBRG']).all():
        out = np.concatenate((im[1:H:2, 0:W:2, :],
                              im[0:H:2, 0:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[0:H:2, 1:W:2, :]), axis=2)
    else:
        raise ValueError('Unsupported CFA configuration: {}'.format(cfa))
    return out


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = oct_unet(in_image, alpha)

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
    gt_files = glob.glob(gt_dir + '%s.JPG' % test_id)
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
    gt_full = np.expand_dims(np.float32(gt_tif / 255.0), axis=0)

    input_full = np.minimum(input_full, 1.0)
    H = input_full.shape[1]
    W = input_full.shape[2]
    
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    input_full = input_full[:, yy:yy + ps, xx:xx + ps, :]
    gt_full = gt_full[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]
    scale_full = scale_full[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

    output = sess.run(out_image, feed_dict={in_image: input_full})
    output = np.minimum(np.maximum(output, 0), 1)

    output = output[0, :, :, :]
    gt_full = gt_full[0, :, :, :]
    scale_full = scale_full[0, :, :, :]
    scale_full = scale_full * np.mean(gt_full) / np.mean(
        scale_full)  # scale the low-light image to the same mean of the groundtruth

    gt_full = cv2.resize(gt_full, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_CUBIC)
    # print(output.shape, scale_full.shape, gt_full.shape)
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
