# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import rawpy
import tensorflow as tf

tf.set_random_seed(819)
import tensorflow.contrib.slim as slim
import numpy as np
# import rawpy
import glob

from octconv_unet import oct_unet

input_dir = '../../datasets/SID/Sony/short/'
gt_dir = '../../datasets/SID/Sony/long/'
checkpoint_dir = './checkpoint/Sony_oct_ssim/'
result_dir = './result_Sony_oct_ssim/'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

ps = 512
alpha = 0.25
DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


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


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
# out_image = network(in_image)
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
for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

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

        output_im = tf.image.convert_image_dtype(output, tf.float32)
        gt_im = tf.image.convert_image_dtype(gt_full, tf.float32)
        psnr[str(test_id) + '_' + str(k)] = sess.run(
            tf.image.psnr(output_im, gt_im, max_val=1.0))
        ssim[str(test_id) + '_' + str(k)] = sess.run(
            tf.image.ssim(output_im, gt_im, max_val=1.0))
        print("psnr: ", psnr[str(test_id) + '_' + str(k)])
        print("ssim: ", ssim[str(test_id) + '_' + str(k)])

        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
        scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
        scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))

if len(psnr) > 0 and len(ssim) > 0:
    print("average psnr:", sum(psnr.values()) / len(psnr))
    print("average ssim:", sum(ssim.values()) / len(ssim))
