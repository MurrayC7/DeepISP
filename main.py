import argparse
import os
import tensorflow as tf

tf.set_random_seed(819)
from .model import Unet

parser = argparse.ArgumentParser(description='deepisp')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='fivek', help='path of the dataset')

args = parser.parse_args()


def main():
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig, graph=tf.get_default_graph()) as sess:
        model = Unet(sess, args)
        model.train(args) if args.phase == 'train' else model.test(args)


if __name__ == '__main__':
    tf.app.run()
