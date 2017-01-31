# coding: utf-8
import pdb
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import random
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from siam import get_xdats, Xnn
import tensorflow as tf

def init():
    parser = argparse.ArgumentParser()
    # inout
    parser.add_argument("--ckpt_file", default='out/sample/mnist/siam/dummy/model/c-9999')
    parser.add_argument("--out_dir", default='out/sample/mnist/siam_plot/dummy')
    parser.add_argument("--train_image_dir", default='dat/mnist_image/image/train')
    parser.add_argument("--test_image_dir", default='dat/mnist_image/image/test')
    parser.add_argument("--train_label_file", default='dat/mnist_image/label/train.txt')
    parser.add_argument("--test_label_file", default='dat/mnist_image/label/test.txt')
    parser.add_argument("--seed", type=int, default=1)
    # tf
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    args = parser.parse_args()

    if os.path.exists(args.out_dir) is True:
        raise Exception('Existed %s' % args.out_dir)
    else:
        os.makedirs(args.out_dir)
        print('Made dir: %s' % args.out_dir)
    return args

def test(args):
    # data
    print('Reading data sets ...')
    label_groups = [[val] for val in range(10)]
    class Xdats(object):
        pass
    xdats = Xdats()
    print('Reading %s ...' % args.test_label_file)
    print('Reading images from %s ...' % args.test_image_dir)
    xdats.val = get_xdats(args, args.test_label_file, args.test_image_dir, label_groups, nrows=100)
    print('Reading %s ...' % args.train_label_file)
    print('Reading images from %s ...' % args.train_image_dir)
    xdats.trn = get_xdats(args, args.train_label_file, args.train_image_dir, label_groups, nrows=600, shuffle=True)

    # out
    out_dir = args.out_dir
    '''
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)
        print('Made dir: %s' % out_dir)
    print('Writing %s' % out_dir)
    '''

    # test
    print('Testing ...')
    feat_dim = len(xdats.trn[0].feats[0])
    with tf.Graph().as_default():
        print('Init network ...')
        sys.stdout.flush()
        x = Xnn(label_names=label_groups, data_shape=[feat_dim])

        config = tf.ConfigProto(
            device_count = {'GPU': 1},
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3), 
        )
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        saver.restore(sess, args.ckpt_file)

        eval_batch_size = args.eval_batch_size


        def plot_feat(xdats, out_file, font_size=14, figure_height=8, figure_width=6, dpi=60):
            featss = x.do_feat(sess, xdats, eval_batch_size)

            fig = plt.figure()
            fig.set_figwidth(figure_width)
            fig.set_figheight(figure_height)
            ax = fig.add_subplot(111)
            for i, label_group in enumerate(label_groups):
                feats = featss[i]
                us = [feat[0] for feat in feats]
                vs = [feat[1] for feat in feats]
                ax.plot(us, vs, '.', alpha=1.0)
                for u, v in zip(us, vs):
                    ax.text(u, v, i)
            #print('Writing %s' % out_file)
            #img.save(out_file)
            plt.rcParams.update({'font.size': font_size})
            plt.tight_layout()
            plt.grid()
            #plt.xticks(rotation=90)
            plt.savefig(out_file, dpi=dpi, bbox_inches='tight', transparent=True)
            plt.close()
        print('Reconstruct Validation Data ...')
        out_file = os.path.join(out_dir, 'val.png')
        print('Writing %s' % out_file)
        plot_feat(xdats.val, out_file)

        print('Reconstruct Training Data ...')
        out_file = os.path.join(out_dir, 'trn.png')
        print('Writing %s' % out_file)
        plot_feat(xdats.trn, out_file)

def run(args):
    # print args
    for key in dir(args):
        if key[0] == '_':
            continue
        print('%s: %s' % (key, getattr(args, key)))
    test(args)

if  __name__ == "__main__":
    args = init()
    #pdb.set_trace()
    run(args)
