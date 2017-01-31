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

from ae import get_xdats, Xnn
import tensorflow as tf

def init():
    parser = argparse.ArgumentParser()
    # inout
    parser.add_argument("--ckpt_file", default='out/sample/mnist/ae/dummy2/model/c-5999')
    parser.add_argument("--out_dir", default='out/sample/mnist/recon/dummy2')
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
    xdats.val = get_xdats(args, args.test_label_file, args.test_image_dir, label_groups, nrows=10)
    print('Reading %s ...' % args.train_label_file)
    print('Reading images from %s ...' % args.train_image_dir)
    xdats.trn = get_xdats(args, args.train_label_file, args.train_image_dir, label_groups, nrows=60, shuffle=True)

    # out
    out_dir = os.path.join(args.out_dir, 'recon')
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)
        print('Made dir: %s' % out_dir)
    print('Writing %s' % out_dir)

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

        def to_image(feat):
            feat = feat.reshape((28, 28))
            img = Image.fromarray(feat)
            return img
        def do_recon(xdats, out_dir2):
            reconss = x.do_recon(sess, xdats, eval_batch_size)
            for i, label_group in enumerate(label_groups):
                for j, feat in enumerate(xdats[i].feats):
                    recon = reconss[i][j]

                    feat = feat.reshape((28, 28)) * 255
                    recon = recon.reshape((28, 28)) * 255
                    con = np.concatenate((feat, recon), axis=1)
                    img = Image.fromarray(con)
                    img = img.convert('L')
                    out_file = '%d_%d.png' % (i, j)
                    out_file = os.path.join(out_dir2, out_file)
                    #print('Writing %s' % out_file)
                    img.save(out_file)
        print('Reconstruct Validation Data ...')
        out_dir2 = os.path.join(out_dir, 'val')
        os.makedirs(out_dir2)
        print('Writing %s' % out_dir2)
        do_recon(xdats.val, out_dir2)

        print('Reconstruct Training Data ...')
        out_dir2 = os.path.join(out_dir, 'trn')
        os.makedirs(out_dir2)
        print('Writing %s' % out_dir2)
        do_recon(xdats.trn, out_dir2)

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
