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

import tensorflow as tf

def init():
    parser = argparse.ArgumentParser()
    # inout
    parser.add_argument("--out_dir", default='out/sample/mnist/conv_ae/dummy2')
    parser.add_argument("--train_image_dir", default='dat/mnist_image/image/train')
    parser.add_argument("--test_image_dir", default='dat/mnist_image/image/test')
    parser.add_argument("--train_label_file", default='dat/mnist_image/label/train.txt')
    parser.add_argument("--test_label_file", default='dat/mnist_image/label/test.txt')
    parser.add_argument("--seed", type=int, default=1)
    # tf
    parser.add_argument("--keep_prob", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--reg_coeff", type=float, default=0.0001)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--out_steps", type=int, default=6000)
    parser.add_argument("--decay_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    args = parser.parse_args()

    if os.path.exists(args.out_dir) is True:
        raise Exception('Existed %s' % args.out_dir)
    else:
        os.makedirs(args.out_dir)
        print('Made dir: %s' % args.out_dir)
    return args

def print_tensor(t):
    print('%s: %s' % (t.op.name, t.get_shape().as_list()))

class Xnn(object):
    """a classifier"""
    def __init__(self, label_names, data_shape=[100]):
        self.label_names = label_names
        self.cls_num = len(label_names)
        self.data_shape = data_shape
        self.build_graph()

    def new_weight(self, shape, stddev=0.01, reg_coeff=0.0):
        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
        ret = tf.get_variable('w', shape, initializer=initializer, regularizer=tf.contrib.layers.l2_regularizer(reg_coeff))
        return ret

    def new_bias(self, shape, value=0, reg_coeff=0.0):
        initializer = tf.constant_initializer(value=value, dtype=tf.float32)
        ret = tf.get_variable('b', shape, initializer=initializer, regularizer=tf.contrib.layers.l2_regularizer(reg_coeff))
        return ret

    def build_graph(self):
        cls_num = self.cls_num
        data_shape = self.data_shape
        batch_size = None

        self.lr = lr = tf.placeholder(tf.float32)
        self.momentum = momentum = tf.placeholder(tf.float32)
        self.reg_coeff = reg_coeff = tf.placeholder(tf.float32)
        self.keep_prob = keep_prob = tf.placeholder(tf.float32)
        self.data = data = tf.placeholder(tf.float32, shape=[batch_size] + data_shape)
        print_tensor(data)
        self.labels = labels = tf.placeholder(tf.int32, shape=(batch_size))

        parameters = []
        with tf.variable_scope('conv1'):
            w = self.new_weight([3, 3, 1, 32], reg_coeff=reg_coeff)
            s = [1, 2, 2, 1]
            b = self.new_bias([32], 0, reg_coeff=reg_coeff)
            conv = tf.nn.conv2d(data, w, strides=s, padding='SAME')
            bias = conv + b
            conv1 = tf.nn.sigmoid(bias)
            conv1 = tf.nn.dropout(conv1, keep_prob)
            parameters += [w, b]
            print_tensor(conv1)
        with tf.variable_scope('deconv1'):
            w = self.new_weight([3, 3, 1, 32], reg_coeff=reg_coeff)
            s = [1, 2, 2, 1]
            b = self.new_bias([1], 0, reg_coeff=reg_coeff)
            dyn_batch_size = tf.shape(data)[0]
            recon_shape = tf.pack([dyn_batch_size, data_shape[0], data_shape[1], data_shape[2]])
            conv = tf.nn.conv2d_transpose(conv1, w, output_shape=recon_shape, strides=s, padding='SAME')
            bias = conv + b
            deconv1 = bias
            parameters += [w, b]
            print_tensor(deconv1)

        #labels = tf.to_int64(labels)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(deconv1, data, name='ce')
        self.ce_loss = ce_loss = tf.reduce_mean(ce, name='ce_loss')

        self.recon = recon = tf.nn.sigmoid(deconv1)
        se = tf.pow(recon - data, 2)
        self.se_loss = se_loss = tf.reduce_mean(se, name='se_loss')

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = loss = ce_loss + sum(reg_losses)

        # train
        optimizer = tf.train.MomentumOptimizer(lr, momentum)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train = optimizer.minimize(loss, global_step=global_step) # do not know why global_step is needed

        # evaluation
        self.trn_loss = tf.placeholder(tf.float32)
        self.trn_se_loss = tf.placeholder(tf.float32)
        self.val_loss = tf.placeholder(tf.float32)
        self.val_se_loss = tf.placeholder(tf.float32)

        # summary
        tf.scalar_summary('lr', lr)
        tf.scalar_summary('loss', loss)
        tf.scalar_summary('se_loss', se_loss)
        tf.scalar_summary('trn_loss', self.trn_loss)
        tf.scalar_summary('trn_se_loss', self.trn_se_loss)
        tf.scalar_summary('val_loss', self.val_loss)
        tf.scalar_summary('val_se_loss', self.val_se_loss)

    def get_one_feed_dict(self, xdat, batch_size, lr=0.0, momentum=0.0, reg_coeff=0.0, keep_prob=1.0):
        data, labels, paths = xdat.next_batch(batch_size)
        feed_dict = {
            self.data: data,
            self.labels: labels,
            self.lr: lr,
            self.momentum: momentum,
            self.reg_coeff: reg_coeff,
            self.keep_prob: keep_prob,
        }
        return feed_dict

    def get_feed_dict(self, xdats, batch_size, lr=0.0, momentum=0.0, reg_coeff=0.0, keep_prob=1.0):
        assert len(xdats) == len(self.label_names)
        ret = {}
        ret[self.lr] = lr
        ret[self.momentum] = momentum
        ret[self.reg_coeff] = reg_coeff
        ret[self.keep_prob] = keep_prob

        # each batch_size
        xdat_num = len(xdats)
        data_nums = [xdat.num for xdat in xdats]
        nonzero_num = sum([val > 0 for val in data_nums])
        assert batch_size >= nonzero_num
        quo_num = batch_size // nonzero_num
        rem_num = batch_size % nonzero_num
        a_batch_sizes = [quo_num] * nonzero_num
        for i in range(rem_num):
            a_batch_sizes[i] += 1
        for i, data_num in enumerate(data_nums):
            if 0 == data_num:
                a_batch_sizes.insert(i, 0)

        feats = []
        labels = []
        for xdat, a_batch_size in zip(xdats, a_batch_sizes):
            a_feats, a_labels, a_paths = xdat.next_batch(a_batch_size)
            feats += list(a_feats)
            labels += list(a_labels)
        ret[self.data] = np.array(feats)
        ret[self.labels] = np.array(labels)
        return ret

    def do_train(self, sess, feed_dict):
        _, loss, se_loss = sess.run([self.train, self.loss, self.se_loss], feed_dict=feed_dict)
        return loss, se_loss # loss before train

    def do_one_eval(self, sess, xdat, batch_size=None):
        xdat = xdat.copy() # prevent shuffling
        data_num = xdat.num
        if batch_size is None:
            batch_size = data_num

        quo_num = data_num // batch_size
        rem_num = data_num % batch_size
        batch_sizes = [batch_size] * quo_num
        if rem_num > 0:
            batch_sizes.append(rem_num)
        batch_num = len(batch_sizes)

        loss_sum = 0.0
        se_loss_sum = 0.0
        for i in xrange(batch_num): 
            feed_dict = self.get_one_feed_dict(xdat, batch_sizes[i])
            a_loss, a_se_loss = sess.run([self.loss, self.se_loss], feed_dict=feed_dict)
            loss_sum += a_loss * batch_sizes[i]
            se_loss_sum += a_se_loss * batch_sizes[i]
        loss = 0.0
        se_loss = 0.0
        if data_num > 0:
            loss = loss_sum / data_num
            se_loss = se_loss_sum / data_num
        return loss, se_loss, data_num

    def do_eval(self, sess, xdats, batch_size=None):
        assert len(xdats) == len(self.label_names)
        m_loss = []
        m_se_loss = []
        m_data_num = []
        for xdat in xdats:
            loss, se_loss, data_num = self.do_one_eval(sess, xdat, batch_size)
            m_loss.append(loss)
            m_se_loss.append(se_loss)
            m_data_num.append(data_num)
        loss = np.mean(m_loss)
        se_loss = np.mean(m_se_loss)
        data_num = np.sum(m_data_num)
        print('  loss = %.2e, se_loss = %.2e (# = %d)' % (loss, se_loss, data_num))
        return loss, se_loss

    def do_one_recon(self, sess, xdat, batch_size=None):
        xdat = xdat.copy() # prevent shuffling
        data_num = xdat.num
        if batch_size is None:
            batch_size = data_num

        quo_num = data_num // batch_size
        rem_num = data_num % batch_size
        batch_sizes = [batch_size] * quo_num
        if rem_num > 0:
            batch_sizes.append(rem_num)
        batch_num = len(batch_sizes)

        recons = []
        for i in xrange(batch_num): 
            feed_dict = self.get_one_feed_dict(xdat, batch_sizes[i])
            recon = sess.run(self.recon, feed_dict=feed_dict)
            recons += list(recon)
        return recons

    def do_recon(self, sess, xdats, batch_size=None):
        assert len(xdats) == len(self.label_names)
        reconss = []
        for i, xdat in enumerate(xdats):
            recons = self.do_one_recon(sess, xdat, batch_size)
            reconss.append(recons)
        return reconss

class Xdat(object):
    def __init__(self, feats, labels, paths, shuffle=False):
        assert len(feats) == len(labels)
        self.feats = np.array(feats)
        self.labels = np.array(labels)
        self.paths = np.array(paths)
        self.shuffle = shuffle
        self.num = len(labels)
        self.epoch = 0
        self.index = 0
        self.perm = np.arange(self.num)

    def copy(self):
        return Xdat(self.feats, self.labels, self.paths, self.shuffle)

    def next_batch(self, batch_size):
        """Return the next `batch_size` samples."""
        perm = []
        if 0 == self.num:
            return self.feats[perm], self.labels[perm], self.paths[perm]
        while len(perm) < batch_size:
            start = self.index
            self.index += batch_size
            end = self.index
            perm += list(self.perm)[start:end]
            if self.index >= self.num:
                self.epoch += 1
                self.index = 0
                if self.shuffle is True:
                    random.shuffle(self.perm)
        perm = perm[0:batch_size]
        return self.feats[perm], self.labels[perm], self.paths[perm]

def get_xdats(args, label_file, image_dir, label_groups, nrows=None, shuffle=False):
    # read image_file and label
    df = pd.read_csv(label_file, header=None, delim_whitespace=True, nrows=nrows)
    image_files = list(df[df.columns[0]])
    image_files = [os.path.join(image_dir, val) for val in image_files]
    labels = list(df[df.columns[1]])
    print('#data: %d' % len(labels))
    sys.stdout.flush()

    # extract feature
    feats = []
    for i, image_file in enumerate(image_files):
        img = Image.open(image_file)
        feat = np.asarray(img).reshape((28, 28, 1)) / 255.0
        feats.append(feat)
        #if 0 == i:
            #break

    # convert label to group index
    print('Select only data whose label exists in a label_group')
    label_group_num = len(label_groups)
    g_feats = []
    g_labels = []
    g_names = []
    for feat, label, name in zip(feats, labels, image_files):
        for i, label_group in enumerate(label_groups):
            if label in label_group:
                g_feats.append(feat)
                g_labels.append(i)
                g_names.append(name)
                break
    print('#data: %d' % len(g_labels))

    # divide to groups
    featss = [[] for i in range(label_group_num)]
    labelss = [[] for i in range(label_group_num)]
    namess = [[] for i in range(label_group_num)]
    for feat, label, name in zip(g_feats, g_labels, g_names):
        i = label
        featss[i].append(feat)
        labelss[i].append(label)
        namess[i].append(name)

    # convert to Xdat
    xdats = []
    for i in range(label_group_num):
        xdat = Xdat(featss[i], labelss[i], namess[i], shuffle)
        xdats.append(xdat)
        print('  %d: %d: %s' % (i, xdat.num, label_groups[i]))
    return xdats

def train(args):
    if args.seed is not None:
        random.seed(args.seed)

    # data
    print('Reading data sets ...')
    label_groups = [[val] for val in range(10)]
    class Xdats(object):
        pass
    xdats = Xdats()
    print('Reading %s ...' % args.test_label_file)
    print('Reading images from %s ...' % args.test_image_dir)
    xdats.val = get_xdats(args, args.test_label_file, args.test_image_dir, label_groups, nrows=1000)
    print('Reading %s ...' % args.train_label_file)
    print('Reading images from %s ...' % args.train_image_dir)
    xdats.trn = get_xdats(args, args.train_label_file, args.train_image_dir, label_groups, nrows=6000, shuffle=True)

    # out
    out_dir = os.path.join(args.out_dir, 'model')
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)
        print('Made dir: %s' % out_dir)
    print('Writing %s' % out_dir)

    # train
    print('Training ...')
    data_shape = list(xdats.trn[0].feats[0].shape)
    with tf.Graph().as_default():
        print('Init network ...')
        sys.stdout.flush()
        x = Xnn(label_names=label_groups, data_shape=data_shape)

        config = tf.ConfigProto(
            device_count = {'GPU': 1},
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3), 
        )
        sess = tf.Session(config=config)
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        summary = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(out_dir, sess.graph)

        lr = args.lr
        momentum = args.momentum
        reg_coeff = args.reg_coeff
        batch_size = args.batch_size
        eval_batch_size = args.eval_batch_size

        print('Training Data Eval:')
        trn_loss, trn_se_loss = x.do_eval(sess, xdats.trn, eval_batch_size)
        print('Validation Data Eval:')
        val_loss, val_se_loss = x.do_eval(sess, xdats.val, eval_batch_size)
        sys.stdout.flush()

        for step in xrange(args.max_steps):
            feed_dict = x.get_feed_dict(xdats.trn, batch_size, lr, momentum, reg_coeff)
            feed_dict[x.trn_loss] = trn_loss
            feed_dict[x.trn_se_loss] = trn_se_loss
            feed_dict[x.val_loss] = val_loss
            feed_dict[x.val_se_loss] = val_se_loss

            feed_dict[x.keep_prob] = args.keep_prob
            start_time = time.time()
            loss, se_loss = x.do_train(sess, feed_dict)
            duration = time.time() - start_time
            feed_dict[x.keep_prob] = 1.0

            if (step + 1) % args.eval_steps == 0:
                print('Training Data Eval:')
                trn_loss, trn_se_loss = x.do_eval(sess, xdats.trn, eval_batch_size)
                feed_dict[x.trn_loss] = trn_loss
                feed_dict[x.trn_se_loss] = trn_se_loss
                print('Validation Data Eval:')
                val_loss, val_se_loss = x.do_eval(sess, xdats.val, eval_batch_size)
                feed_dict[x.val_loss] = val_loss
                feed_dict[x.val_se_loss] = val_se_loss

            if (step + 1) % args.log_steps == 0:
                print('Step %d: lr = %.2e, loss = %.2e, se_loss = %.2e (%.3f sec)' % (step, lr, loss, se_loss, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % args.out_steps == 0 or (step + 1) == args.max_steps:
                out_file_prefix = '%s/c' % out_dir
                out_file = saver.save(sess, out_file_prefix, global_step=step)
                print('Wrote %s' % out_file)
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if args.decay_steps > 0 and (step + 1) % args.decay_steps == 0:
                if args.lr_decay < 1.0 and args.lr_decay > 0.0:
                    lr *= args.lr_decay
            sys.stdout.flush()

        # eval
        print('Training Data Eval:')
        x.do_eval(sess, xdats.trn, eval_batch_size)
        print('Validation Data Eval:')
        x.do_eval(sess, xdats.val, eval_batch_size)

def run(args):
    # print args
    for key in dir(args):
        if key[0] == '_':
            continue
        print('%s: %s' % (key, getattr(args, key)))
    train(args)

if  __name__ == "__main__":
    args = init()
    #pdb.set_trace()
    run(args)
