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
    parser.add_argument("--out_dir", default='out/sample/mnist/siam/dummy2')
    parser.add_argument("--train_image_dir", default='dat/mnist_image/image/train')
    parser.add_argument("--test_image_dir", default='dat/mnist_image/image/test')
    parser.add_argument("--train_label_file", default='dat/mnist_image/label/train.txt')
    parser.add_argument("--test_label_file", default='dat/mnist_image/label/test.txt')
    parser.add_argument("--seed", type=int, default=1)
    # tf
    parser.add_argument("--keep_prob", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--reg_coeff", type=float, default=0.0001)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_steps", type=int, default=100)
    #parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--out_steps", type=int, default=10000)
    parser.add_argument("--decay_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    #parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--margin", type=float, default=1)
    args = parser.parse_args()

    if os.path.exists(args.out_dir) is True:
        raise Exception('Existed %s' % args.out_dir)
    else:
        os.makedirs(args.out_dir)
        print('Made dir: %s' % args.out_dir)
    return args

def get_confmat(u_label_num, labels, preds):
    cf = np.zeros((u_label_num, u_label_num), dtype=np.int)
    for label, pred in zip(labels, preds):
        cf[pred][label] += 1
    '''
    feat_num = len(feats)
    for j in range(feat_num):
        real_index = labels[j]
        indexes = np.argsort(feats[j])[::-1]
        max_index = indexes[0]
        cf[max_index][real_index] += 1
    '''
    return cf

def get_acc(cf):
    return 1.0 * np.sum(cf.diagonal()) / np.sum(cf)

def print_tensor(t):
    print('%s: %s' % (t.op.name, t.get_shape().as_list()))

class Xnn(object):
    """siamese encoder"""
    def __init__(self, label_names, margin=1.0, data_shape=[100]):
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
        self.data = data = tf.placeholder(tf.float32, shape=[batch_size] +  data_shape)
        print_tensor(data)
        self.data_p = data_p = tf.placeholder(tf.float32, shape=[batch_size] +  data_shape)
        print_tensor(data_p)
        self.labels = labels = tf.placeholder(tf.int32, shape=(batch_size))

        parameters = []
        with tf.variable_scope('fc1'):
            data_dim = data.get_shape().as_list()[1]
            data_dim_p = data_p.get_shape().as_list()[1]
            w = self.new_weight([data_dim, 100], reg_coeff=reg_coeff)
            b = self.new_bias([100], 0, reg_coeff=reg_coeff)
            parameters += [w, b]

            fc = tf.matmul(data, w)
            bias = fc + b
            fc1 = tf.nn.relu(bias)
            fc1 = tf.nn.dropout(fc1, keep_prob)
            print_tensor(fc1)

            fc_p = tf.matmul(data_p, w)
            bias_p = fc_p + b
            fc1_p = tf.nn.relu(bias_p)
            fc1_p = tf.nn.dropout(fc1_p, keep_prob)
            print_tensor(fc1_p)

        with tf.variable_scope('feat'):
            fc1_dim = fc1.get_shape().as_list()[1]
            fc1_dim_p = fc1_p.get_shape().as_list()[1]
            w = self.new_weight([fc1_dim, 2], reg_coeff=reg_coeff)
            b = self.new_bias([2], 0, reg_coeff=reg_coeff)
            parameters += [w, b]

            fc = tf.matmul(fc1, w)
            bias = fc + b
            feat = tf.nn.dropout(bias, keep_prob)
            print_tensor(feat)

            fc_p = tf.matmul(fc1_p, w)
            bias_p = fc_p + b
            feat_p = tf.nn.dropout(bias_p, keep_prob)
            print_tensor(feat_p)

        # contrastive loss (label = 1 if same else 0)
        margin = 1.0
        labels = tf.cast(labels, tf.float32)
        self.feat = feat
        self.feat_p = feat_p
        d = tf.reduce_sum(tf.square(feat - feat_p), 1)
        d_sqrt = tf.sqrt(d)
        _c_loss = (1.0 - labels) * tf.square(tf.maximum(0., margin - d_sqrt)) + labels * d
        c_loss = 0.5 * tf.reduce_mean(_c_loss)

        # loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = loss = c_loss + sum(reg_losses)

        # train
        #optimizer = tf.train.GradientDescentOptimizer(lr)
        optimizer = tf.train.MomentumOptimizer(lr, momentum)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train = optimizer.minimize(loss, global_step=global_step)
        #self.train = optimizer.minimize(loss) # do not know why global_step is needed

        '''
        # evaluation
        self.trn_loss = tf.placeholder(tf.float32)
        self.val_loss = tf.placeholder(tf.float32)
        '''

        # summary
        tf.scalar_summary('lr', lr)
        tf.scalar_summary('loss', loss)
        '''
        tf.scalar_summary('trn_loss', self.trn_loss)
        tf.scalar_summary('val_loss', self.val_loss)
        '''

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

        xdat_num = len(xdats)
        perm = range(xdat_num)
        diff_num = batch_size / 2
        same_num = batch_size - diff_num
        feats = []
        feats_p = []
        old_labels = []
        old_labels_p = []
        labels = []

        # same
        quo_num = same_num // xdat_num
        rem_num = same_num % xdat_num
        each_nums = [quo_num] * xdat_num
        random.shuffle(perm)
        for i in range(rem_num):
            each_nums[perm[i]] += 1
        for xdat, each_num in zip(xdats, each_nums):
            a_feats, a_labels, _ = xdat.next_batch(each_num * 2)
            for j in range(each_num):
                feats.append(a_feats[j * 2])
                feats_p.append(a_feats[j * 2 + 1])
                old_labels.append(a_labels[j * 2])
                old_labels_p.append(a_labels[j * 2 + 1])
                labels.append(1)

        # diff
        for i in range(diff_num):
            random.shuffle(perm)
            a_feats, a_labels, _ = xdats[perm[0]].next_batch(1)
            a_feats_p, a_labels_p, _ = xdats[perm[1]].next_batch(1)
            feats.append(a_feats[0])
            feats_p.append(a_feats_p[0])
            old_labels.append(a_labels[0])
            old_labels_p.append(a_labels_p[0])
            labels.append(0)

        ret[self.data] = np.array(feats)
        ret[self.data_p] = np.array(feats_p)
        ret[self.labels] = np.array(labels)
        return ret

    def do_train(self, sess, feed_dict):
        _, loss = sess.run([self.train, self.loss], feed_dict=feed_dict)
        return loss # loss before train

    '''
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
        correct_sum = 0.0
        for i in xrange(batch_num): 
            feed_dict = self.get_one_feed_dict(xdat, batch_sizes[i])
            a_loss, correct_num = sess.run([self.loss, self.correct_num], feed_dict=feed_dict)
            loss_sum += a_loss * batch_sizes[i]
            correct_sum += correct_num
        loss = 0.0
        accuracy = 0.0
        if data_num > 0:
            loss = loss_sum / data_num
            accuracy = correct_sum / data_num
        return loss, accuracy, correct_sum, data_num

    def do_eval(self, sess, xdats, batch_size=None):
        assert len(xdats) == len(self.label_names)
        m_loss = []
        m_acc = []
        m_correct_sum = []
        m_data_num = []
        for xdat in xdats:
            loss, acc, correct_sum, data_num = self.do_one_eval(sess, xdat, batch_size)
            m_loss.append(loss)
            m_acc.append(acc)
            m_correct_sum.append(correct_sum)
            m_data_num.append(data_num)
        loss = np.mean(m_loss)
        #acc = np.mean(m_acc)
        correct_sum = np.sum(m_correct_sum)
        data_num = np.sum(m_data_num)
        acc = 1.0 * correct_sum / data_num # total accuracy
        print('  loss = %.2e, correct = %0.04f (%d / %d)' % (loss, acc, correct_sum, data_num))
        return loss, acc

    def do_one_eval2(self, sess, xdat, batch_size=None):
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

        probs = []
        for i in xrange(batch_num): 
            feed_dict = self.get_one_feed_dict(xdat, batch_sizes[i])
            softmax = sess.run(self.softmax, feed_dict=feed_dict)
            probs += list(softmax)
        return probs

    def do_eval2(self, sess, xdats, batch_size=None):
        assert len(xdats) == len(self.label_names)
        labels = []
        preds = []
        for xdat in xdats:
            probs = self.do_one_eval2(sess, xdat, batch_size)
            labels += list(xdat.labels)
            preds += [np.argmax(prob) for prob in probs]

        print('Confusion matrix')
        sys.stdout.flush()
        cf = get_confmat(self.cls_num, labels, preds)
        for i, label_name in enumerate(self.label_names):
            print('%d, %s' % (i, label_name))
        print(cf)
        print('acc: %.4f' % get_acc(cf))
        sys.stdout.flush()
    '''

    def do_one_feat(self, sess, xdat, batch_size=None):
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
        correct_sum = 0.0
        feats = []
        for i in xrange(batch_num): 
            feed_dict = self.get_one_feed_dict(xdat, batch_sizes[i])
            a_feats = sess.run(self.feat, feed_dict=feed_dict)
            feats += list(a_feats)
        return feats

    def do_feat(self, sess, xdats, batch_size=None):
        assert len(xdats) == len(self.label_names)
        m_feats = []
        for xdat in xdats:
            feats = self.do_one_feat(sess, xdat, batch_size)
            m_feats.append(feats)
        return m_feats


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
        feat = np.asarray(img).flatten() / 255.0
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

def train_tf(args):
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
    feat_dim = len(xdats.trn[0].feats[0])
    with tf.Graph().as_default():
        print('Init network ...')
        sys.stdout.flush()
        x = Xnn(label_names=label_groups, margin=args.margin, data_shape=[feat_dim])

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
        '''
        eval_batch_size = args.eval_batch_size

        print('Training Data Eval:')
        trn_loss, trn_acc = x.do_eval(sess, xdats.trn, eval_batch_size)
        print('Validation Data Eval:')
        val_loss, val_acc = x.do_eval(sess, xdats.val, eval_batch_size)
        sys.stdout.flush()
        '''

        for step in xrange(args.max_steps):
            feed_dict = x.get_feed_dict(xdats.trn, batch_size, lr, momentum, reg_coeff)
            '''
            feed_dict[x.trn_loss] = trn_loss
            feed_dict[x.trn_acc] = trn_acc
            feed_dict[x.val_loss] = val_loss
            feed_dict[x.val_acc] = val_acc
            '''

            feed_dict[x.keep_prob] = args.keep_prob
            start_time = time.time()
            loss = x.do_train(sess, feed_dict)
            duration = time.time() - start_time
            feed_dict[x.keep_prob] = 1.0

            '''
            if (step + 1) % args.eval_steps == 0:
                print('Training Data Eval:')
                trn_loss, trn_acc = x.do_eval(sess, xdats.trn, eval_batch_size)
                feed_dict[x.trn_loss] = trn_loss
                feed_dict[x.trn_acc] = trn_acc
                print('Validation Data Eval:')
                val_loss, val_acc = x.do_eval(sess, xdats.val, eval_batch_size)
                feed_dict[x.val_loss] = val_loss
                feed_dict[x.val_acc] = val_acc
            '''

            if (step + 1) % args.log_steps == 0:
                print('Step %d: lr = %.2e, loss = %.2e (%.3f sec)' % (step, lr, loss, duration))
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
        '''
        print('Training Data Eval:')
        x.do_eval2(sess, xdats.trn, eval_batch_size)
        print('Validation Data Eval:')
        x.do_eval2(sess, xdats.val, eval_batch_size)
        '''

def run(args):
    # print args
    for key in dir(args):
        if key[0] == '_':
            continue
        print('%s: %s' % (key, getattr(args, key)))
    train_tf(args)

if  __name__ == "__main__":
    args = init()
    #pdb.set_trace()
    run(args)
