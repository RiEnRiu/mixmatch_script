#-*-coding:utf-8-*-

import functools
import sys
import os
mixmatch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'mixmatch')
if mixmatch_dir not in sys.path:
    sys.path.append(mixmatch_dir)

from tqdm import tqdm
from absl import app
from absl import flags
from easydict import EasyDict
from libml import layers, utils, models
from libml.data import *
from libml.data_pair import *
from libml.layers import MixMode
import tensorflow as tf
import numpy as np
import cv2
import shutil
# from shutil import copyfile
from make_tfrecord import scan_images


FLAGS = flags.FLAGS
flags.DEFINE_string('dir', '', 'Data set path.')
flags.DEFINE_float('high_thresh', 0.7, '')
flags.DEFINE_float('low_thresh', 0.3, '')


class MixMatch(models.MultiModel):
    def augment(self, x, l, beta, **kwargs):
        assert 0, 'Do not call.'

    def guess_label(self, y, classifier, T, **kwargs):
        del kwargs
        logits_y = [classifier(yi, training=True) for yi in y]
        logits_y = tf.concat(logits_y, 0)
        # Compute predicted probability distribution py.
        p_model_y = tf.reshape(tf.nn.softmax(logits_y), [len(y), -1, self.nclass])
        p_model_y = tf.reduce_mean(p_model_y, axis=0)
        # Compute the target distribution.
        p_target = tf.pow(p_model_y, 1. / T)
        p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
        return EasyDict(p_target=p_target, p_model=p_model_y)

    def model(self, batch, lr, wd, ema, beta, w_match, warmup_kimg=1024, nu=2, mixmode='xxy.yxy', **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [None, nu] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
        augment = MixMode(mixmode)
        classifier = functools.partial(self.classifier, **kwargs)

        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        guess = self.guess_label(tf.split(y, nu), classifier, T=0.5, **kwargs)
        ly = tf.stop_gradient(guess.p_target)
        lx = tf.one_hot(l_in, self.nclass)
        xy, labels_xy = augment([x_in] + tf.split(y, nu), [lx] + [ly] * nu, [beta, beta])
        x, y = xy[0], xy[1:]
        labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
        del xy, labels_xy

        batches = layers.interleave([x] + y, batch)
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits = [classifier(batches[0], training=True)]
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        for batchi in batches[1:]:
            logits.append(classifier(batchi, training=True))
        logits = layers.interleave(logits, batch)
        logits_x = logits[0]
        logits_y = tf.concat(logits[1:], 0)

        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))
        loss_l2u = tf.reduce_mean(loss_l2u)
        tf.summary.scalar('losses/xe', loss_xe)
        tf.summary.scalar('losses/l2u', loss_l2u)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + w_match * loss_l2u, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(batches[0], training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        return EasyDict(
            x=x_in, y=y_in, label=l_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))




class DataSet_2:
    def __init__(self, name, train_labeled, train_unlabeled, test, valid, eval_labeled, eval_unlabeled,
                 height=32, width=32, colors=3, nclass=10, mean=0, std=1, p_labeled=None, p_unlabeled=None):
        self.name = name
        self.train_labeled = train_labeled
        self.train_unlabeled = train_unlabeled
        self.eval_labeled = eval_labeled
        self.eval_unlabeled = eval_unlabeled
        self.test = test
        self.valid = valid
        self.height = height
        self.width = width
        self.colors = colors
        self.nclass = nclass
        self.mean = mean
        self.std = std
        self.p_labeled = p_labeled
        self.p_unlabeled = p_unlabeled

    @classmethod
    def creator(cls, name, seed, label, valid, augment, parse_fn=default_parse, do_memoize=True, colors=3,
                nclass=10, height=32, width=32, name_suffix=''):
        if not isinstance(augment, list):
            augment = [augment] * 2
        # fullname = '.%d@%d' % (seed, label)
        # root = os.path.join(DATA_DIR, 'SSL', name + fullname)
        root = './tfrecord/{0}'.format(name)
        fn = memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)

        def create():
            p_labeled = p_unlabeled = None
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment

            if FLAGS.p_unlabeled:
                sequence = FLAGS.p_unlabeled.split(',')
                p_unlabeled = np.array(list(map(float, sequence)), dtype=np.float32)
                p_unlabeled /= np.max(p_unlabeled)

            train_labeled = parse_fn(dataset([root + '-label.tfrecord']))
            train_unlabeled = parse_fn(dataset([root + '-unlabel.tfrecord']).skip(valid))
            if FLAGS.whiten:
                mean, std = compute_mean_std(train_labeled.concatenate(train_unlabeled))
            else:
                mean, std = 0, 1

            return cls(name, # name + name_suffix + fullname + '-' + str(valid),
                       train_labeled=fn(train_labeled).map(augment[0], para),
                       train_unlabeled=fn(train_unlabeled).map(augment[1], para),
                       eval_labeled=parse_fn(dataset([root + '-label.tfrecord'])),
                       eval_unlabeled=parse_fn(dataset([root + '-unlabel.tfrecord']).skip(valid)),
                       valid=parse_fn(dataset([root + '-unlabel.tfrecord']).take(valid)),
                       # test=parse_fn(dataset([os.path.join(DATA_DIR, '%s-test.tfrecord' % name)])),
                       # valid=parse_fn(dataset([root + '-tmp.tfrecord'])),
                       test=parse_fn(dataset([root + '-unlabel.tfrecord']).take(valid)),
                       nclass=nclass, colors=colors, p_labeled=p_labeled, p_unlabeled=p_unlabeled,
                       height=height, width=width, mean=mean, std=std)

        return create # name + name_suffix + fullname + '-' + str(valid), create




def main(argv):
    del argv  # Unused.

    high_thresh = FLAGS.high_thresh
    low_thresh = FLAGS.low_thresh

    dataset_dir = FLAGS.dir
    dataset_name = os.path.basename(os.path.abspath(dataset_dir))
    unlabel_dir = os.path.join(FLAGS.dir,'UNLABEL')

    # label
    label_dict, _, _, _, unlabel_full_path_list = scan_images(dataset_dir)

    # dataset
    dataset = DataSet_2.creator(dataset_name, 0, 0, 1, [augment_cifar10, stack_augment(augment_cifar10)], colors=3, nclass=len(label_dict), height=64, width=64) ()

    # model
    model_dir = './experiments/{0}/MixMatch_archresnet_batch128_beta0.75_ema0.999_filters32_lr0.0002_nclass11_repeat4_scales4_w_match75.0_wd0.02/tf'
    model_dir = model_dir.format(dataset_name)
    with open(os.path.join(model_dir,'checkpoint'),'r') as fp:
        lines = fp.readlines()
        model_file_name = lines[0].split()[1][1:-1]
        print(model_file_name)
        model_path = os.path.join(model_dir, model_file_name)
    mm = MixMatch(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=0.0001,
        wd=0.02,
        arch="resnet",
        batch=128,
        nclass=dataset.nclass,
        ema=0.999,
        beta=0.75,
        w_match=75.0,
        scales=4,
        filters=32,
        repeat=4)
    mm.eval_mode(model_path)

    # test and move file
    for p in tqdm(unlabel_full_path_list):
        img = cv2.imread(p).astype(np.float64)
        img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        img = img * (2.0 / 255) - 1.0
        img = img.reshape(-1, 64, 64, 3)
        logits = mm.session.run(mm.ops.classify_op, feed_dict={mm.ops.x: img})
        a = np.argmax(logits)
        conf = logits[0,a]
        result = label_dict[a]
        if conf>=high_thresh:
            b = os.path.join(unlabel_dir, result)
        elif conf<=low_thresh:
            b = os.path.join(unlabel_dir, 'HARD', result)
        else:
            b = os.path.join(unlabel_dir, 'UNLABEL')
        if not os.path.isdir(b):
            os.makedirs(b)
        if not os.path.isfile(os.path.join(b, os.path.basename(p))):
            shutil.move(p,b)


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)

# Where images is of dimensions (batch, height, width, colors)
# The predicted classes would be np.argmax(logits)
# To compute distribution would require you to compute the softmax of the logits.
