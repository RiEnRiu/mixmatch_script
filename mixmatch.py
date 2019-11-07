# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MixMatch training.
- Ensure class consistency by producing a group of `nu` augmentations of the same image and guessing the label for the
  group.
- Sharpen the target distribution.
- Use the sharpened distribution directly as a smooth label in MixUp.
"""
import sys
import os
mixmatch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'mixmatch')
if mixmatch_dir not in sys.path:
    sys.path.append(mixmatch_dir)


import functools

from absl import app
from absl import flags
from easydict import EasyDict
from libml import layers, utils, models
# from libml.data_pair import DATASETS
from libml.layers import MixMode
import tensorflow as tf

from libml.data import *
from libml.data_pair import *

import json

FLAGS = flags.FLAGS


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
        print(loss_l2u)
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
    assert FLAGS.nu == 2
    # print(DATASETS)
    dataset_json_path = './tfrecord/{0}-class.json'.format(FLAGS.dataset)
    with open(dataset_json_path,'r') as f:
        label_dict = json.load(f)

    dataset = DataSet_2.creator(FLAGS.dataset, 0, 0, 1, [augment_cifar10, stack_augment(augment_cifar10)], colors=3, nclass=len(label_dict), height=64, width=64) ()

    log_width = utils.ilog2(dataset.width)
    model = MixMatch(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,

        beta=FLAGS.beta,
        w_match=FLAGS.w_match,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    # model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)
    # model.train(FLAGS.train_kimg+1, FLAGS.train_kimg+1)
    print(FLAGS.train_kimg, FLAGS.report_kimg)
    model.train(FLAGS.train_kimg, FLAGS.report_kimg)



if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 128)
    FLAGS.set_default('lr', 0.0002)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)


