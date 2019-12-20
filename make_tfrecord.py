#-*-coding:utf-8-*-
import argparse

import cv2
import time
import os
import numpy as np
import glob
from shutil import copyfile
import tensorflow as tf
from tqdm import trange,tqdm
import sys
import json


def _encode_jpeg(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_jpeg(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord(b, label, filename):
    print('Saving dataset:', filename)
    time.sleep(1)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in trange(len(b), desc='Building records'):
            feat = dict(image=_bytes_feature(b[x]),
                        label=_int64_feature(label[x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)

def scan_images(data_p):
    labels = os.listdir(data_p)
    labels.pop(labels.index('UNLABEL'))
    if 'NEGATIVE' in labels:
        labels.pop(labels.index('NEGATIVE'))

    label_dict = {}
    label_cid_list = []
    label_full_path_list = []
    for cid, label in enumerate(labels):
        label_dict[cid] = label
        p1 = os.path.join(data_p, label)
        for img_full_name in os.listdir(p1):
            label_full_path_list.append(os.path.join(p1, img_full_name))
            label_cid_list.append(cid)

    dir_to_scan = set()
    dir_to_scan.add(os.path.join(data_p,'UNLABEL'))
    unlabel_full_path_list = []
    while len(dir_to_scan)!=0:
        p = dir_to_scan.pop()
        for x in os.listdir(p):
            pp = os.path.join(p,x)
            if os.path.isfile(pp):
                unlabel_full_path_list.append(pp)
            elif os.path.isdir(pp):
                dir_to_scan.add(pp)
    unlabel_cid_list = [len(label_dict)]*len(unlabel_full_path_list)

    return label_dict, label_cid_list, label_full_path_list, \
                       unlabel_cid_list, unlabel_full_path_list





def make_tfrecord(data_p):
    dataset_name = os.path.basename(os.path.abspath(data_p))
    out_p = './tfrecord'
    if os.path.isdir(out_p)==False:
        os.makedirs(out_p)
    label_tfrecord_name = '{0}-label.tfrecord'.format(dataset_name)
    unlabel_tfrecord_name = '{0}-unlabel.tfrecord'.format(dataset_name)
    class_json_name = '{0}-class.json'.format(dataset_name)

    # scan files
    label_dict, label_cid_list, label_full_path_list, unlabel_cid_list, unlabel_full_path_list = \
    scan_images(data_p)

    # write labels
    with open(os.path.join(out_p, class_json_name), 'w') as f:
        json.dump(label_dict,f,indent=4,sort_keys=True)

    # dataset-label.tfrecord
    print('transform labeled data...')
    label_tf_path = os.path.join(out_p, label_tfrecord_name)
    img_mat = np.empty((0,64,64,3),dtype=np.uint8)
    cid_list = []
    for i in trange(len(label_cid_list)):
    # for cid, img_full_path in tqdm(zip(label_cid_list, label_full_path_list)):
        cid, img_full_path = label_cid_list[i], label_full_path_list[i]
        img = cv2.imread(img_full_path)
        if img is None:
            continue
        img64 = cv2.resize(img, (64, 64), cv2.INTER_AREA).reshape(-1, 64, 64, 3)
        img_mat = np.concatenate((img_mat, img64), axis=0)
        cid_list.append(cid)
    write_tfrecord(_encode_jpeg(img_mat), np.array(cid_list), label_tf_path)

    #  dataset-unlabel.tfrecord
    print('transform unlabeled data...')
    unlabel_tf_path = os.path.join(out_p, unlabel_tfrecord_name)
    img_mat = np.empty((0,64,64,3),dtype=np.uint8)
    cid_list = []
    for i in trange(len(unlabel_cid_list)):
    # for cid, img_full_path in tqdm(zip(unlabel_cid_list, unlabel_full_path_list)):
        cid, img_full_path = unlabel_cid_list[i], unlabel_full_path_list[i]
        img = cv2.imread(img_full_path)
        if img is None:
            continue
        img64 = cv2.resize(img, (64, 64), cv2.INTER_AREA).reshape(-1, 64, 64, 3)
        img_mat = np.concatenate((img_mat, img64), axis=0)
        cid_list.append(cid)
    write_tfrecord(_encode_jpeg(img_mat), np.array(cid_list), unlabel_tf_path)

    # write one sample for valid and test
    one_sample_tf_path = os.path.join(out_p, '{0}-tmp.tfrecord'.format(dataset_name))
    write_tfrecord(_encode_jpeg(img_mat[0:1]), np.array(cid_list[0:1]), one_sample_tf_path)

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir', type = str,required=True, help = 'Data set path.')
    args = parser.parse_args()

    make_tfrecord(args.dir)
        


              




