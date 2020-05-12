#!/usr/bin/env python3
# coding: utf-8

import os

import tensorflow as tf
from tqdm import tqdm

from utils.tfio import _load

FILES_TRAIN = './train.configs/train_aug_120x120.list.train'
FILES_VAL = './train.configs/train_aug_120x120.list.val'

PARAM_TRAIN = './train.configs/param_all_norm.pkl'
PARAM_VAL = './train.configs/param_all_norm_val.pkl'


def build_example(img_path, label):
    img_raw = open(img_path, 'rb').read()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/label': tf.train.Feature(float_list=tf.train.FloatList(value=label)),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    }))
    return example


files = open(FILES_VAL, 'r').read().splitlines()
gt = _load(PARAM_VAL)

print(len(files), files[0], gt.shape, next(zip(files, gt)))
writer = tf.io.TFRecordWriter('test_aug.tfrecord')

for filename, label in tqdm(zip(files, gt), total=len(files)):
    tf_example = build_example(os.path.join('./train_aug_120x120/', filename), label)
    writer.write(tf_example.SerializeToString())
writer.close()
