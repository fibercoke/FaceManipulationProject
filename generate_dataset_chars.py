import pickle
import time
import matplotlib.pyplot as plt
import tqdm
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
import os

flags.DEFINE_string('dataset_path', './data/new_my_chars_train_data/', 'path to data dir')
flags.DEFINE_string('dataset_output_path', './data/new_my_chars_train_data.tfrecord', 'path to output dataset file')
flags.DEFINE_string('classes_output_path', './data/new_my_chars_train_data.names', 'path to output class file')

def build_example(img_path, label):
    img_raw = open(img_path, 'rb').read()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    }))
    return example

def main(_argv):
    classes = list(filename for filename in os.listdir(FLAGS.dataset_path)
                   if os.path.isdir(os.path.join(FLAGS.dataset_path, filename)) and len(filename) == 1)
    with open(FLAGS.classes_output_path, 'w', encoding='utf8') as fp:
        fp.writelines('\n'.join(classes))
        pass
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes_output_path).read().splitlines())}

    lst_img = []
    lst_label = []
    logging.info("saved and loaded class map")
    writer = tf.io.TFRecordWriter(FLAGS.dataset_output_path)

    example_numbers = 0
    for cls in tqdm.tqdm(classes):
        label = class_map[cls]
        cls_path = os.path.join(FLAGS.dataset_path, cls)
        for filename in os.listdir(cls_path):
            if filename.endswith('.jpg'):
                tf_example = build_example(os.path.join(cls_path, filename), label)
                writer.write(tf_example.SerializeToString())
                example_numbers += 1

    writer.close()
    logging.info("Done")
    print('All exampe:', example_numbers)
    pass


if __name__ == '__main__':
    app.run(main)
