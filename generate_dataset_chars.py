import os

import tensorflow as tf
import tqdm
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('dataset_path', './data/my_chars_fix_test_data/', 'path to data dir')
flags.DEFINE_string('dataset_output_path', './data/my_chars_fix_data_test.tfrecord', 'path to output dataset file')
flags.DEFINE_string('classes_output_path', './data/my_chars_fix_data_test.names', 'path to output class file')


def build_example(img_path, label):
    img_raw = open(img_path, 'rb').read()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
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
