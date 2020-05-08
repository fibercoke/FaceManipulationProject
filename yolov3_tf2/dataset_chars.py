import tensorflow as tf
from absl.flags import FLAGS


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = (x_train-127.5) / 128
    return x_train


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    'image/label': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    # 'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    # 'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    # 'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    # 'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    # 'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    y_train = x['image/label']
    return x_train, y_train


def load_tfrecord_dataset(file_pattern, size=32):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, size))