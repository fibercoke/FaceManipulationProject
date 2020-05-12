import tensorflow as tf

IMAGE_FEATURE_MAP = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/label': tf.io.FixedLenFeature([62], tf.float32)
}


def parse_tfrecord(tfrecord, size=120):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    y_train = x['image/label']
    return x_train, y_train
