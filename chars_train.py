from time import strftime

import tensorflow as tf
import yolov3_tf2.dataset_chars as dataset
from absl import app, flags
from absl.flags import FLAGS
from chars_model import CModel
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

DATA_SIZE = 13163
VAL_SIZE = int(DATA_SIZE * 0.20)
time_str = strftime("%Y-%m-%d-%H%M%S")

flags.DEFINE_string('ckpt_name',
                    'chars_checkpoints/my_resnet101v2_train_acc_{val_accuracy:.4f}_{epoch}_%s.tf' % time_str,
                    'checkpoint name')
flags.DEFINE_string('tb_log_dir', 'chars_logs/chars/my_resnet101v2_%s' % time_str, 'tensorboard log dir')
flags.DEFINE_integer('size', 32, 'size of each character should be resize to')
flags.DEFINE_integer('epochs', 200, 'epoch num')
flags.DEFINE_integer('batch_size', 32, 'size of each batch')
flags.DEFINE_string('dataset_path', './data/chars_data.tfrecord', 'path to output dataset file')
flags.DEFINE_string('train_dataset_path', './data/my_chars_data_train.tfrecord', 'path to output dataset file')
flags.DEFINE_string('test_dataset_path', './data/my_chars_data_test.tfrecord', 'path to output dataset file')
flags.DEFINE_string('classes_path', './data/chars_data.names', 'path to output class file')
flags.DEFINE_bool('on_my_data', True, 'is on my data')


def main(_argv):
    if FLAGS.on_my_data:
        train_dataset = dataset.load_tfrecord_dataset(FLAGS.train_dataset_path, size=FLAGS.size)
        train_dataset = train_dataset.map(lambda x, y: (dataset.transform_images(x, FLAGS.size), y))
        train_dataset = train_dataset.shuffle(buffer_size=1024)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(FLAGS.batch_size)

        val_dataset = dataset.load_tfrecord_dataset(FLAGS.test_dataset_path, size=FLAGS.size)
        val_dataset = val_dataset.map(lambda x, y: (dataset.transform_images(x, FLAGS.size), y))
        val_dataset = val_dataset.shuffle(buffer_size=1024)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(FLAGS.batch_size)
    else:
        all_dataset = dataset.load_tfrecord_dataset(FLAGS.dataset_path, size=FLAGS.size)
        all_dataset = all_dataset.map(lambda x, y: (dataset.transform_images(x, FLAGS.size), y))
        all_dataset = all_dataset.shuffle(buffer_size=1024)
        val_dataset, train_dataset = all_dataset.take(VAL_SIZE), all_dataset.skip(VAL_SIZE)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(FLAGS.batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(FLAGS.batch_size)

    class_map = {name: idx for idx, name in enumerate(open(FLAGS.classes_path).read().splitlines())}
    class_num = len(class_map)
    model = CModel(FLAGS.size, class_num, 'imagenet')

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', verbose=1, patience=12),
        EarlyStopping(monitor='val_accuracy', patience=20, verbose=1),
        ModelCheckpoint(FLAGS.ckpt_name, verbose=1, monitor='val_accuracy',
                        save_best_only=True, save_weights_only=True, mode='max'),
        TensorBoard(log_dir=FLAGS.tb_log_dir)
    ]

    history = model.fit(train_dataset,
                        epochs=FLAGS.epochs,
                        callbacks=callbacks,
                        validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
