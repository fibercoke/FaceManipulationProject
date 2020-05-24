from time import strftime
from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
import yolov3_tf2.dataset_plate as dataset


flags.DEFINE_string('train_dataset_path', './data/out-training.tfrecord', 'path to training dataset file')
flags.DEFINE_string('test_dataset_path', './data/out-validation.tfrecord', 'path to validation dataset file')
flags.DEFINE_string('classes', './data/wddb_classes.names', 'path to output class file')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('prefix', 'yolov3_tiny_wddb_step2',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 250, 'number of epochs')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_boolean('restore', True, 'continue training?')
flags.DEFINE_string('restore_weights', './checkpoints/yolov3_tiny_wddb20200524-110249_028.ckpt', 'restore path')



def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
    if FLAGS.restore:
        model.load_weights(FLAGS.restore_weights).expect_partial()

    train_dataset = dataset.load_tfrecord_dataset(FLAGS.train_dataset_path, FLAGS.classes, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_tfrecord_dataset(FLAGS.test_dataset_path, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    optimizer = tf.keras.optimizers.SGD()
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss)

    model.summary()

    time_str = strftime("%Y%m%d-%H%M%S")
    prefix = FLAGS.prefix + time_str

    callbacks = [
        ReduceLROnPlateau(patience=10, verbose=1, factor=0.2),
        EarlyStopping(patience=25, verbose=1, restore_best_weights=True),
        ModelCheckpoint(os.path.join('checkpoints', prefix + '_{epoch:03d}.ckpt'),
                        verbose=1,
                        save_weights_only=True),
        TensorBoard(log_dir=os.path.join('logs', prefix))
    ]

    history = model.fit(train_dataset, epochs=FLAGS.epochs, callbacks=callbacks, validation_data=val_dataset)


if __name__ == '__main__':
    app.run(main)
