from time import strftime
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
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


flags.DEFINE_string('dataset', './data/train.tfrecord', 'path to dataset')
flags.DEFINE_string('val_dataset', './data/test.tfrecord', 'path to validation dataset')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/plate.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 250, 'number of epochs')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')


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

    train_dataset = dataset.load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_tfrecord_dataset(FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss)

    model.summary()

    timestr = strftime("%Y%m%d-%H%M%S")

    callbacks = [
        ReduceLROnPlateau(patience=12, verbose=1, factor=0.2),
        EarlyStopping(patience=25, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_tiny_train_' + timestr + '_{epoch:03d}.tf',
                        verbose=1,
                        save_weights_only=True),
        TensorBoard(log_dir='logs/yolov3_tiny_train_' + timestr)
    ]

    history = model.fit(train_dataset, epochs=FLAGS.epochs, callbacks=callbacks, validation_data=val_dataset)



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
