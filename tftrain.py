import logging
from time import strftime

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Flatten, Dense

from tf_vdc_loss import VDCLoss
from tf_wpdc_loss import WPDCLoss
from tfrecord_dataset_utils import parse_tfrecord


def main():
    loss = 'pdc'
    use_imagenet = True
    freeze = False
    machine = 'xps'
    base_lr = 2e-4

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler('output.log', mode='w'),
            logging.StreamHandler()
        ]
    )

    # step1: define the model structure
    if use_imagenet:
        base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(120, 120, 3), alpha=1.0,
                                                                 include_top=False,
                                                                 weights='imagenet', input_tensor=None, pooling=None)
        base_model.summary()

        if freeze:
            for layer in base_model.layers:
                layer.trainable = False

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.1))
        model.add(Dense(62, activation='softmax'))
    else:
        model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(120, 120, 3), alpha=1.0, include_top=True,
                                                            weights=None, input_tensor=None, pooling=None, classes=62)

    model.summary()

    # step2: optimization: loss and optimization method
    if loss == 'vdc':
        criterion = VDCLoss()
    elif loss == 'wpdc':
        criterion = WPDCLoss()
    else:
        criterion = keras.losses.MeanSquaredError()

    # optimizer = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum, decay=args.weight_decay, nesterov=True)
    optimizer = keras.optimizers.Adam(learning_rate=base_lr)

    # step3: load dataset
    train_dataset = tf.data.TFRecordDataset('train_aug.tfrecord')
    train_dataset = train_dataset.map(lambda x: parse_tfrecord(x, 120))
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size=128)

    val_dataset = tf.data.TFRecordDataset('test_aug.tfrecord')
    val_dataset = val_dataset.map(lambda x: parse_tfrecord(x, 120))
    val_dataset = val_dataset.shuffle(buffer_size=1024)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size=32)

    # step4: run

    time_str = strftime("%Y-%m-%d-%H%M%S")

    prefix = [machine]
    if use_imagenet:
        prefix.append('imagenet')
        if freeze:
            prefix.append('fixed')

    prefix.append(loss)
    prefix.append(time_str)
    prefix = '_'.join(prefix)

    callbacks = [
        ReduceLROnPlateau(verbose=1, patience=12),
        EarlyStopping(patience=20, verbose=1),
        ModelCheckpoint('checkpoints/' + prefix + '-{epoch}.tf', verbose=1,
                        save_best_only=False, save_weights_only=True),
        TensorBoard(log_dir='logs/' + prefix + '-tensorboard_logdir')
    ]

    model.compile(optimizer=optimizer, loss=criterion)
    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=100,
              callbacks=callbacks,
              workers=8,
              initial_epoch=0)


if __name__ == '__main__':
    main()
