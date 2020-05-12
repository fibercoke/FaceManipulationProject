import logging
from time import strftime

import logging
from time import strftime

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Flatten, Dense

from tf_vdc_loss import VDCLoss
from tfrecord_dataset_utils import parse_tfrecord


def main():
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
    use_imagenet = True
    if use_imagenet:
        base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(120, 120, 3), alpha=1.0,
                                                                 include_top=False,
                                                                 weights='imagenet', input_tensor=None, pooling=None)
        base_model.summary()

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
    criterion = VDCLoss()
    # criterion = keras.losses.MeanSquaredError()
    # if args.loss.lower() == 'wpdc':
    #    print(args.opt_style)
    #    criterion = WPDCLoss(opt_style=args.opt_style).cuda()
    #    logging.info('Use WPDC Loss')
    # elif args.loss.lower() == 'vdc':
    #    criterion = VDCLoss(opt_style=args.opt_style).cuda()
    #    logging.info('Use VDC Loss')
    # elif args.loss.lower() == 'pdc':
    #    criterion = nn.MSELoss(size_average=args.size_average).cuda()
    #    logging.info('Use PDC loss')
    # else:
    #    raise Exception(f'Unknown Loss {args.loss}')

    # optimizer = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum, decay=args.weight_decay, nesterov=True)
    optimizer = keras.optimizers.Adam()
    # step 2.1 resume
    # if args.resume:
    #    if Path(args.resume).is_file():
    #        logging.info(f'=> loading checkpoint {args.resume}')
    #
    #        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
    #        # checkpoint = torch.load(args.resume)['state_dict']
    #        model.load_state_dict(checkpoint)
    #
    #    else:
    #        logging.info(f'=> no checkpoint found at {args.resume}')

    # step3: data
    # normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization
    # train_x = tf.data.TextLineDataset(args.filelists_train).map(read_img(args.root))
    # train_y = tf.data.Dataset.from_tensor_slices(_load(args.param_fp_train))
    # val_x = tf.data.TextLineDataset(args.filelists_val).map(read_img(args.root))
    # val_y = tf.data.Dataset.from_tensor_slices(_load(args.param_fp_val))
    # train_dataset = tf.data.Dataset.zip((train_x, train_y))\
    train_dataset = tf.data.TFRecordDataset('train_aug.tfrecord')
    train_dataset = train_dataset.map(lambda x: parse_tfrecord(x, 120))
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size=128)
    # val_dataset = tf.data.Dataset.zip((val_x, val_y))\

    val_dataset = tf.data.TFRecordDataset('test_aug.tfrecord')
    val_dataset = val_dataset.map(lambda x: parse_tfrecord(x, 120))
    val_dataset = val_dataset.shuffle(buffer_size=1024)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size=32)
    # IPython.embed()

    # model.load_weights('checkpoints/xps_first').expect_partial()

    # step4: run

    time_str = strftime("%Y-%m-%d-%H%M%S")
    callbacks = [
        ReduceLROnPlateau(verbose=1, patience=12),
        EarlyStopping(patience=20, verbose=1),
        ModelCheckpoint('checkpoints/imagenet_xps_first_{epoch}.tf', verbose=1,
                        save_best_only=False, save_weights_only=True),
        TensorBoard(log_dir="logs/imagenet_tensorboard_log_" + time_str)
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
